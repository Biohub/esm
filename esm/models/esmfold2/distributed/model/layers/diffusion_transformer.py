# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Distributed diffusion token transformer for ESMFold2.

Distributes ``DiffusionTransformer`` (``AttentionPairBias`` +
``ConditionedTransitionBlock`` per block) by sharding the token row axis on
``cp_axis_0`` (the diffusion memory ceiling is the L×L token attention).

Design: every per-token operation — AdaLN, the q/k/v/gate/out projections, the
output gate, and the whole transition block — is *channel-wise* on the
row-sharded token repr, so it runs **as-is on the local shard** using the serial
submodule's (identically replicated) parameters. Only the attention score/context
(``q·k`` over the full key axis + the 2-D-sharded pair bias) needs cross-rank
communication, handled by :func:`attention_pair_bias_gather`. This keeps the code
a thin shell over the serial layer and bit-exact with it (modulo the gather,
which is lossless).

Token reprs (``a``, ``s``) are DTensors ``(B, L, d)`` with placements
``(Shard(0), Shard(1), Replicate())`` (rows on ``cp_axis_0``); the pair ``z`` is
``(B, L, L, d_pair)`` with ``(Shard(0), Shard(1), Shard(2))``. Inference-only,
single diffusion sample (``num_diffusion_samples`` handled by the caller/sampler).
"""

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate, Shard

from esm.models.esmfold2.distributed.model.layers.attention_pair_bias import (
    attention_pair_bias_gather,
)
from esm.models.esmfold2.layers import AttentionPairBias as SerialAttentionPairBias
from esm.models.esmfold2.layers import (
    ConditionedTransitionBlock as SerialConditionedTransitionBlock,
)
from esm.models.esmfold2.layers import (
    DiffusionTransformer as SerialDiffusionTransformer,
)

_ROW_PL = [Shard(0), Shard(1), Replicate()]
_PAIR_PL = [Shard(0), Shard(1), Shard(2)]


def _row_dtensor(local: torch.Tensor, mesh) -> DTensor:
    return DTensor.from_local(local.contiguous(), device_mesh=mesh, placements=_ROW_PL)


class AttentionPairBiasDistributed(nn.Module):
    """Distributed ``AttentionPairBias`` (standard, non-fused path).

    Holds the serial layer; runs its per-token ops on local shards and routes the
    attention through :func:`attention_pair_bias_gather`.
    """

    def __init__(self, layer: SerialAttentionPairBias) -> None:
        super().__init__()
        if not isinstance(layer, SerialAttentionPairBias):
            raise TypeError(
                f"layer must be AttentionPairBias, got {type(layer).__name__}"
            )
        self.layer = layer
        self.num_heads = layer.num_heads
        self.head_dim = layer.head_dim
        self.scale = layer.scale
        self.use_conditioning = hasattr(layer, "adaln")
        self.has_pair = hasattr(layer, "pair_bias_proj")

    def forward(
        self,
        a_dt: DTensor,
        s_dt: DTensor | None,
        z_dt: DTensor,
        key_mask_dt: DTensor | None = None,
    ) -> DTensor:
        mesh = a_dt.device_mesh
        layer = self.layer
        h, hd = self.num_heads, self.head_dim

        a_local = a_dt.to_local()
        b, l_loc = a_local.shape[0], a_local.shape[1]

        # --- per-token ops on the local shard (channel-wise) ---
        if s_dt is not None and self.use_conditioning:
            s_local = s_dt.to_local()
            x_local = layer.adaln(a_local, s_local)
        else:
            s_local = None
            x_local = layer.pre_norm(a_local)

        q_local = layer.q_proj(x_local).view(b, l_loc, h, hd)
        k_local, v_local = layer.kv_proj(x_local).chunk(2, dim=-1)
        k_local = k_local.reshape(b, l_loc, h, hd)
        v_local = v_local.reshape(b, l_loc, h, hd)
        g_local = torch.sigmoid(layer.g_proj(x_local)).view(b, l_loc, h, hd)

        # --- pair bias on the local (2-D-sharded) pair shard ---
        z_local = z_dt.to_local()  # (B, rows_loc, cols_loc, d_pair)
        bias_local = layer.pair_bias_proj(layer.pair_norm(z_local))  # (.., H)
        bias_dt = DTensor.from_local(
            bias_local.contiguous(), device_mesh=mesh, placements=_PAIR_PL
        )

        # --- distributed attention (gather strategy) ---
        q_dt = _row_dtensor(q_local, mesh)
        k_dt = _row_dtensor(k_local, mesh)
        v_dt = _row_dtensor(v_local, mesh)
        o_dt = attention_pair_bias_gather(
            q_dt, k_dt, v_dt, bias_dt, self.scale, key_mask_dt=key_mask_dt
        )

        # --- gate + output projection on the local shard ---
        ctx_local = (g_local * o_dt.to_local()).reshape(b, l_loc, h * hd)
        out_local = layer.out_proj(ctx_local)
        if s_local is not None and self.use_conditioning:
            out_local = torch.sigmoid(layer.out_gate(s_local)) * out_local
        return _row_dtensor(out_local, mesh)


class ConditionedTransitionDistributed(nn.Module):
    """Distributed ``ConditionedTransitionBlock`` — purely per-token, so just the
    serial block run on the local shard."""

    def __init__(self, layer: SerialConditionedTransitionBlock) -> None:
        super().__init__()
        if not isinstance(layer, SerialConditionedTransitionBlock):
            raise TypeError(
                f"layer must be ConditionedTransitionBlock, got {type(layer).__name__}"
            )
        self.layer = layer

    def forward(self, a_dt: DTensor, s_dt: DTensor | None) -> DTensor:
        mesh = a_dt.device_mesh
        a_local = a_dt.to_local()
        s_local = s_dt.to_local() if s_dt is not None else None
        return _row_dtensor(self.layer(a_local, s_local), mesh)


class DiffusionTransformerDistributed(nn.Module):
    """Distributed ``DiffusionTransformer``: per block ``x = x + attn(x,s,z); x =
    x + transition(x,s)``, matching the serial loop."""

    def __init__(self, transformer: SerialDiffusionTransformer) -> None:
        super().__init__()
        if not isinstance(transformer, SerialDiffusionTransformer):
            raise TypeError(
                f"transformer must be DiffusionTransformer, got {type(transformer).__name__}"
            )
        self.attn_blocks = nn.ModuleList(
            [AttentionPairBiasDistributed(b) for b in transformer.attn_blocks]  # ty:ignore[invalid-argument-type]
        )
        self.transition_blocks = nn.ModuleList(
            [ConditionedTransitionDistributed(b) for b in transformer.transition_blocks]  # ty:ignore[invalid-argument-type]
        )

    def forward(
        self,
        a_dt: DTensor,
        s_dt: DTensor | None,
        z_dt: DTensor,
        key_mask_dt: DTensor | None = None,
    ) -> DTensor:
        x = a_dt
        for attn, transition in zip(self.attn_blocks, self.transition_blocks):
            x = x + attn(x, s_dt, z_dt, key_mask_dt=key_mask_dt)
            x = x + transition(x, s_dt)
        return x
