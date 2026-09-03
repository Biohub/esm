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

"""Distributed LM->pair builder for ESMFold2 (Fix #14, inference-only).

The serial ``LanguageModelShim`` turns the ESM-C hidden states ``[B, L, 81,
d_model]`` into the pair representation ``lm_z`` ``[B, L, L, d_z]``. The L×L
blow-up happens entirely inside ``SingleToPair``:

    x = downproject(x)                                 # [B, L, dp]
    feat = cat([x_i * x_j, x_i - x_j], dim=-1)         # [B, L, L, 2*dp]  <-- the floor
    lm_z = output_mlp(feat)                            # [B, L, L, d_z]

Materialising that full ``[B, L, L, *]`` on every rank is the binding per-rank
peak measured by the T0 phase sweep (the ``language_model`` phase, identical at
2×2 and 4×4 -> unsharded). This module builds ``lm_z`` directly as a 2-D-sharded
``(Shard(0), Shard(1), Shard(2))`` DTensor (row token i on cp_axis_0, col token j
on cp_axis_1), so the L×L tensor is never resident full on any rank.

The outer op ``f(x_i, x_j)`` mirrors ``OuterProductMeanDistributed``: the row
block ``x_i`` is held locally; the column block ``x_j`` is fetched with a single
``TransposeComm`` (the (q,p) peer's row block is exactly rank (p,q)'s column
block on a square grid). All channel-wise ops (downproject, output_mlp, the
trailing LayerNorm) run on the local shard via replicated params — only the
outer op needs communication.
"""

from math import lcm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from esm.models.esmfold2.distributed.comm import TransposeComm
from esm.models.esmfold2.distributed.model.layers.layernorm import (
    LayerNormParamsReplicated,
)
from esm.models.esmfold2.distributed.model.layers.linear import LinearParamsReplicated
from esm.models.esmfold2.layers import LanguageModelShim as SerialLanguageModelShim
from esm.models.esmfold2.layers import SingleToPair as SerialSingleToPair

_PAIR_PLACEMENTS = [Shard(0), Shard(1), Shard(2)]
_TOKEN_PLACEMENTS = [Shard(0), Shard(1), Replicate()]


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class SingleToPairDistributed(nn.Module):
    """Distributed (transpose-based) ``SingleToPair``.

    ``forward`` takes a per-token DTensor ``x`` with placements
    ``(Shard(0), Shard(1), Replicate())`` (token L sharded on cp_axis_0,
    replicated on cp_axis_1) and returns the pair DTensor with placements
    ``(Shard(0), Shard(1), Shard(2))``.
    """

    def __init__(self, layer: SerialSingleToPair, dist_manager) -> None:
        super().__init__()
        if not isinstance(layer, SerialSingleToPair):
            raise TypeError(f"layer must be SingleToPair, got {type(layer).__name__}")
        self.device_mesh = dist_manager.device_mesh_subgroups

        self.downproject = LinearParamsReplicated(layer.downproject, self.device_mesh)
        # output_mlp = Sequential(Linear, GELU, Linear)
        self.out_in = LinearParamsReplicated(layer.output_mlp[0], self.device_mesh)  # ty:ignore[invalid-argument-type]
        self.gelu = layer.output_mlp[1]
        self.out_proj = LinearParamsReplicated(layer.output_mlp[2], self.device_mesh)  # ty:ignore[invalid-argument-type]

        # (i,j) <-> (j,i) transpose to fetch the column-token block.
        self.transpose = TransposeComm(
            dist_manager.group["cp"], dist_manager.layout_subgroups["cp"]
        )

    def forward(self, x: DTensor) -> DTensor:
        mesh = self.device_mesh
        L = x.shape[1]

        x = self.downproject(x)  # (B, L, dp), placements (S0, S1, R)

        # Row block i is local; fetch column block j from the transpose peer.
        a_local = x.to_local().contiguous()  # (B, sL_i, dp)
        b_q = self.transpose.enqueue_to_dispatch(a_local)
        self.transpose.wait_until_finished()
        b_q = b_q.contiguous()  # (B, sL_j, dp)

        # cat([x_i * x_j, x_i - x_j]) on the local (sL_i, sL_j) tile — mirrors the
        # serial cat([x.unsqueeze(2) * x.unsqueeze(1), x.unsqueeze(2) - x.unsqueeze(1)]).
        ai = a_local.unsqueeze(2)  # (B, sL_i, 1, dp)
        bj = b_q.unsqueeze(1)  # (B, 1, sL_j, dp)
        feat_local = torch.cat([ai * bj, ai - bj], dim=-1).contiguous()

        b_size = feat_local.shape[0]
        c_feat = feat_local.shape[-1]
        feat_shape = torch.Size((b_size, L, L, c_feat))
        feat_dt = DTensor.from_local(
            feat_local,
            device_mesh=mesh,
            placements=_PAIR_PLACEMENTS,
            shape=feat_shape,
            stride=_contiguous_strides(tuple(feat_shape)),
        )

        out = self.out_in(feat_dt)  # (B, L, L, out_dim), (S0, S1, S2)
        # GELU is pointwise → run on the local shard, preserve the DTensor metadata.
        gelu_local = self.gelu(out.to_local())
        out = DTensor.from_local(
            gelu_local.contiguous(),
            device_mesh=mesh,
            placements=_PAIR_PLACEMENTS,
            shape=out.shape,
            stride=out.stride(),
        )
        return self.out_proj(out)


class LanguageModelShimDistributed(nn.Module):
    """Distributed ``LanguageModelShim`` producing a sharded ``lm_z`` DTensor.

    The per-token front end (``base_z_linear`` + the ``base_z_combine`` softmax
    mix) is cheap (per-token, ~MB) and runs replicated on each rank exactly as in
    the serial shim. Only the L×L ``SingleToPair`` blow-up and the trailing
    LayerNorm are distributed.

    ``forward`` returns ``(lm_z_dt, n_orig)``: a padded, 2-D-sharded pair DTensor
    and the original (pre-pad) token count. The token axis is padded to
    ``shard_factor = lcm(cp_axis_0, cp_axis_1)`` so the pair shards evenly — the
    same padding contract the recycle engine uses for ``z`` / ``z_init``. Padded
    rows/cols carry no signal (masked downstream by the padded pair mask; sliced
    to ``n_orig`` after any gather), so they need not be explicitly zeroed.
    """

    def __init__(self, layer: SerialLanguageModelShim, dist_manager) -> None:
        super().__init__()
        if not isinstance(layer, SerialLanguageModelShim):
            raise TypeError(
                f"layer must be LanguageModelShim, got {type(layer).__name__}"
            )
        self.device_mesh = dist_manager.device_mesh_subgroups
        # device_mesh is (dp, cp_axis_0, cp_axis_1); pad to lcm so the pair shards evenly.
        self.shard_factor = lcm(self.device_mesh.size(1), self.device_mesh.size(2))

        # Per-token front end runs replicated (kept as the serial submodules).
        self.base_z_linear = layer.base_z_linear
        self.base_z_combine = layer.base_z_combine

        # base_z_mlp = Sequential(SingleToPair, LayerNorm(d_z))
        self.single_to_pair = SingleToPairDistributed(layer.base_z_mlp[0], dist_manager)  # ty:ignore[invalid-argument-type]
        self.norm = LayerNormParamsReplicated(layer.base_z_mlp[1], self.device_mesh)  # ty:ignore[invalid-argument-type]

    def forward(self, hidden_states: torch.Tensor) -> tuple[DTensor, int]:
        # Per-token front end (replicated, full L — cheap, no L×L here).
        lm_single = self.base_z_linear(hidden_states)  # [B, L, 81, d_z]
        weights = self.base_z_combine.softmax(0)  # [81]
        lm_single = (weights @ lm_single).squeeze(-2)  # [B, L, d_z]

        n_orig = lm_single.shape[1]
        pad = (self.shard_factor - n_orig % self.shard_factor) % self.shard_factor
        if pad:
            lm_single = F.pad(lm_single, (0, 0, 0, pad))

        x_dt = distribute_tensor(
            lm_single.contiguous(), self.device_mesh, _TOKEN_PLACEMENTS
        )
        lm_z = self.single_to_pair(x_dt)  # (S0, S1, S2), [B, Lpad, Lpad, d_z]
        lm_z = self.norm(lm_z)
        return lm_z, n_orig
