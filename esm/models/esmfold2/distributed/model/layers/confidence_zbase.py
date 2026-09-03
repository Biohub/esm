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

"""Distributed z_base / s->z builder for the ESMFold2 confidence head (#6).

Reproduces the serial pre-trunk pair construction
(``modeling_esmfold2.py:ConfidenceHead.forward`` lines 187-216) keeping the pair a
sharded ``(Shard(0), Shard(1), Shard(2))`` DTensor:

    z_base  = z_norm(z) [+ rel_pos] [+ token_bonds]
            + s_to_z(s)[:, :, None]            # row i, broadcast over j
            + s_to_z_transpose(s)[:, None, :]  # col j, broadcast over i
            + s_to_z_prod_out(prod_in1(s)[:, :, None] * prod_in2(s)[:, None, :])
    pair    = z_base + dist_bin_pairwise_embed(distogram_bins)

Because ``s_inputs`` (hence ``s = s_inputs_norm(s_inputs)`` and all the per-token
projections) is replicated, the row term needs only the local cp_axis_0 rows and
the column term only the local cp_axis_1 cols — both obtained by slicing the cheap
full ``[B, L, d_pair]`` projection two ways (``distribute_tensor`` of a replicated
tensor = a local slice, no communication). So the whole build is **local and
bit-exact** (no transpose, no reduction): every term lands on the same
``(sLi, sLj)`` tile the pair owns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from esm.models.esmfold2.distributed.model.layers.layernorm import (
    LayerNormParamsReplicated,
)

_PAIR = [Shard(0), Shard(1), Shard(2)]
_ROW = [Shard(0), Shard(1), Replicate()]  # token L sharded on cp_axis_0 (rows i)
_COL = [Shard(0), Replicate(), Shard(1)]  # token L sharded on cp_axis_1 (cols j)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class ConfidenceZBaseDistributed(nn.Module):
    """Builds the confidence head's pre-trunk pair as a sharded DTensor.

    Reads the projection submodules off the serial ``ConfidenceHead``; only the
    pair-channel ``z_norm`` is wrapped (it runs on the sharded ``z``). The
    per-token projections run on the replicated ``s_inputs`` exactly as serial.
    """

    def __init__(self, layer, dist_manager) -> None:
        super().__init__()
        self.device_mesh = dist_manager.device_mesh_subgroups
        self.z_norm = LayerNormParamsReplicated(layer.z_norm, self.device_mesh)
        # Per-token / pair-channel ops: replicated params, run as-is.
        self.s_inputs_norm = layer.s_inputs_norm
        self.s_to_z = layer.s_to_z
        self.s_to_z_transpose = layer.s_to_z_transpose
        self.s_to_z_prod_in1 = layer.s_to_z_prod_in1
        self.s_to_z_prod_in2 = layer.s_to_z_prod_in2
        self.s_to_z_prod_out = layer.s_to_z_prod_out
        self.dist_bin_pairwise_embed = layer.dist_bin_pairwise_embed

    def forward(
        self,
        z: DTensor,
        s_inputs: torch.Tensor,
        distogram_bins: torch.Tensor,
        relative_position_encoding: torch.Tensor | None = None,
        token_bonds_encoding: torch.Tensor | None = None,
    ) -> DTensor:
        mesh = self.device_mesh
        L = z.shape[1]

        s = self.s_inputs_norm(s_inputs)  # full [B, L, d_inputs], replicated

        def row(t):  # local cp_axis_0 rows
            return distribute_tensor(t.contiguous(), mesh, _ROW).to_local()

        def col(t):  # local cp_axis_1 cols
            return distribute_tensor(t.contiguous(), mesh, _COL).to_local()

        a_row = row(self.s_to_z(s))  # (B, sLi, d_pair)
        b_col = col(self.s_to_z_transpose(s))  # (B, sLj, d_pair)
        p1_row = row(self.s_to_z_prod_in1(s))  # (B, sLi, d_pair)
        p2_col = col(self.s_to_z_prod_in2(s))  # (B, sLj, d_pair)

        # rel_pos / token_bonds may arrive already sharded (built block-local by
        # PairInitDistributed) — use the local shard directly; only a full tensor
        # needs the distribute-then-take-local round trip.
        def _pair_local(t):
            if isinstance(t, DTensor):
                return t.to_local()
            return distribute_tensor(t.contiguous(), mesh, _PAIR).to_local()

        zb = self.z_norm(z).to_local()  # (B, sLi, sLj, d_pair)
        if relative_position_encoding is not None:
            zb = zb + _pair_local(relative_position_encoding)
        if token_bonds_encoding is not None:
            zb = zb + _pair_local(token_bonds_encoding)

        zb = zb + a_row[:, :, None, :] + b_col[:, None, :, :]
        prod = p1_row[:, :, None, :] * p2_col[:, None, :, :]  # (B, sLi, sLj, d_pair)
        zb = zb + F.linear(prod, self.s_to_z_prod_out.weight)

        # distogram_bins may arrive already sharded (block cdist by
        # build_sharded_distogram_bins) — take the local shard directly.
        bins_local = (
            distogram_bins.to_local()
            if isinstance(distogram_bins, DTensor)
            else distribute_tensor(distogram_bins.contiguous(), mesh, _PAIR).to_local()
        )  # (B, sLi, sLj)
        zb = zb + self.dist_bin_pairwise_embed(bins_local)

        b_size = zb.shape[0]
        d_pair = zb.shape[-1]
        full_shape = torch.Size((b_size, L, L, d_pair))
        return DTensor.from_local(
            zb.contiguous(),
            device_mesh=mesh,
            placements=_PAIR,
            shape=full_shape,
            stride=_contiguous_strides(tuple(full_shape)),
        )
