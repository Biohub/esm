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

"""Distributed OuterProductMean for ESMFold2's MSA encoder (inference-only).

The serial op maps an MSA representation m (B, L, M, d_msa) into a pair update
(B, L, L, d_pair):

    outer[b,i,j] = sum_m a[b,i,m] (x) b[b,j,m]          # einsum "bimc,bjmd->bijcd"
    z = Wout(outer) / n_valid                            # divide_outer_before_proj=False

Under 2D context parallelism the pair z is sharded ``(Shard(0), Shard(1),
Shard(2))`` — row token i on cp_axis_0, col token j on cp_axis_1. The MSA m is
sharded ``(Shard(0), Shard(1), Replicate())`` — token L on cp_axis_0, MSA depth
M replicated on cp_axis_1.

Because M is replicated, the contraction over m is fully local. The tile owned
by rank (p, q) needs row block p (held locally) and col block q. Col block q is
exactly the row block held by the transpose peer (q, p), so a single transpose
of the b operand (and the mask) suffices — no ring rotation is required (unlike
boltz-cp, which shards the contracted dimension and therefore must ring-reduce).
"""

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Shard

from esm.models.esmfold2.distributed.comm import TransposeComm
from esm.models.esmfold2.distributed.model.layers.layernorm import (
    LayerNormParamsReplicated,
)
from esm.models.esmfold2.distributed.model.layers.linear import LinearParamsReplicated
from esm.models.esmfold2.layers import OuterProductMean as SerialOuterProductMean


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class OuterProductMeanDistributed(nn.Module):
    """Distributed (transpose-based) OuterProductMean.

    Parameters
    ----------
    layer:
        The serial OuterProductMean to distribute.
    dist_manager:
        DistributedManager with the CP group / subgroups set up.
    """

    def __init__(self, layer: SerialOuterProductMean, dist_manager) -> None:
        super().__init__()
        if not isinstance(layer, SerialOuterProductMean):
            raise TypeError(
                f"layer must be OuterProductMean, got {type(layer).__name__}"
            )
        self.device_mesh = dist_manager.device_mesh_subgroups
        self.d_hidden = layer.d_hidden
        self.divide_outer_before_proj = layer.divide_outer_before_proj

        self.norm = LayerNormParamsReplicated(layer.norm, self.device_mesh)
        self.W = LinearParamsReplicated(layer.W, self.device_mesh)
        self.Wout = LinearParamsReplicated(layer.Wout, self.device_mesh)

        # Transpose (i,j) <-> (j,i) to fetch the column-token block.
        self.transpose = TransposeComm(
            dist_manager.group["cp"], dist_manager.layout_subgroups["cp"]
        )

    def forward(self, m: DTensor, msa_attention_mask: DTensor) -> DTensor:
        mesh = self.device_mesh
        dh = self.d_hidden
        L = m.shape[1]

        m_norm = self.norm(m)
        x = self.W(m_norm)  # (B, L, M, 2*d_hidden), placements (S0, S1, R)
        x = x * msa_attention_mask.unsqueeze(-1).to(x.dtype)

        x_local = x.to_local()
        a_local, b_local = torch.chunk(x_local, 2, dim=-1)
        a_local = a_local.contiguous()
        b_local = b_local.contiguous()
        mask_local = msa_attention_mask.to_local().to(a_local.dtype)  # (B, sL, M)

        # Fetch the column-token block of b and the mask via a single transpose.
        packed = torch.cat([b_local, mask_local.unsqueeze(-1)], dim=-1).contiguous()
        recv = self.transpose.enqueue_to_dispatch(packed)
        self.transpose.wait_until_finished()
        b_q = recv[..., :dh].contiguous()
        mask_q = recv[..., dh].contiguous()  # (B, sL, M)

        outer = torch.einsum("bimc,bjmd->bijcd", a_local, b_q).flatten(-2).contiguous()
        n_valid_local = (
            torch.einsum("bim,bjm->bij", mask_local, mask_q)
            .unsqueeze(-1)
            .clamp(min=1.0)
            .contiguous()
        )

        b_size = outer.shape[0]
        c_out = outer.shape[-1]
        z_shape = torch.Size((b_size, L, L, c_out))
        z_dt = DTensor.from_local(
            outer,
            device_mesh=mesh,
            placements=[Shard(0), Shard(1), Shard(2)],
            shape=z_shape,
            stride=_contiguous_strides(tuple(z_shape)),
        )
        nv_shape = torch.Size((b_size, L, L, 1))
        n_valid_dt = DTensor.from_local(
            n_valid_local,
            device_mesh=mesh,
            placements=[Shard(0), Shard(1), Shard(2)],
            shape=nv_shape,
            stride=_contiguous_strides(tuple(nv_shape)),
        )

        if self.divide_outer_before_proj:
            return self.Wout(z_dt / n_valid_dt)
        return self.Wout(z_dt) / n_valid_dt
