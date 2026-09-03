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

"""Distributed RowAttentionPooling for the ESMFold2 confidence head (#6, inference).

The serial op (``modeling_esmfold2_common.py:RowAttentionPooling``) pools the pair
``z`` (B, L, L, d_pair) over the column axis j into a single repr (B, L, d_single):

    scores = attn_proj(z).squeeze(-1)          # (B, L, L)
    scores = scores + mask_bias_over_j         # -1e9 where col j is padding
    weights = softmax(scores, dim=-1)          # over j
    pooled  = einsum("bnm,bnmd->bnd", weights, z)   # sum over j
    return out_proj(pooled)                    # (B, L, d_single)

Under 2D CP the pair is sharded ``(Shard(0), Shard(1), Shard(2))`` (row i on
cp_axis_0, col j on cp_axis_1). The softmax and the weighted sum are both over
j = cp_axis_1, so we do a **distributed softmax + weighted-sum reduction along the
column group** — never gathering ``z``'s columns:

  * attn_proj is channel-wise → local scores (B, sLi, sLj);
  * softmax over the full j: all-reduce MAX then SUM over cp_axis_1 (online stats);
  * pooled: local ``einsum`` over the local j, then all-reduce SUM over cp_axis_1.

The result is row-sharded on cp_axis_0, replicated on cp_axis_1 — returned as a
``(Shard(0), Shard(1), Replicate())`` single-repr DTensor.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from esm.models.esmfold2.distributed.model.layers.linear import LinearParamsReplicated
from esm.models.esmfold2.layers import RowAttentionPooling as SerialRowAttentionPooling


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class RowAttentionPoolingDistributed(nn.Module):
    """Distributed RowAttentionPooling (inference-only).

    ``forward`` takes the pair DTensor ``z`` ``(Shard(0), Shard(1), Shard(2))`` and
    the full token mask ``(B, L)``; returns the single-repr DTensor
    ``(Shard(0), Shard(1), Replicate())`` of shape ``(B, L, d_single)``.
    """

    def __init__(self, layer: SerialRowAttentionPooling, dist_manager) -> None:
        super().__init__()
        if not isinstance(layer, SerialRowAttentionPooling):
            raise TypeError(
                f"layer must be RowAttentionPooling, got {type(layer).__name__}"
            )
        self.device_mesh = dist_manager.device_mesh_subgroups
        self.attn_proj = LinearParamsReplicated(layer.attn_proj, self.device_mesh)
        self.out_proj = LinearParamsReplicated(layer.out_proj, self.device_mesh)
        # device_mesh dims: (dp, cp_axis_0, cp_axis_1); j is cp_axis_1 (col group).
        self.col_group = self.device_mesh.get_group(2)

    def forward(self, z: DTensor, mask: torch.Tensor) -> DTensor:
        mesh = self.device_mesh
        # Column-shard the token mask to match z's local columns j (cp_axis_1).
        mask_cols = distribute_tensor(
            mask.contiguous(), mesh, [Shard(0), Replicate(), Shard(1)]
        ).to_local()  # (B, sLj)

        scores = self.attn_proj(z).to_local().squeeze(-1)  # (B, sLi, sLj)
        neg = torch.full_like(scores, -1e9)
        scores = torch.where(mask_cols[:, None, :].bool(), scores, neg)

        # Distributed softmax over the full j (cp_axis_1).
        local_max = scores.amax(dim=-1, keepdim=True)  # (B, sLi, 1)
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=self.col_group)
        e = torch.exp(scores - local_max)
        local_sum = e.sum(dim=-1, keepdim=True)  # (B, sLi, 1)
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=self.col_group)
        weights = e / local_sum  # (B, sLi, sLj) — globally normalized over j

        # Weighted sum over the local j, then reduce across the column group.
        z_local = z.to_local()  # (B, sLi, sLj, d_pair)
        pooled = torch.einsum("bij,bijd->bid", weights, z_local).contiguous()
        dist.all_reduce(pooled, op=dist.ReduceOp.SUM, group=self.col_group)
        # pooled is now the full-j pool: row-sharded on cp_axis_0, identical across
        # cp_axis_1 -> wrap as (Shard(0), Shard(1), Replicate()).
        b_size, s_li, d_pair = pooled.shape
        full_shape = torch.Size((b_size, z.shape[1], d_pair))
        pooled_dt = DTensor.from_local(
            pooled,
            device_mesh=mesh,
            placements=[Shard(0), Shard(1), Replicate()],
            shape=full_shape,
            stride=_contiguous_strides(tuple(full_shape)),
        )
        return self.out_proj(pooled_dt)  # (B, L, d_single), (S0, S1, R)
