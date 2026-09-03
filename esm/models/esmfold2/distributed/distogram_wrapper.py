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

"""Distributed distogram head for ESMFold2 (inference-only).

The serial op is a single channel-wise Linear over the symmetrized pair:

    distogram_logits = distogram_head(z + z.transpose(-2, -3))   # (B, L, L, bins)

``z.transpose(-2, -3)`` swaps the two token axes (rows i <-> cols j). Under 2D CP
the pair is sharded ``(Shard(0), Shard(1), Shard(2))``; the (p, q) rank's
transposed tile ``zᵀ[i∈p, j∈q] = z[j∈q, i∈p]`` lives on the transpose peer
``(q, p)`` — one ``TransposeComm`` fetches that tile, and a local axis-swap
arranges it. The Linear is per-pair channel-wise → local. Only the small
``bins``-channel logits are gathered (vs. the full ``d_pair`` pair), so the full
``z`` (256 ch) is never re-gathered for the distogram.
"""

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Shard

from esm.models.esmfold2.distributed.comm import TransposeComm
from esm.models.esmfold2.distributed.model.layers.linear import LinearParamsReplicated

_PAIR = [Shard(0), Shard(1), Shard(2)]


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class DistogramHeadCPWrapper(nn.Module):
    """Distributed distogram head. ``forward_sharded`` takes the sharded pair
    DTensor (padded) + the original length and returns the full (sliced)
    ``distogram_logits`` — gathering only the ``bins``-channel output."""

    def __init__(self, distogram_head: nn.Linear, dist_manager) -> None:
        super().__init__()
        if not isinstance(distogram_head, nn.Linear):
            raise TypeError(
                f"distogram_head must be nn.Linear, got {type(distogram_head).__name__}"
            )
        self.device_mesh = dist_manager.device_mesh_subgroups
        self.head = LinearParamsReplicated(distogram_head, self.device_mesh)
        self.transpose = TransposeComm(
            dist_manager.group["cp"], dist_manager.layout_subgroups["cp"]
        )

    def forward_sharded(self, z_dt: DTensor, n_orig: int) -> torch.Tensor:
        mesh = self.device_mesh
        # Match serial: distogram runs on z.float().
        z_local = z_dt.to_local().float().contiguous()  # (B, sLi, sLj, c) = z[i,j]

        recv = self.transpose.enqueue_to_dispatch(z_local)
        self.transpose.wait_until_finished()
        # recv = z's (q,p) tile = z[a∈q, b∈p]; zᵀ[i∈p, j∈q] = z[j,i] = recv[j, i].
        zT_local = recv.transpose(1, 2).contiguous()  # (B, sLi, sLj, c)

        sym_local = z_local + zT_local
        L = z_dt.shape[1]
        b_size, _, _, c = sym_local.shape
        full_shape = torch.Size((b_size, L, L, c))
        sym_dt = DTensor.from_local(
            sym_local,
            device_mesh=mesh,
            placements=_PAIR,
            shape=full_shape,
            stride=_contiguous_strides(tuple(full_shape)),
        )
        logits_dt = self.head(sym_dt)  # (S0, S1, S2) [B, L, L, bins]
        return logits_dt.full_tensor()[:, :n_orig, :n_orig, :].contiguous()
