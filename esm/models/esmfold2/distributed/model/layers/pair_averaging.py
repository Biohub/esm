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

"""Distributed MSAPairWeightedAveraging for ESMFold2's MSA encoder (inference).

The serial op (AF3 Algorithm 10) updates the MSA representation m using the
pair representation as attention bias:

    bias = compute_bias(pair)                              # (B, L, L, h)
    attn = softmax_j(masked(bias))                         # over col token j
    out[b,i,m,h,d] = (sum_j attn[b,i,j,h] v[b,j,m,h,d]) * gate[b,i,m,h,d]
    return Wout(out)

Under 2D CP the pair (and bias) is sharded ``(Shard(0), Shard(1), Shard(2))``
and m / v / gate are sharded ``(Shard(0), Shard(1), Replicate())`` (token L on
cp_axis_0, MSA depth M replicated). The softmax is over j = cp_axis_1, which is
sharded, and the value index is the (replicated) MSA depth with token on
cp_axis_0.

Two communication strategies (selectable via ``comm``):

* ``"gather"`` (default): ``DTensor.redistribute`` gathers the full j of the
  bias/mask and the full token axis of v, then the attention is computed
  locally over the full j in natural order — bit-exact with the serial op. The
  gathered buffers are small (``L²·h`` and ``L·M·c``, far below the sharded
  pair), so this is the better choice for typical grids / MSA depths.

* ``"ring"`` (boltz-style): never materialises the full j. ``v`` is transposed
  so block ``q`` aligns with ``bias[block p, block q]``, then both ring along
  the column axis (``comm_row``) accumulating with an online softmax
  (``tiled_softmax_attention_update``). Pays off only at larger grids (n >= 3)
  with deep MSAs, where the gathered buffers would be large. The online-softmax
  reassociation makes it close-but-not-bit-exact vs. the serial op.
"""

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate, Shard

from esm.models.esmfold2.distributed.comm import Ring2DComm
from esm.models.esmfold2.distributed.model.layers.layernorm import (
    LayerNormParamsReplicated,
)
from esm.models.esmfold2.distributed.model.layers.linear import LinearParamsReplicated
from esm.models.esmfold2.distributed.utils import tiled_softmax_attention_update
from esm.models.esmfold2.layers import (
    MSAPairWeightedAveraging as SerialMSAPairWeightedAveraging,
)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class MSAPairWeightedAveragingDistributed(nn.Module):
    """Distributed MSAPairWeightedAveraging (inference-only).

    Parameters
    ----------
    layer:
        The serial MSAPairWeightedAveraging to distribute.
    dist_manager:
        DistributedManager with the CP group / subgroups set up.
    comm:
        ``"gather"`` (default, bit-exact all-gather) or ``"ring"`` (boltz-style
        online-softmax ring).
    """

    def __init__(
        self, layer: SerialMSAPairWeightedAveraging, dist_manager, comm: str = "gather"
    ) -> None:
        super().__init__()
        if not isinstance(layer, SerialMSAPairWeightedAveraging):
            raise TypeError(
                f"layer must be MSAPairWeightedAveraging, got {type(layer).__name__}"
            )
        if comm not in ("gather", "ring"):
            raise ValueError(f"comm must be 'gather' or 'ring', got {comm!r}")
        self.device_mesh = dist_manager.device_mesh_subgroups
        self.comm_mode = comm
        self.n_heads = layer.n_heads
        self.head_width = layer.head_width

        self.norm_single = LayerNormParamsReplicated(
            layer.norm_single, self.device_mesh
        )
        # compute_bias is nn.Sequential(LayerNorm(d_pair), Linear(d_pair, n_heads))
        self.bias_norm = LayerNormParamsReplicated(
            layer.compute_bias[0],  # ty:ignore[invalid-argument-type]
            self.device_mesh,
        )
        self.bias_lin = LinearParamsReplicated(layer.compute_bias[1], self.device_mesh)  # ty:ignore[invalid-argument-type]
        self.Wv = LinearParamsReplicated(layer.Wv, self.device_mesh)
        self.Wgate = LinearParamsReplicated(layer.Wgate, self.device_mesh)
        self.Wout = LinearParamsReplicated(layer.Wout, self.device_mesh)

        if comm == "ring":
            # ring_v owns the value transpose + value column-ring; ring_bias
            # owns an independent bias column-ring (same schedule, separate
            # comm handles so both can be in flight together).
            self.ring_v = Ring2DComm(
                dist_manager.group["cp"],
                dist_manager.subgroups["cp"][0],
                dist_manager.layout_subgroups["cp"],
            )
            self.ring_bias = Ring2DComm(
                dist_manager.group["cp"],
                dist_manager.subgroups["cp"][0],
                dist_manager.layout_subgroups["cp"],
            )

    def forward(
        self, m: DTensor, pair: DTensor, pair_attention_mask: DTensor
    ) -> DTensor:
        if self.comm_mode == "ring":
            return self._forward_ring(m, pair, pair_attention_mask)
        return self._forward_gather(m, pair, pair_attention_mask)

    # -- shared projections ------------------------------------------------
    def _project(self, m: DTensor, pair: DTensor):
        msa_normed = self.norm_single(m)
        v_dt = self.Wv(msa_normed)  # (B, L, M, h*dh), (S0, S1, R)
        gate_dt = torch.sigmoid(self.Wgate(msa_normed))  # (B, L, M, h*dh)
        bias_dt = self.bias_lin(self.bias_norm(pair))  # (B, L, L, h), (S0, S1, S2)
        return v_dt, gate_dt, bias_dt

    def _finish(self, m: DTensor, o_local: torch.Tensor, gate_local: torch.Tensor):
        h, dh = self.n_heads, self.head_width
        b_size, s_l, m_depth = o_local.shape[0], o_local.shape[1], o_local.shape[2]
        o_local = (
            (o_local * gate_local).reshape(b_size, s_l, m_depth, h * dh).contiguous()
        )
        out_shape = torch.Size((b_size, m.shape[1], m_depth, h * dh))
        out_dt = DTensor.from_local(
            o_local,
            device_mesh=self.device_mesh,
            placements=[Shard(0), Shard(1), Replicate()],
            shape=out_shape,
            stride=_contiguous_strides(tuple(out_shape)),
        )
        return self.Wout(out_dt)

    # -- gather strategy (default, bit-exact) ------------------------------
    def _forward_gather(
        self, m: DTensor, pair: DTensor, pair_attention_mask: DTensor
    ) -> DTensor:
        mesh = self.device_mesh
        h, dh = self.n_heads, self.head_width
        v_dt, gate_dt, bias_dt = self._project(m, pair)

        bias_full = (
            bias_dt.redistribute(mesh, [Shard(0), Shard(1), Replicate()])
            .to_local()
            .contiguous()
        )  # (B, sL, L, h)
        mask_full = (
            pair_attention_mask.redistribute(mesh, [Shard(0), Shard(1), Replicate()])
            .to_local()
            .contiguous()
        )  # (B, sL, L)
        bias_full = bias_full.masked_fill(~mask_full.unsqueeze(-1).bool(), -1e5)
        attn = torch.softmax(bias_full, dim=-2)  # softmax over j

        v_full = (
            v_dt.redistribute(mesh, [Shard(0), Replicate(), Replicate()])
            .to_local()
            .contiguous()
        )  # (B, L, M, h*dh)

        b_size, s_l = attn.shape[0], attn.shape[1]
        l_full, m_depth = v_full.shape[1], v_full.shape[2]
        v_full = v_full.reshape(b_size, l_full, m_depth, h, dh)
        gate_local = gate_dt.to_local().reshape(b_size, s_l, m_depth, h, dh)

        o_local = torch.einsum("bijh,bjmhd->bimhd", attn, v_full)
        return self._finish(m, o_local, gate_local)

    # -- ring strategy (boltz-style online softmax) ------------------------
    def _forward_ring(
        self, m: DTensor, pair: DTensor, pair_attention_mask: DTensor
    ) -> DTensor:
        h, dh = self.n_heads, self.head_width
        n = self.device_mesh.size(1)  # CP axis size
        v_dt, gate_dt, bias_dt = self._project(m, pair)

        # Local blocks: bias[i in p, j in q], v[block p], mask[i in p, j in q].
        bias_local = bias_dt.to_local()  # (B, sLi, sLj, h)
        mask_local = pair_attention_mask.to_local()  # (B, sLi, sLj)
        # Pre-mask: each block's mask is co-located with its bias and rings
        # along with it, so masking once before the ring is correct.
        bias_local = bias_local.masked_fill(
            ~mask_local.unsqueeze(-1).bool(), -1e5
        ).contiguous()
        v_local = v_dt.to_local().contiguous()  # (B, sLp, M, h*dh)

        # Transpose v: block p -> block q (aligns v with the local bias block).
        v_q = self.ring_v.comm_2d_trans.enqueue_to_dispatch(v_local)
        self.ring_v.comm_2d_trans.wait_until_finished()

        b_size, s_li = bias_local.shape[0], bias_local.shape[1]
        m_depth = v_q.shape[2]
        bias_buf = [bias_local, torch.empty_like(bias_local)]
        v_buf = [v_q.contiguous(), torch.empty_like(v_q)]
        i_ready, i_recv = 0, 1
        o = lse = amax = None

        for k in range(n):
            b_blk = bias_buf[i_ready]
            v_blk = v_buf[i_ready]
            if k < n - 1:
                bias_buf[i_recv] = self.ring_bias.comm_row.enqueue_to_dispatch(
                    b_blk, bias_buf[i_recv]
                )
                v_buf[i_recv] = self.ring_v.comm_row.enqueue_to_dispatch(
                    v_blk, v_buf[i_recv]
                )

            amax_blk = b_blk.amax(dim=2, keepdim=True)  # (B, sLi, 1, h)
            lse_blk = torch.logsumexp(b_blk - amax_blk, dim=2, keepdim=True)
            p = torch.softmax(b_blk, dim=2)  # (B, sLi, sLj, h)
            v_blk_r = v_blk.reshape(b_size, v_blk.shape[1], m_depth, h, dh)
            o_blk = torch.einsum("bijh,bjmhd->bimhd", p, v_blk_r)  # (B, sLi, M, h, dh)

            # Arrange so the softmax-reduced axes (M, dh) are the trailing
            # feature and the per-(i, h) lse/amax broadcast over them.
            o_blk2 = o_blk.permute(0, 1, 3, 2, 4).reshape(b_size, s_li, h, m_depth * dh)
            lse_blk2 = lse_blk.permute(0, 1, 3, 2).reshape(b_size, s_li, h, 1)
            amax_blk2 = amax_blk.permute(0, 1, 3, 2).reshape(b_size, s_li, h, 1)
            o, lse, amax = tiled_softmax_attention_update(
                o_blk2, lse_blk2, amax_blk2, o, lse, amax
            )

            if k < n - 1:
                self.ring_bias.comm_row.wait_until_finished()
                self.ring_v.comm_row.wait_until_finished()
                i_ready ^= 1
                i_recv ^= 1

        # (B, sLi, h, M*dh) -> (B, sLi, M, h, dh)
        o_local = (
            o.reshape(b_size, s_li, h, m_depth, dh).permute(0, 1, 3, 2, 4).contiguous()  # ty:ignore[unresolved-attribute]
        )
        gate_local = gate_dt.to_local().reshape(b_size, s_li, m_depth, h, dh)
        return self._finish(m, o_local, gate_local)
