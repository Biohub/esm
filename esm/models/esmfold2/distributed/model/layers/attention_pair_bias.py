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

"""Distributed token self-attention with a 2-D-sharded pair bias (diffusion).

This is the *core attention mechanism* for distributing ESMFold2's diffusion
token transformer (``AttentionPairBias``). It shards the expensive ``L×L``
attention over the query-row (``cp_axis_0``) axis, which is the diffusion's
memory ceiling; the cheap 1-D token reprs and the atom encoder/decoder stay
replicated (see ``fix_9x_#2.md``).

Two strategies, mirroring the MSA pair-averaging design:

* ``"gather"`` (implemented here): query rows are sharded on ``cp_axis_0``;
  ``k``/``v`` and the bias *row* are gathered along the key/column axis, then a
  single local softmax attention runs over the full key axis — **bit-exact**
  with the serial op. Per-rank attention/bias is ``L²·H / P`` (sharded on
  ``cp_axis_0``), down from the full ``L²``. The gathered ``k``/``v`` are the
  cheap 1-D ``L·H·D``.
* ``"ring"`` (future): also shard the key axis on ``cp_axis_1`` and ring k/v/z
  with online softmax (``AttentionPairBiasComm`` + ``tiled_softmax_attention_update``),
  removing the full-key-axis gather. A memory optimisation over ``"gather"``.

Inference-only. The ``q``/``k``/``v`` token reprs use placements
``(Shard(0), Shard(1), Replicate())`` (rows on ``cp_axis_0``); the pair bias uses
``(Shard(0), Shard(1), Shard(2))`` (the trunk's 2-D pair sharding).
"""

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard

_ROW_PL = [Shard(0), Shard(1), Replicate()]  # token rows on cp_axis_0
_ROW_GATHERED_PL = [Shard(0), Replicate(), Replicate()]  # key axis gathered
_PAIR_PL = [Shard(0), Shard(1), Shard(2)]  # 2-D pair sharding
_PAIR_ROWS_PL = [Shard(0), Shard(1), Replicate()]  # bias rows on cp0, cols gathered


def attention_pair_bias_gather(
    q_dt: DTensor,
    k_dt: DTensor,
    v_dt: DTensor,
    bias_dt: DTensor,
    scale: float,
    key_mask_dt: DTensor | None = None,
) -> DTensor:
    """Self-attention with an additive per-head pair bias, "gather" strategy.

    Bit-exact with the serial reference::

        logits[b,i,j,h] = scale * (q[b,i,h,:] . k[b,j,h,:]) + bias[b,i,j,h]
        attn            = softmax_j(logits)          # over keys j
        o[b,i,h,:]      = sum_j attn[b,i,j,h] v[b,j,h,:]

    Parameters
    ----------
    q_dt, k_dt, v_dt:
        Token reprs ``(B, L, H, D)`` with placements ``(Shard(0), Shard(1),
        Replicate())`` — query/key rows sharded on ``cp_axis_0``.
    bias_dt:
        Per-head pair bias ``(B, L, L, H)`` with placements ``(Shard(0),
        Shard(1), Shard(2))`` — the trunk's 2-D pair sharding.
    scale:
        Attention logit scale (``head_dim**-0.5``).
    key_mask_dt:
        Optional key mask ``(B, L)`` (``True`` = keep), placements ``(Shard(0),
        Shard(1), Replicate())``. Masked keys get ``-inf`` logit.

    Returns
    -------
    Output ``(B, L, H, D)`` with placements ``(Shard(0), Shard(1), Replicate())``
    (query rows on ``cp_axis_0``).
    """
    mesh = q_dt.device_mesh

    # Query rows stay sharded on cp_axis_0; gather the key axis (cheap 1-D k/v)
    # and the bias *row* (rows block stays on cp_axis_0, columns gathered).
    q_local = q_dt.to_local()  # (B, Lq_local, H, D)
    k_full = k_dt.redistribute(mesh, _ROW_GATHERED_PL).to_local()  # (B, L, H, D)
    v_full = v_dt.redistribute(mesh, _ROW_GATHERED_PL).to_local()  # (B, L, H, D)
    bias_row = bias_dt.redistribute(
        mesh, _PAIR_ROWS_PL
    ).to_local()  # (B, Lq_local, L, H)

    # logits (B, Lq_local, L, H)
    logits = torch.einsum("bihd,bjhd->bijh", q_local, k_full) * scale
    logits = logits + bias_row.to(logits.dtype)

    if key_mask_dt is not None:
        key_mask = key_mask_dt.redistribute(mesh, _ROW_GATHERED_PL).to_local()  # (B, L)
        neg = torch.finfo(logits.dtype).min
        logits = logits + torch.where(key_mask.bool()[:, None, :, None], 0.0, neg).to(
            logits.dtype
        )

    attn = torch.softmax(logits, dim=2).to(v_full.dtype)  # over keys j
    o_local = torch.einsum("bijh,bjhd->bihd", attn, v_full)  # (B, Lq_local, H, D)

    return DTensor.from_local(
        o_local.contiguous(), device_mesh=mesh, placements=_ROW_PL
    )
