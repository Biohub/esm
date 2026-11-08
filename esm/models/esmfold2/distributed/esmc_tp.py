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

"""Tensor-parallel ESM-C over the CP ranks (inference-only).

The replicated ESM-C 6B forward is the dominant per-rank floor (~19.8 GB live,
identical on every rank). This module shards its SwiGLU MLP, ~67% of ESM-C's
params, across the CP ranks.

fc1 is column-parallel: each rank keeps its slice of both gate and up. fc2 is
row-parallel: each rank keeps the matching input columns. One all-reduce after
fc2 restores the full, replicated block output. ``ffn_hidden`` (6912) divides
4/9/16, so this works at every CP grid.

Attention TP (blocked to TP=4 by n_heads=40) is not implemented.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from esm.models.esmc.layers import EsmcLayerNormMLP
from esm.models.esmfold2.fp8 import Fp8LayerNormMLP, Fp8Linear


class AllReduced(nn.Module):
    """Runs one rank's MLP shard and sums the partial outputs across ranks."""

    def __init__(self, shard: nn.Module, tp_group: dist.ProcessGroup) -> None:
        super().__init__()
        self.shard = shard
        self.tp_group = tp_group

    def forward(self, x: Tensor) -> Tensor:
        out = self.shard(x)
        dist.all_reduce(out, group=self.tp_group)
        return out


def _gate_up_rows(fc1: Tensor, ffn: int, lo: int, hi: int) -> Tensor:
    """fc1 is packed ``[gate; up]``, so a shard takes the same rows from each half."""
    return torch.cat([fc1[lo:hi], fc1[ffn + lo : ffn + hi]])


def _shard_bf16(src: EsmcLayerNormMLP, lo: int, hi: int) -> EsmcLayerNormMLP:
    with torch.device("meta"):
        shard = EsmcLayerNormMLP(src.hidden_size, hi - lo, src.eps)
    shard.layer_norm_weight = src.layer_norm_weight
    shard.layer_norm_bias = src.layer_norm_bias
    ffn = src.ffn_hidden_size
    shard.fc1_weight = nn.Parameter(
        _gate_up_rows(src.fc1_weight, ffn, lo, hi), requires_grad=False
    )
    shard.fc2_weight = nn.Parameter(
        src.fc2_weight[:, lo:hi].contiguous(), requires_grad=False
    )
    return shard


def _shard_fp8(src: Fp8LayerNormMLP, lo: int, hi: int) -> Fp8LayerNormMLP:
    # Scales are per output row, so a column slice of fc2 keeps its scales.
    ffn = src.fc2.in_features
    fc1 = Fp8Linear(
        _gate_up_rows(src.fc1.weight_fp8, ffn, lo, hi),
        _gate_up_rows(src.fc1.weight_scale, ffn, lo, hi),
    )
    fc2 = Fp8Linear(src.fc2.weight_fp8[:, lo:hi].contiguous(), src.fc2.weight_scale)
    return Fp8LayerNormMLP(
        src.layer_norm_weight, src.layer_norm_bias, src.eps, fc1, fc2
    )


def tp_shard_esmc_mlp(model: nn.Module, dist_manager, tp_group=None) -> int:
    """Replace every ESM-C block's SwiGLU MLP with its tensor-parallel shard
    over ``tp_group`` (default: the full CP process group). Returns the number of
    blocks sharded. No-op (returns 0) if ESM-C isn't loaded.
    """
    esmc = getattr(model, "esmc", None)
    if esmc is None:
        return 0

    if tp_group is None:
        tp_group = dist_manager.group["cp"]
    tp_size = dist.get_world_size(tp_group)
    rank = dist.get_rank(tp_group)
    if tp_size == 1:
        return 0

    n = 0
    for block in esmc.transformer.blocks:
        ffn = block.ffn
        width = (
            ffn.fc2.in_features
            if isinstance(ffn, Fp8LayerNormMLP)
            else ffn.ffn_hidden_size
        )
        if width % tp_size:
            raise ValueError(
                f"ESM-C ffn_hidden={width} not divisible by tp_size={tp_size}"
            )
        lo, hi = rank * (width // tp_size), (rank + 1) * (width // tp_size)
        if isinstance(ffn, Fp8LayerNormMLP):
            shard: nn.Module = _shard_fp8(ffn, lo, hi)
        elif isinstance(ffn, EsmcLayerNormMLP):
            shard = _shard_bf16(ffn, lo, hi)
        else:
            raise TypeError(f"cannot shard block.ffn of type {type(ffn).__name__}")
        block.ffn = AllReduced(shard, tp_group)
        n += 1
    return n
