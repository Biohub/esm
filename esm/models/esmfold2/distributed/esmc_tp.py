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

The replicated ESM-C 6B forward is the dominant per-rank floor (T0 re-hook:
~19.8 GB live, identical on every rank). Tensor parallelism shards its weights
AND intra-block activations across the CP ranks.

**Phase 1 (this module): the SwiGLU MLP** — ~67% of ESM-C's params. Each
``UnifiedTransformerBlock.ffn`` is a Transformer Engine ``LayerNormMLP``; TE has
native TP (``set_parallel_mode=True`` → column-parallel fc1 with the SwiGLU
gate/up split handled internally + row-parallel fc2 + an internal all-reduce), so
the wrap is: construct a TP ``LayerNormMLP`` over the CP process group and copy the
full weights in sharded. The MLP's ``ffn_hidden`` (6912) divides 4/9/16, so this
works at every CP grid (unlike head-parallel attention, blocked by n_heads=40).

Validated bit-exact (modulo bf16 rounding) under ``torch.inference_mode`` — the
column/row shard + internal all-reduce reproduce the serial output; the ESM-C
forward already runs under the model's ``@torch.inference_mode`` so TE-TP (plain
sharded matmuls + NCCL, no autograd hooks) is inference-safe.

Attention TP (qkv column + out_proj row + the full-width QK-LayerNorm all-reduce,
constrained to TP=4 by the 40 heads) is a later phase — see ``fix_9x_ESMC_TP.md``.
"""

import torch
import torch.nn as nn


def _tp_layernorm_mlp(old, tp_group, tp_size: int, rank: int):
    """Build a TE LayerNormMLP TP-shard of ``old`` (a serial te.LayerNormMLP),
    holding only this rank's slice. Bit-exact (bf16) vs the full module."""
    import transformer_engine.pytorch as te

    hidden = old.layer_norm_weight.shape[0]
    ffn = old.fc2_weight.shape[1]  # fc2 = [hidden, ffn]
    if ffn % tp_size:
        raise ValueError(f"ESM-C ffn_hidden={ffn} not divisible by tp_size={tp_size}")
    dtype = old.fc1_weight.dtype
    device = old.fc1_weight.device

    new = te.LayerNormMLP(
        hidden,
        ffn,
        eps=getattr(old, "eps", 1e-5),
        normalization=getattr(old, "normalization", "LayerNorm"),
        activation=getattr(old, "activation", "swiglu"),
        bias=False,  # ESM-C trained with bias=False
        zero_centered_gamma=getattr(old, "zero_centered_gamma", False),
        set_parallel_mode=True,
        tp_group=tp_group,
        tp_size=tp_size,
        sequence_parallel=False,
        params_dtype=dtype,
    ).to(device=device, dtype=dtype)
    new.eval()

    fl = ffn // tp_size
    with torch.no_grad():
        new.layer_norm_weight.copy_(old.layer_norm_weight)  # ty:ignore[call-non-callable]
        new.layer_norm_bias.copy_(old.layer_norm_bias)  # ty:ignore[unresolved-attribute]
        # fc1 full = [2*ffn, hidden] laid out [gate(ffn); up(ffn)]; column-parallel
        # SwiGLU shard = cat([gate[r], up[r]]) (confirmed bit-exact by the spike).
        gate, up = old.fc1_weight[:ffn], old.fc1_weight[ffn:]
        new.fc1_weight.copy_(
            torch.cat(
                [gate[rank * fl : (rank + 1) * fl], up[rank * fl : (rank + 1) * fl]],
                dim=0,
            )
        )  # ty:ignore[call-non-callable]
        # fc2 full = [hidden, ffn]; row-parallel shards the input (ffn) dim.
        new.fc2_weight.copy_(old.fc2_weight[:, rank * fl : (rank + 1) * fl])  # ty:ignore[call-non-callable]
    return new


def tp_shard_esmc_mlp(model: nn.Module, dist_manager, tp_group=None) -> int:
    """Replace every ESM-C block's SwiGLU MLP with a TE tensor-parallel shard
    over ``tp_group`` (default: the full CP process group). Returns the number of
    blocks sharded. No-op (returns 0) if ESM-C isn't loaded.

    Each rank ends up holding ``1/tp_size`` of every MLP weight; the per-block
    forward all-reduces internally so the block output stays replicated — the
    ESM-C hidden states reach all CP ranks unchanged for the #14 lm_z builder.
    """
    esmc = getattr(model, "esmc", None)
    if esmc is None:
        return 0
    blocks = esmc.transformer.blocks

    if tp_group is None:
        tp_group = dist_manager.group["cp"]
    tp_size = torch.distributed.get_world_size(tp_group)
    rank = torch.distributed.get_rank(tp_group)
    if tp_size == 1:
        return 0

    n = 0
    for block in blocks:
        ffn = getattr(block, "ffn", None)
        # Only the TE LayerNormMLP path is supported (the accelerated build);
        # the pure-PyTorch fallback would need a DTensor TP and isn't on this path.
        if ffn is None or type(ffn).__name__ != "LayerNormMLP":
            raise TypeError(
                f"expected block.ffn to be a TE LayerNormMLP, got "
                f"{type(ffn).__name__ if ffn is not None else None}"
            )
        block.ffn = _tp_layernorm_mlp(ffn, tp_group, tp_size, rank)
        del ffn
        n += 1
    return n
