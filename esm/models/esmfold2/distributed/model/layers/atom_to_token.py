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

"""Distributed atom<->token gather/scatter for ESMFold2's diffusion module.

ESMFold2 maps atoms to tokens with an **index** (``atom_to_token: [B, A]`` int64),
not boltz-cp's one-hot ``[B, A, n_tokens]`` matrix, so these are index-based
gather/scatter (``torch.gather`` / ``scatter_reduce``) rather than a one-hot
matmul — strictly less memory (no ``A×L`` materialisation).

Sharding convention (matches boltz-cp's 1D sequence reprs): a sequence tensor
(atoms or tokens) on the ``(dp, cp_axis_0, cp_axis_1)`` mesh has placements
``(Shard(0), Shard(1), Replicate())`` — the sequence axis (dim 1) is split across
``cp_axis_0`` and replicated across ``cp_axis_1``. (The pair stays 2-D sharded;
the diffusion's heavy L×L work is handled separately by the ring attention.)

Inference-only (no autograd) — ESMFold2's CP path runs under
``@torch.inference_mode``.

These are the atom<->token data-movement primitives. The atom encoder/decoder's
sliding-window attention (``swa_window_size``) imposes a separate constraint when
sharding the atom axis — atom shards must align to window boundaries (shard at a
multiple of the window) or exchange halos — handled where the atom transformer is
distributed, not here.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from esm.models.esmfold2.layers import gather_token_to_atom

_SEQ_PL = [Shard(0), Shard(1), Replicate()]  # (dp, cp_axis_0=seq, cp_axis_1)
_SEQ_GATHERED_PL = [Shard(0), Replicate(), Replicate()]  # seq gathered on cp_axis_0


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def dist_gather_token_to_atom(
    token_dt: DTensor, atom_to_token_idx_dt: DTensor
) -> DTensor:
    """Broadcast per-token features to per-atom features (sharded).

    Parameters
    ----------
    token_dt:
        Token features ``(B, L, d)`` with placements ``(Shard(0), Shard(1),
        Replicate())`` (token axis split on ``cp_axis_0``).
    atom_to_token_idx_dt:
        Per-atom global token index ``(B, A)`` int64, same placements (atom axis
        split on ``cp_axis_0``).

    Returns
    -------
    Atom features ``(B, A, d)`` with placements ``(Shard(0), Shard(1),
    Replicate())`` (atom axis split on ``cp_axis_0``).
    """
    mesh = token_dt.device_mesh
    # Gather the full token axis onto every cp_axis_0 rank (token repr is the
    # cheap 1-D L×d, not L×L), so each rank can index its local atoms.
    token_full = token_dt.redistribute(mesh, _SEQ_GATHERED_PL).to_local().contiguous()
    idx_local = atom_to_token_idx_dt.to_local()
    atom_local = gather_token_to_atom(token_full, idx_local)

    b = atom_local.shape[0]
    a_global = atom_to_token_idx_dt.shape[1]
    d = atom_local.shape[-1]
    shape = torch.Size((b, a_global, d))
    return DTensor.from_local(
        atom_local.contiguous(),
        device_mesh=mesh,
        placements=_SEQ_PL,
        shape=shape,
        stride=_contiguous_strides(tuple(shape)),
    )


def _local_scatter_sum_count(
    atom_local: torch.Tensor,
    idx_local: torch.Tensor,
    n_tokens: int,
    mask_local: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local (per-rank) scatter-add of atom features into token bins, plus a
    per-token contributing-atom count. Masked atoms are routed to a throwaway
    bin ``n_tokens`` and dropped."""
    b, _, d = atom_local.shape
    idx_use = idx_local
    n_out = n_tokens
    if mask_local is not None:
        idx_use = torch.where(mask_local.bool(), idx_local, n_tokens)
        n_out = n_tokens + 1

    idx_e = idx_use.unsqueeze(-1).expand(b, idx_use.shape[1], d)
    s = torch.zeros(b, n_out, d, device=atom_local.device, dtype=atom_local.dtype)
    s.scatter_add_(1, idx_e, atom_local)
    c = torch.zeros(b, n_out, 1, device=atom_local.device, dtype=atom_local.dtype)
    c.scatter_add_(
        1,
        idx_use.unsqueeze(-1),
        torch.ones(
            b, idx_use.shape[1], 1, device=atom_local.device, dtype=atom_local.dtype
        ),
    )
    return s[:, :n_tokens], c[:, :n_tokens]


def dist_scatter_atom_to_token(
    atom_dt: DTensor,
    atom_to_token_idx_dt: DTensor,
    n_tokens: int,
    atom_mask_dt: DTensor | None = None,
) -> DTensor:
    """Aggregate per-atom features to per-token features (mean), sharded.

    Bit-equivalent to the serial ``scatter_atom_to_token`` (mean over the atoms
    of each token, empty tokens → 0): each rank scatter-adds its local atom shard
    into a full-L sum + count, the two are all-reduced across ``cp_axis_0``, then
    divided to a mean and re-sharded onto the token axis.

    Parameters
    ----------
    atom_dt:
        Atom features ``(B, A, d)`` with placements ``(Shard(0), Shard(1),
        Replicate())`` (atom axis split on ``cp_axis_0``).
    atom_to_token_idx_dt:
        Per-atom global token index ``(B, A)`` int64, same placements.
    n_tokens:
        Global token count ``L``.
    atom_mask_dt:
        Optional per-atom bool mask ``(B, A)``, same placements.

    Returns
    -------
    Token features ``(B, L, d)`` with placements ``(Shard(0), Shard(1),
    Replicate())`` (token axis split on ``cp_axis_0``).
    """
    mesh = atom_dt.device_mesh
    atom_local = atom_dt.to_local()
    idx_local = atom_to_token_idx_dt.to_local()
    mask_local = atom_mask_dt.to_local() if atom_mask_dt is not None else None

    s_local, c_local = _local_scatter_sum_count(
        atom_local, idx_local, n_tokens, mask_local
    )

    # Sum the per-rank partials across the cp_axis_0 sequence-shard group. The
    # mesh-dim-1 process group is exactly that axis; cp_axis_1 holds replicas, so
    # each cp_axis_1 column independently reconstructs the same full token tensor.
    cp0_group = mesh.get_group(1)
    if dist.is_initialized() and dist.get_world_size(cp0_group) > 1:
        dist.all_reduce(s_local, op=dist.ReduceOp.SUM, group=cp0_group)
        dist.all_reduce(c_local, op=dist.ReduceOp.SUM, group=cp0_group)

    mean_full = s_local / c_local.clamp(min=1.0)  # empty tokens -> 0

    # mean_full is the full (B, L, d) tensor, identical on every cp_axis_0 rank in
    # a column. Wrap as gathered-on-cp_axis_0, then reshard to the token axis
    # (Replicate -> Shard is a local chunk, no further communication).
    b, d = mean_full.shape[0], mean_full.shape[-1]
    shape = torch.Size((b, n_tokens, d))
    full_dt = DTensor.from_local(
        mean_full.contiguous(),
        device_mesh=mesh,
        placements=_SEQ_GATHERED_PL,
        shape=shape,
        stride=_contiguous_strides(tuple(shape)),
    )
    return full_dt.redistribute(mesh, _SEQ_PL)
