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

"""Distributed sharded pair-initialisation for ESMFold2 (inference-only).

The serial forward builds the initial pair tensors — ``z_init`` (an outer sum of
two per-token projections), the relative-position encoding, and the token-bond
encoding — as full ``[B, L, L, d_pair]`` tensors on *every* rank, then only
shards them at the recycle-loop boundary (``distribute_tensor`` inside
``CPRecycleEngine.run_loop``). That makes the per-rank init peak scale like a
single GPU (several full L×L tensors resident at once, plus ``rel_pos`` /
``token_bonds`` held full for the whole forward), so CP buys no memory relief for
the init phase and OOMs at short L.

``PairInitDistributed`` builds each of these directly as a 2-D-sharded
``(Shard(0), Shard(1), Shard(2))`` DTensor by computing only this rank's
``(row, col)`` block — the full L×L tensor is never resident. Per-token inputs
(``x_inputs`` and the index tensors) are cheap ``[B, L, *]`` and stay replicated;
only their O(L²) outer combinations are sharded. Each is wrapped as a DTensor so
the recycle engine / structure-head wrapper consume them with no gather.

The channel-wise submodules (``z_init_1`` / ``z_init_2``, the ``token_bonds``
Linear, and ``rel_pos.embed``) are the model's own layers, run on the local
block — their weights are identical on every rank, so the block result equals the
corresponding slice of the serial full tensor.
"""

from math import lcm, sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Shard

from esm.models.esmfold2.layers import ResIdxAsymIdSymIdEntityIdEncoding

_PAIR_PL = [Shard(0), Shard(1), Shard(2)]

# Per-rank RNG seed base for the sharded initial pair-state draw. Kept off the
# global generator so the recycle loop's MSA-subsample RNG stays synchronized
# across ranks (see ``init_pair_state``).
_Z_INIT_SEED = 0x2717


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _block_geometry(n_orig: int, device_mesh):
    """This rank's pair-block geometry: ``(lpad, s0, s1, row_slice, col_slice)``.

    The token axis is padded to ``shard_factor = lcm(cp_axis_0, cp_axis_1)``; the
    rank holds rows ``row_slice`` (its cp_axis_0 coordinate) × cols ``col_slice``
    (its cp_axis_1 coordinate) of the ``[lpad, lpad]`` pair.
    """
    cp0 = device_mesh.size(1)
    cp1 = device_mesh.size(2)
    shard_factor = lcm(cp0, cp1)
    coord = device_mesh.get_coordinate()
    assert coord is not None, "device mesh has no coordinate for this rank"
    row_rank, col_rank = int(coord[1]), int(coord[2])
    pad = (shard_factor - n_orig % shard_factor) % shard_factor
    lpad = n_orig + pad
    s0 = lpad // cp0
    s1 = lpad // cp1
    row = slice(row_rank * s0, (row_rank + 1) * s0)
    col = slice(col_rank * s1, (col_rank + 1) * s1)
    return lpad, s0, s1, row, col


def build_sharded_pair_mask(tok_mask: torch.Tensor, device_mesh) -> DTensor:
    """This rank's ``[B, s0, s1]`` block of the outer-product pair mask as a
    ``(Shard0, Shard1, Shard2)`` DTensor — never the full ``[B, L, L]``.

    Outer product of the
    row / col slices of the per-token mask (``tok_mask[:, row]`` ⊗
    ``tok_mask[:, col]``), value-identical to
    ``distribute_tensor(tok_mask[:,:,None] * tok_mask[:,None,:])`` but with no full
    L×L intermediate. The token axis is padded to
    ``shard_factor = lcm(cp_axis_0, cp_axis_1)`` (padded positions → 0, matching
    the serial zero-pad). Shared by ``PairInitDistributed.pair_mask`` (recycle
    pair mask) and the MSA encoder's ``forward_sharded`` (per-iteration mask).

    ``tok_mask`` is ``[B, L]`` (any numeric / bool dtype); the returned DTensor is
    float32 — cast at the call site to the consumer dtype.
    """
    b = tok_mask.shape[0]
    lpad, _, _, row, col = _block_geometry(tok_mask.shape[1], device_mesh)
    pad = lpad - tok_mask.shape[1]

    tm = tok_mask.float()
    if pad:
        tm = F.pad(tm, (0, pad))
    block = tm[:, row].unsqueeze(2) * tm[:, col].unsqueeze(1)  # [B, s0, s1]

    shape = torch.Size((b, lpad, lpad))
    return DTensor.from_local(
        block.contiguous(),
        device_mesh=device_mesh,
        placements=_PAIR_PL,
        stride=_contiguous_strides(tuple(shape)),
        shape=shape,
    )


def build_sharded_distogram_bins(
    rep_coords: torch.Tensor, boundaries: torch.Tensor, device_mesh
) -> DTensor:
    """This rank's ``[B, s0, s1]`` block of the distance-distogram bins as a
    ``(Shard0, Shard1, Shard2)`` DTensor — ``cdist`` on the row / col coordinate
    blocks only, never the full ``[B, N, N]`` distance matrix.

    ``rep_coords`` is ``[B, N, 3]`` per-token representative-atom coordinates
    (replicated, cheap). ``boundaries`` is the distance-bin edge buffer. Padded
    rows/cols → bin 0, matching the serial ``F.pad(distogram_bins)`` zero-fill.
    This has no serial counterpart: the serial confidence head computes the full
    ``[B, N, N]`` cdist instead.
    """
    b = rep_coords.shape[0]
    n = rep_coords.shape[1]
    lpad, _, _, row, col = _block_geometry(n, device_mesh)
    pad = lpad - n

    coords = F.pad(rep_coords, (0, 0, 0, pad)) if pad else rep_coords
    row_c = coords[:, row]  # [B, s0, 3]
    col_c = coords[:, col]  # [B, s1, 3]
    dist = torch.cdist(row_c, col_c, compute_mode="donot_use_mm_for_euclid_dist")
    bins = (dist.unsqueeze(-1) > boundaries).sum(dim=-1).long()  # [B, s0, s1]
    if pad:
        rr = torch.arange(row.start, row.stop, device=rep_coords.device) < n
        cc = torch.arange(col.start, col.stop, device=rep_coords.device) < n
        bins = bins * (rr[:, None] & cc[None, :])  # padded positions → bin 0

    shape = torch.Size((b, lpad, lpad))
    return DTensor.from_local(
        bins.contiguous(),
        device_mesh=device_mesh,
        placements=_PAIR_PL,
        stride=_contiguous_strides(tuple(shape)),
        shape=shape,
    )


class PairInitDistributed(nn.Module):
    """Builds ``z_init`` / ``rel_pos`` / ``token_bonds`` as 2-D-sharded DTensors.

    ``forward`` returns ``(z_init_dt, rel_pos_dt, token_bonds_dt, n_orig)``: the
    three pair tensors as padded ``(Shard0, Shard1, Shard2)`` DTensors (token axis
    padded to ``shard_factor = lcm(cp_axis_0, cp_axis_1)`` — the same padding
    contract the recycle engine and LM->pair builder use) plus the original
    (pre-pad) token count. ``z_init_dt`` already folds in ``rel_pos`` +
    ``token_bonds`` (matching the serial ``z_init`` sum); ``rel_pos_dt`` /
    ``token_bonds_dt`` are returned separately for the structure / confidence
    heads.
    """

    def __init__(self, model, dist_manager) -> None:
        super().__init__()
        self.device_mesh = dist_manager.device_mesh_subgroups
        # device_mesh is (dp, cp_axis_0, cp_axis_1); rows shard on cp_axis_0, cols on cp_axis_1.
        self.cp_axis_0 = self.device_mesh.size(1)
        self.cp_axis_1 = self.device_mesh.size(2)
        self.shard_factor = lcm(self.cp_axis_0, self.cp_axis_1)
        coord = self.device_mesh.get_coordinate()
        assert coord is not None, "device mesh has no coordinate for this rank"
        self.row_rank = int(coord[1])
        self.col_rank = int(coord[2])

        # Model's own channel-wise layers (run on the local block; weights replicated).
        self.z_init_1 = model.z_init_1
        self.z_init_2 = model.z_init_2
        self.token_bonds = model.token_bonds
        self.rel_pos: ResIdxAsymIdSymIdEntityIdEncoding = model.rel_pos

    def _block_valid_mask(
        self, n_orig: int, lpad: int, device: torch.device
    ) -> torch.Tensor | None:
        """This rank's ``[s0, s1]`` boolean mask marking (row, col) positions that
        are *real* tokens (global index < ``n_orig``), or ``None`` if there is no
        CP padding. Used to zero the padded rows/cols so the sharded pair matches
        the serial ``_pad_pair`` (zero-fill) contract — padded positions must not
        leak nonzero values into the masked trunk. Note this is a *position* mask
        (CP padding only), NOT the token attention mask: in-range but attention-
        masked tokens stay nonzero here, exactly as in the serial full tensor."""
        if lpad - n_orig <= 0:
            return None
        s0 = lpad // self.cp_axis_0
        s1 = lpad // self.cp_axis_1
        r0 = self.row_rank * s0
        c0 = self.col_rank * s1
        row_valid = torch.arange(r0, r0 + s0, device=device) < n_orig
        col_valid = torch.arange(c0, c0 + s1, device=device) < n_orig
        return row_valid[:, None] & col_valid[None, :]  # [s0, s1] bool

    def _wrap(self, local: torch.Tensor, b: int, lpad: int, d: int) -> DTensor:
        shape = torch.Size((b, lpad, lpad, d))
        return DTensor.from_local(
            local.contiguous(),
            device_mesh=self.device_mesh,
            placements=_PAIR_PL,
            shape=shape,
            stride=_contiguous_strides(tuple(shape)),
        )

    def _sharded_rel_pos(
        self,
        residue_index: torch.Tensor,
        asym_id: torch.Tensor,
        sym_id: torch.Tensor,
        entity_id: torch.Tensor,
        token_index: torch.Tensor,
        row: slice,
        col: slice,
    ) -> torch.Tensor:
        """This rank's ``[B, s0, s1, d_pair]`` block of the relative-position
        encoding. Byte-for-byte the arithmetic of
        ``ResIdxAsymIdSymIdEntityIdEncoding.forward`` but with the index tensors
        sliced to the row / col block before the outer differences (so no full
        ``[B, L, L, *]`` one-hot intermediate)."""
        rp = self.rel_pos
        bins_r = rp.n_relative_residx_bins
        bins_c = rp.n_relative_chain_bins

        ri_r, ri_c = residue_index[:, row], residue_index[:, col]
        ai_r, ai_c = asym_id[:, row], asym_id[:, col]
        si_r, si_c = sym_id[:, row], sym_id[:, col]
        ei_r, ei_c = entity_id[:, row], entity_id[:, col]
        ti_r, ti_c = token_index[:, row], token_index[:, col]

        same_chain = ai_r.unsqueeze(2) == ai_c.unsqueeze(1)
        same_res = ri_r.unsqueeze(2) == ri_c.unsqueeze(1)
        same_ent = ei_r.unsqueeze(2) == ei_c.unsqueeze(1)

        d_res = torch.clip(
            ri_r.unsqueeze(2) - ri_c.unsqueeze(1) + bins_r, 0, 2 * bins_r
        )
        d_res = torch.where(same_chain, d_res, 2 * bins_r + 1)
        a_res = F.one_hot(d_res, 2 * bins_r + 2)

        d_tok = torch.clip(
            ti_r.unsqueeze(2) - ti_c.unsqueeze(1) + bins_r, 0, 2 * bins_r
        )
        d_tok = torch.where(same_chain & same_res, d_tok, 2 * bins_r + 1)
        a_tok = F.one_hot(d_tok, 2 * bins_r + 2)

        d_ch = torch.clip(si_r.unsqueeze(2) - si_c.unsqueeze(1) + bins_c, 0, 2 * bins_c)
        d_ch = torch.where(same_chain, 2 * bins_c + 1, d_ch)
        a_ch = F.one_hot(d_ch, 2 * bins_c + 2)

        feats = torch.cat(
            [
                a_res.float(),
                a_tok.float(),
                same_ent.float().unsqueeze(-1),
                a_ch.float(),
            ],
            dim=-1,
        )
        return rp.embed(feats)  # [B, s0, s1, d_pair]

    def forward(
        self,
        x_inputs: torch.Tensor,
        residue_index: torch.Tensor,
        asym_id: torch.Tensor,
        sym_id: torch.Tensor,
        entity_id: torch.Tensor,
        token_index: torch.Tensor,
        token_bonds: torch.Tensor,
    ) -> tuple[DTensor, DTensor, DTensor, int]:
        b = x_inputs.shape[0]
        n_orig = x_inputs.shape[1]
        pad = (self.shard_factor - n_orig % self.shard_factor) % self.shard_factor
        lpad = n_orig + pad
        s0 = lpad // self.cp_axis_0
        s1 = lpad // self.cp_axis_1
        row = slice(self.row_rank * s0, (self.row_rank + 1) * s0)
        col = slice(self.col_rank * s1, (self.col_rank + 1) * s1)

        # --- z_init: outer SUM of two per-token projections, sliced to the block.
        # z1/z2 are [B, L, d] (per-token, cheap — not L×L), so building them full and
        # slicing costs O(L*d), not O(L²).
        z1 = self.z_init_1(x_inputs)
        z2 = self.z_init_2(x_inputs)
        d = z1.shape[-1]
        if pad:
            z1 = F.pad(z1, (0, 0, 0, pad))
            z2 = F.pad(z2, (0, 0, 0, pad))
        z_init_local = z1[:, row].unsqueeze(2) + z2[:, col].unsqueeze(
            1
        )  # [B, s0, s1, d]

        # --- relative-position encoding (index tensors sliced to row/col block) ---
        def _pad_idx(idx: torch.Tensor) -> torch.Tensor:
            return F.pad(idx, (0, pad)) if pad else idx

        rel_local = self._sharded_rel_pos(
            _pad_idx(residue_index),
            _pad_idx(asym_id),
            _pad_idx(sym_id),
            _pad_idx(entity_id),
            _pad_idx(token_index),
            row,
            col,
        )  # [B, s0, s1, d]

        # --- token-bond encoding: slice the (valid) input block, then pad the small
        # block to [s0, s1] — never materialises a full padded L×L×1 tensor.
        r0 = self.row_rank * s0
        c0 = self.col_rank * s1
        tb_v = token_bonds[
            :, r0 : min(r0 + s0, n_orig), c0 : min(c0 + s1, n_orig), :
        ].float()
        pr = s0 - tb_v.shape[1]
        pc = s1 - tb_v.shape[2]
        if pr or pc:
            tb_v = F.pad(tb_v, (0, 0, 0, pc, 0, pr))
        tb_local = self.token_bonds(tb_v)  # [B, s0, s1, d]

        z_comb = z_init_local + rel_local + tb_local
        # Zero the CP-padded rows/cols so the sharded pair matches the serial
        # zero-padding (else e.g. z_init_local[pad_row, j] = z2[j] would leak).
        vmask = self._block_valid_mask(n_orig, lpad, x_inputs.device)
        if vmask is not None:
            z_comb = z_comb * vmask[None, :, :, None].to(z_comb.dtype)
            rel_local = rel_local * vmask[None, :, :, None].to(rel_local.dtype)
            tb_local = tb_local * vmask[None, :, :, None].to(tb_local.dtype)

        z_init_dt = self._wrap(z_comb, b, lpad, d)
        rel_pos_dt = self._wrap(rel_local, b, lpad, d)
        token_bonds_dt = self._wrap(tb_local, b, lpad, d)
        return z_init_dt, rel_pos_dt, token_bonds_dt, n_orig

    def pair_mask(self, tok_mask: torch.Tensor) -> DTensor:
        """This rank's sharded ``[B, s0, s1]`` block of the pair attention mask
        (see :func:`build_sharded_pair_mask`)."""
        return build_sharded_pair_mask(tok_mask, self.device_mesh)

    def init_pair_state(self, z_init_dt: DTensor, n_orig: int) -> DTensor:
        """Sharded random initial pair state (the serial ``_init_pair_state``).

        Draws this rank's local block from a *per-rank* generator (distinct seed)
        so (a) the full L×L state is never resident and (b) the global RNG — which
        the recycle loop's MSA subsample relies on to stay in lockstep across
        ranks — is untouched. Not bit-exact with the serial full-tensor
        ``trunc_normal_`` draw (a valid-but-different iid sample, and ``clamp`` vs
        resample at the ±3σ bound), matching the shard-local-dropout contract
        already used in the recycle engine; validate by end-to-end pLDDT/pTM
        parity. CP-padded rows/cols are zeroed to match the serial ``_pad_pair``.
        """
        ref = z_init_dt.to_local()
        b, lpad, _, d = z_init_dt.shape
        std = sqrt(2.0 / (5.0 * ref.shape[-1]))
        gen = torch.Generator(device=ref.device)
        gen.manual_seed(_Z_INIT_SEED + torch.distributed.get_rank())
        state = torch.empty(ref.shape, dtype=torch.float32, device=ref.device)
        state.normal_(0.0, std, generator=gen).clamp_(-3 * std, 3 * std)
        state = state.to(dtype=ref.dtype)
        vmask = self._block_valid_mask(int(n_orig), int(lpad), ref.device)
        if vmask is not None:
            state = state * vmask[None, :, :, None].to(state.dtype)
        return self._wrap(state, int(b), int(lpad), int(d))
