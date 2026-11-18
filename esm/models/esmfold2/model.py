"""PyTorch ESMFold2 model — the standard released architecture.

Quickstart::

    from transformers import EsmFold2Model

    model = EsmFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
    open("ubq.pdb", "w").write(model.infer_protein_as_pdb("MQIFVKTLTGKT..."))

For multi-chain / ligand / MSA inputs see ``ESMFold2InputBuilder`` in the
companion ``esm`` package.
"""

import math
from collections.abc import Callable
from contextlib import nullcontext
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from esm.models.esmc import EsmcModel
from esm.models.esmc.checkpoint_layout import published_to_native_subtree
from esm.models.esmfold2 import rowpair
from esm.models.esmfold2.config import EsmFold2Config, default_module_flags
from esm.models.esmfold2.constants import MOL_TYPE_NONPOLYMER
from esm.models.esmfold2.fp8 import convert_esmc_to_fp8_
from esm.models.esmfold2.hf_checkpoint import hf_state_dict_to_native, is_hf_layout
from esm.models.esmfold2.layers import (
    _VALID_MEMORY_MODES,
    CHAR_VOCAB_SIZE,
    MAX_ATOMIC_NUMBER,
    MEMORY_BIG,
    NUM_RES_TYPES,
    DiffusionStructureHead,
    FoldingTrunk,
    InputsEmbedder,
    LanguageModelShim,
    MSAPairWeightedAveraging,
    OuterProductMean,
    ResIdxAsymIdSymIdEntityIdEncoding,
    RowAttentionPooling,
    SwiGLUMLP,
    TriangleMultiplicativeUpdate,
    _categorical_mean,
    _compute_intra_token_idx,
    _memory_big,
    _to_fused_kernel_dtype,
    compute_lm_hidden_states,
    fast_trimul_refusal_reason,
    gather_rep_atom_coords,
    gather_token_to_atom,
    maybe_apply_msa_column_masking,
    maybe_subsample_msa,
    row_bands,
    warn_fast_trimul_refused,
)
from esm.models.hub import HubPreTrainedModel, resolve_model_dir

_EPS = 1e-6

# Default for the triangle / OPM / pair-transition L² ops. Caps peak memory
# so L≈2k folds on an 80 GB GPU (~76 GB peak at chunk=128 for L=1438;
# chunk=64 leaves headroom for the largest foldbench targets). Override via
# ``model.set_chunk_size(...)``; pass None to disable chunking (faster for
# short L but OOM-prone past ~600).
_DEFAULT_CHUNK_SIZE = 64

# Below this, ESM-C's weights never set the peak, so the memory mode leaves them
# resident rather than pay the round trip to the host.
_ESMC_OFFLOAD_MIN_TOKENS = 1500

# ``ConfidenceHead._pair_scores``: per-row reductions, chain TM rows, PAE / PDE.
PairScores = dict[str, Tensor | list[Tensor | None] | None]

# A pair encoding by local row band: ``fn(lo, hi) -> [B, hi - lo, L, C]``.
RowsFn = Callable[[int, int], Tensor]

# Keys ``prepare_esmfold2_input`` emits that inference has no consumer for.
# ``forward`` accepts and drops these and rejects every other unknown keyword; it
# cannot drop the catch-all entirely because ``fold`` splats the whole feature
# dict, which is a strict superset of the signature.
_IGNORED_FEATURE_KEYS = frozenset(
    {"pocket_feature", "gt_coords", "is_resolved", "frames_idx"}
)


class PairTransition(nn.Module):
    """LayerNorm + SwiGLU feed-forward residual block on the pair representation."""

    def __init__(self, d_model: int, expansion_ratio: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLUMLP(d_model, expansion_ratio=expansion_ratio, bias=False)
        self._chunk_size: int | None = _DEFAULT_CHUNK_SIZE

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._chunk_size = chunk_size

    def forward(self, x: Tensor) -> Tensor:
        if self._chunk_size is None or x.shape[1] <= self._chunk_size:
            return self.ffn(self.norm(x))
        out: list[Tensor] = []
        for s in range(0, x.shape[1], self._chunk_size):
            e = min(s + self._chunk_size, x.shape[1])
            sl = x[:, s:e]
            out.append(self.ffn(self.norm(sl)))
        return torch.cat(out, dim=1)


def _cat_scores(parts: list[PairScores], dim: int) -> PairScores:
    """``ConfidenceHead._pair_scores`` of consecutive pieces, joined along ``dim``."""
    out: PairScores = {
        key: torch.cat([cast(Tensor, p[key]) for p in parts], dim=dim)
        for key in ("pae", "pde", "ptm_per_row", "iptm_per_row")
    }
    chains = [cast(list, p["chain_tm"]) for p in parts]
    out["chain_tm"] = [
        None if col[0] is None else torch.cat(list(col), dim=dim)
        for col in zip(*chains)
    ]
    return out


class ConfidenceHead(nn.Module):
    """Predicts pLDDT, PAE, PDE, resolved-atom probability and distogram bins."""

    boundaries: Tensor

    def __init__(self, config: "EsmFold2Config") -> None:
        super().__init__()
        ch = config.confidence_head
        d_single = config.hidden_size
        d_pair = config.pairwise_hidden_size
        d_inputs = config.single_inputs_size

        boundaries = torch.linspace(ch.min_dist, ch.max_dist, ch.distogram_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(ch.distogram_bins, d_pair)

        self.s_norm = nn.LayerNorm(d_single)
        self.s_inputs_to_single = nn.Linear(d_inputs, d_single, bias=False)
        self.s_to_z = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_transpose = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_in1 = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_in2 = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_out = nn.Linear(d_pair, d_pair, bias=False)
        self.s_input_to_s = nn.Linear(d_inputs, d_single, bias=False)
        self.s_inputs_norm = nn.LayerNorm(d_inputs)
        self.z_norm = nn.LayerNorm(d_pair)

        self.row_attention_pooling = RowAttentionPooling(
            d_pair=d_pair, d_single=d_single
        )

        self.folding_trunk = FoldingTrunk(
            n_layers=ch.num_hidden_layers, d_pair=d_pair, expansion_ratio=4
        )

        # Heads.
        self.plddt_ln = nn.LayerNorm(d_single)
        max_atoms_per_token = 23
        self.plddt_weight = nn.Parameter(
            torch.zeros(max_atoms_per_token, d_single, ch.num_plddt_bins)
        )

        self.pae_ln = nn.LayerNorm(d_pair)
        self.pae_head = nn.Linear(d_pair, ch.num_pae_bins, bias=False)

        self.pde_ln = nn.LayerNorm(d_pair)
        self.pde_head = nn.Linear(d_pair, ch.num_pde_bins, bias=False)

        self.resolved_ln = nn.LayerNorm(d_single)
        # 2 = resolved logits ([unresolved, resolved]).
        self.resolved_weight = nn.Parameter(
            torch.zeros(max_atoms_per_token, d_single, 2)
        )
        self._memory_mode: str | None = None

    def set_kernel_backend(self, backend: str | None) -> None:
        self.folding_trunk.set_kernel_backend(backend)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)

    def set_memory_mode(self, mode: str | None) -> None:
        self._memory_mode = mode

    @staticmethod
    def _repeat_batch(x: Tensor, num_diffusion_samples: int) -> Tensor:
        return (
            x
            if num_diffusion_samples == 1
            else x.repeat_interleave(num_diffusion_samples, 0)
        )

    @staticmethod
    def _flatten_sample_axis(x: Tensor) -> Tensor:
        if x.ndim == 4:
            b, mult, n, c = x.shape
            return x.reshape(b * mult, n, c)
        return x

    def _pair_terms(
        self, s_inputs_normed: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """The single-to-pair terms of the pair base: row, column and product."""
        return (
            self.s_to_z(s_inputs_normed).unsqueeze(2),
            self.s_to_z_transpose(s_inputs_normed).unsqueeze(1),
            self.s_to_z_prod_in1(s_inputs_normed)[:, :, None, :],
            self.s_to_z_prod_in2(s_inputs_normed)[:, None, :, :],
        )

    def _base_band(
        self,
        terms: tuple[Tensor, Tensor, Tensor, Tensor],
        z: Tensor,
        rel: Tensor | None,
        bonds: Tensor | None,
        g0: int,
        g1: int,
    ) -> Tensor:
        """Global pair rows ``[g0, g1)`` of the pair base; the inputs are those rows."""
        row, col, prod_in1, prod_in2 = terms
        out = self.z_norm(z.float())
        if rel is not None:
            out = out + rel
        if bonds is not None:
            out = out + bonds
        out = out + row[:, g0:g1]
        out = out + col
        return out + self.s_to_z_prod_out(prod_in1[:, g0:g1] * prod_in2)

    def _pair_base(
        self,
        s_inputs_normed: Tensor,
        z: Tensor,
        relative_position_encoding: Tensor | None,
        token_bonds_encoding: Tensor | None,
    ) -> Tensor:
        terms = self._pair_terms(s_inputs_normed)

        def band(lo: int, hi: int) -> Tensor:
            rel, bonds = relative_position_encoding, token_bonds_encoding
            return self._base_band(
                terms,
                z[:, lo:hi],
                None if rel is None else rel[:, lo:hi],
                None if bonds is None else bonds[:, lo:hi],
                lo,
                hi,
            )

        if not _memory_big(self):
            return band(0, z.shape[1])
        base = torch.empty_like(z, dtype=torch.float32)
        for lo, hi in row_bands(z.shape[1]):
            base[:, lo:hi] = band(lo, hi)
        return base

    def _pair_outputs(
        self,
        pair_base: Tensor,
        x_pred: Tensor,
        rep_idx: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        rep_coords = gather_rep_atom_coords(x_pred, rep_idx)
        rep_distances = torch.cdist(
            rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist"
        )
        bins = (rep_distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair_base + self.dist_bin_pairwise_embed(bins)

        # FoldingTrunk handles the bf16 cast internally during inference so
        # each block's fused trimul engages. In-place residual avoids an
        # extra fp32 pair allocation.
        with torch.amp.autocast("cuda", enabled=pair.is_cuda, dtype=torch.bfloat16):
            pair_delta = self.folding_trunk(pair, pair_attention_mask=pair_mask)
        pair.add_(pair_delta.float())
        del pair_delta
        single = self.row_attention_pooling(pair, mask)
        if not _memory_big(self):
            return (
                single,
                self.pae_head(self.pae_ln(pair)),
                self.pde_head(self.pde_ln(pair)),
            )
        # Banded, so the fp32 LayerNorm copy of the pair never exists whole.
        pae = pair.new_empty((*pair.shape[:-1], self.pae_head.out_features))
        pde = pair.new_empty((*pair.shape[:-1], self.pde_head.out_features))
        for lo, hi in row_bands(pair.shape[1]):
            pae[:, lo:hi] = self.pae_head(self.pae_ln(pair[:, lo:hi]))
            pde[:, lo:hi] = self.pde_head(self.pde_ln(pair[:, lo:hi]))
        return single, pae, pde

    def forward(
        self,
        s_inputs: Tensor,
        z: Tensor,
        x_pred: Tensor,
        distogram_atom_idx: Tensor,
        token_attention_mask: Tensor,
        atom_to_token: Tensor,
        atom_attention_mask: Tensor,
        asym_id: Tensor,
        mol_type: Tensor,
        num_diffusion_samples: int = 1,
        relative_position_encoding: Tensor | None = None,
        token_bonds_encoding: Tensor | None = None,
    ) -> dict[str, Tensor]:
        s_inputs_normed = self.s_inputs_norm(s_inputs)
        z_base = self._pair_base(
            s_inputs_normed, z, relative_position_encoding, token_bonds_encoding
        )

        x_pred_flat = self._flatten_sample_axis(x_pred)
        rep_idx_m = self._repeat_batch(distogram_atom_idx, num_diffusion_samples).long()
        mask = self._repeat_batch(token_attention_mask, num_diffusion_samples)
        pair_mask = mask[:, :, None].float() * mask[:, None, :].float()

        if _memory_big(self) and num_diffusion_samples > 1:
            parts = [
                self._pair_outputs(
                    z_base[i // num_diffusion_samples : i // num_diffusion_samples + 1],
                    x_pred_flat[i : i + 1],
                    rep_idx_m[i : i + 1],
                    mask[i : i + 1],
                    pair_mask[i : i + 1],
                )
                for i in range(x_pred_flat.shape[0])
            ]
            single, pae_logits, pde_logits = (torch.cat(t, dim=0) for t in zip(*parts))
        else:
            single, pae_logits, pde_logits = self._pair_outputs(
                self._repeat_batch(z_base, num_diffusion_samples),
                x_pred_flat,
                rep_idx_m,
                mask,
                pair_mask,
            )
        return self._finish(
            single=single,
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            x_pred=x_pred,
            distogram_atom_idx=distogram_atom_idx,
            token_attention_mask=token_attention_mask,
            atom_to_token=atom_to_token,
            atom_attention_mask=atom_attention_mask,
            asym_id=asym_id,
            mol_type=mol_type,
            num_diffusion_samples=num_diffusion_samples,
        )

    def scores_rows(
        self,
        *,
        s_inputs: Tensor,
        z: Tensor,
        rel_rows: Callable[[int, int], Tensor] | None,
        bonds_rows: Callable[[int, int], Tensor] | None,
        x_pred: Tensor,
        distogram_atom_idx: Tensor,
        token_attention_mask: Tensor,
        asym_id: Tensor,
        mol_type: Tensor,
        num_diffusion_samples: int,
        layout: rowpair.Layout,
    ) -> tuple[Tensor, PairScores, Tensor]:
        """``(single, pair scores, interface_per_token)`` from this rank's rows.

        ``z`` is the trunk's ``[1, R, N, C]`` rows; ``rel_rows(lo, hi)`` and
        ``bonds_rows(lo, hi)`` give local rows ``[lo, hi)`` of the encodings.
        Per sample, the pair base is rebuilt per band, the head trunk runs in
        place on one shard, and every pair reduction is taken per band. So no
        fp32 shard, no ``[S, N, N, bins]`` logits and no ``[S, N, N]`` distances
        exist; the logits are not returned. pTM / ipTM / pLDDT are whole on
        every rank; PAE and PDE are on rank 0's host only. ``z`` is no longer
        read. Runs inside ``rowpair.sharded(layout)``.
        """
        r0, R = layout.r0, layout.R
        terms = self._pair_terms(self.s_inputs_norm(s_inputs))
        x_pred_flat = self._flatten_sample_axis(x_pred)
        rep_idx_m = self._repeat_batch(distogram_atom_idx, num_diffusion_samples).long()
        mask = self._repeat_batch(token_attention_mask, num_diffusion_samples)
        asym_m = self._repeat_batch(asym_id, num_diffusion_samples)
        is_ligand = (
            self._repeat_batch(mol_type, num_diffusion_samples) == MOL_TYPE_NONPOLYMER
        ).float()
        pair_mask = (
            rowpair.rows_of(mask, layout)[:, :, None].float() * mask[:, None, :].float()
        )
        trunk_dtype = self.folding_trunk.compute_dtype(torch.float32, z.is_cuda)

        singles: list[Tensor] = []
        scores: list[dict] = []
        interfaces: list[Tensor] = []
        for i in range(x_pred_flat.shape[0]):
            rep = gather_rep_atom_coords(x_pred_flat[i : i + 1], rep_idx_m[i : i + 1])

            def pair_band(lo: int, hi: int) -> tuple[Tensor, Tensor]:
                dist = torch.cdist(
                    rep[:, r0 + lo : r0 + hi],
                    rep,
                    compute_mode="donot_use_mm_for_euclid_dist",
                )
                bins = (dist.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
                base = self._base_band(
                    terms,
                    z[:, lo:hi],
                    None if rel_rows is None else rel_rows(lo, hi),
                    None if bonds_rows is None else bonds_rows(lo, hi),
                    r0 + lo,
                    r0 + hi,
                )
                return base + self.dist_bin_pairwise_embed(bins), dist

            trunk_in = torch.empty(
                (1, R, *z.shape[2:]), dtype=trunk_dtype, device=z.device
            )
            for lo, hi in row_bands(R):
                trunk_in[:, lo:hi] = pair_band(lo, hi)[0]
            with torch.amp.autocast("cuda", enabled=z.is_cuda, dtype=torch.bfloat16):
                trunk_out = self.folding_trunk(
                    trunk_in, pair_attention_mask=pair_mask[i : i + 1], inplace=True
                )
            del trunk_in
            band_singles, band_scores, band_interface = [], [], []
            for lo, hi in row_bands(R):
                pair, dist = pair_band(lo, hi)
                pair.add_(trunk_out[:, lo:hi].float())
                g = slice(r0 + lo, r0 + hi)
                band_interface.append(
                    self._interface_rows(
                        dist,
                        asym_m[i : i + 1, g],
                        asym_m[i : i + 1],
                        is_ligand[i : i + 1, g],
                    )
                )
                band_singles.append(self.row_attention_pooling(pair, mask[i : i + 1]))
                band_scores.append(
                    self._pair_scores(
                        self.pae_head(self.pae_ln(pair)),
                        self.pde_head(self.pde_ln(pair)),
                        mask[i : i + 1],
                        asym_m[i : i + 1],
                        r0 + lo,
                    )
                )
                del pair, dist
            del trunk_out
            singles.append(torch.cat(band_singles, dim=1))
            scores.append(_cat_scores(band_scores, dim=1))
            interfaces.append(torch.cat(band_interface, dim=1))

        merged = _cat_scores(scores, dim=0)
        for key in ("ptm_per_row", "iptm_per_row"):
            merged[key] = rowpair.all_gather_rows(
                cast(Tensor, merged[key]).contiguous(), layout, dim=1
            )
        # The [S, N, N] matrices go to rank 0's host: whole on every rank they
        # would be an N^2 term that no rank count shrinks.
        for key in ("pae", "pde"):
            rows = cast(Tensor, merged[key])
            parts = [
                rowpair.gather_rows_to_rank0_host(rows[i : i + 1].contiguous(), layout)
                for i in range(rows.shape[0])
            ]
            merged[key] = (
                None
                if parts[0] is None
                else torch.cat([cast(Tensor, p) for p in parts], dim=0)
            )
        merged["chain_tm"] = [
            None
            if t is None
            else rowpair.all_gather_rows(t.contiguous(), layout, dim=1)
            for t in cast(list, merged["chain_tm"])
        ]
        single = rowpair.all_gather_rows(torch.cat(singles, dim=0), layout, dim=1)
        interface = rowpair.all_gather_rows(torch.cat(interfaces, dim=0), layout, dim=1)
        return single, merged, interface

    def _finish(
        self,
        *,
        single: Tensor,
        pae_logits: Tensor,
        pde_logits: Tensor,
        x_pred: Tensor,
        distogram_atom_idx: Tensor,
        token_attention_mask: Tensor,
        atom_to_token: Tensor,
        atom_attention_mask: Tensor,
        asym_id: Tensor,
        mol_type: Tensor,
        num_diffusion_samples: int,
    ) -> dict[str, Tensor]:
        """Atom-space (pLDDT/resolved) + PAE/PDE + pTM/ipTM back-half.

        Split out of ``forward`` so the CP wrapper (#6) can run the distributed
        pair front (z_base → trunk → row-pool → pae/pde heads), gather the small
        ``single`` / ``pae_logits`` / ``pde_logits``, and reuse this exact
        (serial, replicated) reduction code — one source of truth for the fiddly
        pTM/ipTM/pair_chains logic."""
        scores = self._pair_scores(
            pae_logits,
            pde_logits,
            self._repeat_batch(token_attention_mask, num_diffusion_samples),
            self._repeat_batch(asym_id, num_diffusion_samples),
        )
        out = self._finish_scores(
            single=single,
            scores=scores,
            x_pred=x_pred,
            distogram_atom_idx=distogram_atom_idx,
            token_attention_mask=token_attention_mask,
            atom_to_token=atom_to_token,
            atom_attention_mask=atom_attention_mask,
            asym_id=asym_id,
            mol_type=mol_type,
            num_diffusion_samples=num_diffusion_samples,
        )
        out["pae_logits"] = pae_logits
        out["pde_logits"] = pde_logits
        return out

    @staticmethod
    def _interface_rows(
        rep_distances: Tensor, asym_rows: Tensor, asym: Tensor, is_ligand_rows: Tensor
    ) -> Tensor:
        """Per row token: 1 if within 8 A of another chain and not a ligand."""
        inter_chain = (asym_rows.unsqueeze(-1) != asym.unsqueeze(-2)).float()
        near_contact = (rep_distances < 8).float()
        return (near_contact * inter_chain * (1.0 - is_ligand_rows).unsqueeze(-1)).amax(
            dim=-1
        )

    def _pair_scores(
        self,
        pae_logits: Tensor,
        pde_logits: Tensor,
        mask: Tensor,
        expanded_asym: Tensor,
        row0: int = 0,
    ) -> PairScores:
        """The pair-head reductions of rows ``[row0, row0 + R)``.

        Every value is a per-row reduction over all columns, so a rank computes
        its own rows and the max over rows happens after the gather.
        """
        R = pae_logits.shape[1]
        pae = _categorical_mean(pae_logits, start=0.0, end=32.0).detach()
        pde = _categorical_mean(pde_logits, start=0.0, end=32.0).detach()

        n_bins = pae_logits.shape[-1]
        bin_width = 32.0 / n_bins
        bin_centers = torch.arange(
            0.5 * bin_width, 32.0, bin_width, device=pae_logits.device
        )
        mask_f = mask.float()
        N_res = mask_f.sum(dim=-1, keepdim=True)
        d0 = 1.24 * (N_res.clamp(min=19) - 15) ** (1 / 3) - 1.8
        tm_per_bin = 1 / (1 + (bin_centers / d0) ** 2)
        pae_probs = F.softmax(pae_logits, dim=-1)
        tm_expected = (pae_probs * tm_per_bin[:, None, None, :]).sum(dim=-1)
        del pae_probs

        mask_rows = mask_f[:, row0 : row0 + R]
        pair_mask_2d = mask_rows.unsqueeze(-1) * mask_f.unsqueeze(-2)
        ptm_per_row = (tm_expected * pair_mask_2d).sum(dim=-1) / (
            pair_mask_2d.sum(dim=-1) + _EPS
        )
        asym_rows = expanded_asym[:, row0 : row0 + R]
        inter_chain_mask = (
            asym_rows.unsqueeze(-1) != expanded_asym.unsqueeze(-2)
        ).float() * pair_mask_2d
        iptm_per_row = (tm_expected * inter_chain_mask).sum(dim=-1) / (
            inter_chain_mask.sum(dim=-1) + _EPS
        )

        # chain_tm[c1][:, i] = mean over columns j in chain c1 of tm_expected[i, j].
        n_chains = int(expanded_asym.max().item()) + 1 if mask.shape[0] > 0 else 1
        chain_tm: list[Tensor | None] = []
        for c1 in range(n_chains):
            chain_c1 = (expanded_asym == c1).float() * mask_f
            if chain_c1.sum() == 0:
                chain_tm.append(None)
                continue
            col_mask = chain_c1.unsqueeze(-2)
            chain_tm.append(
                (tm_expected * col_mask).sum(dim=-1) / (col_mask.sum(dim=-1) + _EPS)
            )
        return {
            "pae": pae,
            "pde": pde,
            "ptm_per_row": ptm_per_row,
            "iptm_per_row": iptm_per_row,
            "chain_tm": chain_tm,
        }

    def _finish_scores(
        self,
        *,
        single: Tensor,
        scores: PairScores,
        x_pred: Tensor,
        distogram_atom_idx: Tensor,
        token_attention_mask: Tensor,
        atom_to_token: Tensor,
        atom_attention_mask: Tensor,
        asym_id: Tensor,
        mol_type: Tensor,
        num_diffusion_samples: int,
        interface_per_token: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """``_finish`` from whole-row ``_pair_scores``.

        ``interface_per_token``: :meth:`_interface_rows` over all rows, computed
        by the caller; ``None`` computes it here from ``x_pred``.
        """
        atom_to_token_m = self._repeat_batch(atom_to_token, num_diffusion_samples)
        atom_mask_m = self._repeat_batch(atom_attention_mask, num_diffusion_samples)
        rep_idx_m = self._repeat_batch(distogram_atom_idx, num_diffusion_samples).long()
        mask = self._repeat_batch(token_attention_mask, num_diffusion_samples)
        Bm = single.shape[0]

        atom_mask_f = atom_mask_m.float()
        s_at_atoms = gather_token_to_atom(single, atom_to_token_m)
        s_at_atoms_ln = self.plddt_ln(s_at_atoms)

        intra_idx = _compute_intra_token_idx(atom_to_token_m)
        intra_idx = intra_idx.clamp(max=self.plddt_weight.shape[0] - 1)
        w_plddt = self.plddt_weight[intra_idx]
        plddt_logits = torch.einsum("...c,...cb->...b", s_at_atoms_ln, w_plddt)
        plddt_per_atom = _categorical_mean(plddt_logits, start=0.0, end=1.0)

        L = single.shape[1]
        plddt_sum = torch.zeros(Bm, L, device=single.device, dtype=plddt_per_atom.dtype)
        atom_count = torch.zeros(
            Bm, L, device=single.device, dtype=plddt_per_atom.dtype
        )
        atom_mask_t = atom_mask_f.to(plddt_per_atom.dtype)
        plddt_sum.scatter_add_(1, atom_to_token_m, plddt_per_atom * atom_mask_t)
        atom_count.scatter_add_(1, atom_to_token_m, atom_mask_t)
        plddt = plddt_sum / atom_count.clamp(min=1e-6)

        complex_plddt = (plddt_per_atom * atom_mask_f).sum(dim=-1) / (
            atom_mask_f.sum(dim=-1) + _EPS
        )

        expanded_type = self._repeat_batch(mol_type, num_diffusion_samples)
        expanded_asym = self._repeat_batch(asym_id, num_diffusion_samples)
        is_ligand = (expanded_type == MOL_TYPE_NONPOLYMER).float()
        if interface_per_token is None:
            rep_coords = gather_rep_atom_coords(
                self._flatten_sample_axis(x_pred), rep_idx_m
            )
            interface_per_token = self._interface_rows(
                torch.cdist(
                    rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist"
                ),
                expanded_asym,
                expanded_asym,
                is_ligand,
            )
        iplddt_weight = torch.where(
            is_ligand.bool(),
            torch.full_like(interface_per_token, 2.0),
            interface_per_token,
        )
        iplddt_weight_atoms = gather_token_to_atom(
            iplddt_weight.unsqueeze(-1), atom_to_token_m
        ).squeeze(-1)
        atom_iplddt_w = atom_mask_f * iplddt_weight_atoms
        complex_iplddt = (plddt_per_atom * atom_iplddt_w).sum(dim=-1) / (
            atom_iplddt_w.sum(dim=-1) + _EPS
        )

        plddt_ca = plddt_per_atom.gather(1, rep_idx_m)

        pae = cast(Tensor | None, scores["pae"])
        pde = cast(Tensor | None, scores["pde"])

        # Resolved (per-atom binary).
        s_at_atoms_res = self.resolved_ln(s_at_atoms)
        w_res = self.resolved_weight[intra_idx]
        resolved_logits = torch.einsum("...c,...cb->...b", s_at_atoms_res, w_res)

        mask_f = mask.float()
        ptm = cast(Tensor, scores["ptm_per_row"]).max(dim=-1).values
        iptm = cast(Tensor, scores["iptm_per_row"]).max(dim=-1).values
        chain_tm = cast(list, scores["chain_tm"])
        n_chains = len(chain_tm)
        pair_chains_iptm = torch.zeros(
            Bm, n_chains, n_chains, device=ptm.device, dtype=ptm.dtype
        )
        # pair_chains_iptm[c1, c2] = max over rows i in chain c2 of the mean over
        # columns j in chain c1 of tm_expected[i, j] (max-of-row-mean, as in the
        # global iptm above), so iptm equals the max off-diagonal entry.
        for c1 in range(n_chains):
            avg_tm = chain_tm[c1]
            if avg_tm is None:
                continue
            for c2 in range(n_chains):
                chain_c2 = (expanded_asym == c2).float() * mask_f
                row_vals = avg_tm.masked_fill(chain_c2 == 0, float("-inf"))
                pair_chains_iptm[:, c1, c2] = row_vals.max(dim=-1).values.clamp(min=0.0)

        out = {
            "plddt_logits": plddt_logits,
            "plddt": plddt.detach(),
            "plddt_per_atom": plddt_per_atom.detach(),
            "plddt_ca": plddt_ca.detach(),
            "complex_plddt": complex_plddt.detach(),
            "complex_iplddt": complex_iplddt.detach(),
            "resolved_logits": resolved_logits,
            "ptm": ptm.detach(),
            "iptm": iptm.detach(),
            "pair_chains_iptm": pair_chains_iptm.detach(),
        }
        # Row-sharded, PAE and PDE are assembled on rank 0 only.
        if pae is not None:
            out["pae"] = pae
        if pde is not None:
            out["pde"] = pde
        return out


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


class EsmFold2Model(HubPreTrainedModel):
    """ESMFold2 — all-atom structure prediction with an ESMC PLM backbone.

    This is the standard released ESMFold2 architecture (uses a linear-
    recurrent trunk, internally referred to as "parcae").

    Forward kwargs that callers commonly override:

    * ``num_loops`` (default ``config.num_loops``): trunk refinement
      loops.
    * ``num_diffusion_samples`` (default ``config.num_diffusion_samples``):
      parallel structure samples; the confidence head re-runs once per
      sample, so memory scales linearly. Pass ``1`` for cheap inference.
    * ``num_sampling_steps`` (default ``config.structure_head.inference_num_steps``):
      diffusion ODE solver steps. Lower for speed, higher for quality.
    * ``noise_scale`` / ``step_scale`` / ``max_inference_sigma``: sampler
      overrides, forwarded to ``DiffusionStructureHead.sample``. The two scales
      default to ``config.structure_head``; the sigma cap truncates the schedule.
    * ``msa_max_depth`` / ``msa_column_mask_rate``: inference-time MSA diversity,
      defaulting to ``config.msa_encoder`` when ``None``.

    Unknown keywords raise ``TypeError``; only the inference-irrelevant keys the
    featurizer emits (``_IGNORED_FEATURE_KEYS``) are accepted and dropped.

    Memory / perf knobs:

    * ``model.set_chunk_size(int|None)``: caps L² ops (triangle / OPM /
      pair transition) at this token-axis chunk. Default 64 — fits
      L≈2k on an 80 GB GPU. Pass ``None`` for faster inference at L<600.
    * ``model.set_kernel_backend(None | "fused" | "fast")``:
      select kernel backend (None = reference path, "fused" = bit-exact
      Triton kernels, "fast" = those plus approximate ones).
    * ``model.set_memory_mode(None | "big")``: lowest peak memory, for inputs
      the other modes cannot hold. Inference only, and wants
      ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``. With
      ``shard_rows=True`` in a process group, "big" also splits the pair rows
      across the GPUs.
    """

    config_class = EsmFold2Config
    _keys_to_ignore_on_load_unexpected = [r"\._extra_state$"]
    # Allocated by ``ConfidenceHead.__init__`` and read by nothing; the upstream
    # port drops them on conversion, so its checkpoints do not carry them.
    _keys_to_ignore_on_load_missing = [
        r"^confidence_head\.(s_norm|s_inputs_to_single|s_input_to_s)\."
    ]

    @classmethod
    def _normalize_checkpoint_layout(
        cls, raw: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Translate a checkpoint out of the layout it was published in.

        Two of them: the upstream HuggingFace port renames most of the tree, and
        a bundled ESMC encoder packs its projections differently. Which one is in
        hand is read off the key names; ours passes straight through. The
        nothing-dropped accounting in ``_load_pretrained`` runs after this, so a
        remap that loses a tensor still fails the load.
        """
        if is_hf_layout(raw):
            return hf_state_dict_to_native(raw)
        return published_to_native_subtree(raw, "esmc.")

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__(config)
        d_inputs = config.single_inputs_size
        d_pair = config.pairwise_hidden_size

        self.inputs_embedder = InputsEmbedder(config)
        self.z_init_1 = nn.Linear(d_inputs, d_pair, bias=False)
        self.z_init_2 = nn.Linear(d_inputs, d_pair, bias=False)
        self.rel_pos = ResIdxAsymIdSymIdEntityIdEncoding(
            n_relative_residx_bins=config.n_relative_residx_bins,
            n_relative_chain_bins=config.n_relative_chain_bins,
            d_pair=d_pair,
        )
        self.token_bonds = nn.Linear(1, d_pair, bias=False)
        self.language_model = LanguageModelShim(
            d_z=d_pair, d_model=config.lm_d_model, num_layers=config.lm_num_layers
        )
        # A bundled backbone is described by ``esmc_config`` and arrives in the
        # same checkpoint, so it is built here and populated by the single
        # ``from_pretrained`` pass. Otherwise ``load_esmc`` fetches it later.
        self.esmc: nn.Module | None = (
            EsmcModel(config.esmc_config) if config.esmc_config is not None else None
        )
        # When True, ESM-C is offloaded to CPU after its one-shot hidden-state
        # computation (restored on the next call), freeing its ~6 GB (fp8) for the
        # recycling trunk + diffusion — the freed blocks stay in the caching pool
        # so the trunk reuses them without cudaMalloc, relieving the allocator
        # pressure that drove rank desync / spin-wait. Opt-in via
        # ``set_memory_mode("big")`` or ``wrap_model_with_cp(offload_esmc=...)``.
        self._offload_esmc: bool = False
        self._esmc_parked: bool = False
        self._memory_mode: str | None = None
        self._shard_rows = False

        self.folding_trunk = FoldingTrunk(
            n_layers=config.folding_trunk_num_hidden_layers,
            d_pair=d_pair,
            expansion_ratio=4,
        )
        if config.lm_encoder.enabled:
            self.lm_encoder: FoldingTrunk | None = FoldingTrunk(
                n_layers=config.lm_encoder.num_hidden_layers,
                d_pair=d_pair,
                expansion_ratio=4,
            )
        else:
            self.lm_encoder = None

        self.parcae_input_norm = nn.LayerNorm(d_pair)
        self.parcae_log_a = nn.Parameter(torch.zeros(d_pair))
        parcae_decay_init = math.sqrt(1.0 / 5.0)
        parcae_delta_init = -math.log(parcae_decay_init)
        self.parcae_log_delta = nn.Parameter(
            torch.full(
                (d_pair,), _inverse_softplus(parcae_delta_init), dtype=torch.float32
            )
        )
        self.parcae_b_cont = nn.Parameter(torch.eye(d_pair))
        self.parcae_readout = nn.Linear(d_pair, d_pair, bias=False)
        nn.init.eye_(self.parcae_readout.weight)
        self.parcae_coda = FoldingTrunk(
            n_layers=config.parcae_num_coda_layers, d_pair=d_pair, expansion_ratio=4
        )

        # Heads --------------------------------------------------------------
        self.structure_head = DiffusionStructureHead(config)
        self.distogram_head = nn.Linear(
            d_pair, config.structure_head.distogram_bins, bias=True
        )
        self.confidence_head = ConfidenceHead(config)

        msa_cfg = config.msa_encoder
        self.msa_encoder = None
        if msa_cfg.enabled:
            self.msa_encoder = MSAEncoder(
                d_msa=msa_cfg.hidden_size,
                d_pair=d_pair,
                d_inputs=d_inputs,
                d_hidden=msa_cfg.outer_hidden_size,
                n_layers=msa_cfg.num_hidden_layers,
                n_heads_msa=msa_cfg.num_attention_heads,
                msa_head_width=msa_cfg.head_width,
            )

        self.post_init()

    def load_esmc(self, esmc_model_path: str, precision: str = "fp8") -> None:
        """Fetch the ESMC LM from a separate repo and attach it.

        Only needed when the backbone is *not* bundled into this checkpoint; see
        ``EsmFold2Config.esmc_config``.
        """
        from esm.models.esmc import EsmcModel

        self.esmc = EsmcModel.from_pretrained(
            esmc_model_path,
            device=self.device,
            dtype=torch.float32 if precision == "fp32" else torch.bfloat16,
        )
        self.set_esmc_precision(precision)

    def set_esmc_precision(self, precision: str = "fp8") -> None:
        """Cast the attached ESMC LM in place and freeze it.

        ``precision``: ``"fp8"`` (default; ESMFold2 was trained on an fp8 LM),
        ``"bf16"`` or ``"fp32"``. ``"fp8"`` stores every projection weight in fp8.
        It runs fp8 GEMMs on compute capability >= 8.9 and upcasts elsewhere.
        """
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp8": torch.bfloat16,  # LayerNorms and embeddings stay bf16
        }
        if precision not in dtype_map:
            raise ValueError(
                f"precision must be one of {list(dtype_map)}, got {precision!r}"
            )
        if self.esmc is None:
            raise RuntimeError("no ESMC LM is attached; nothing to cast.")

        # No device move: converting before the move keeps the fp32 checkpoint off the GPU.
        esmc = self.esmc.to(dtype=dtype_map[precision]).eval()
        for p in esmc.parameters():
            p.requires_grad_(False)

        if precision == "fp8":
            convert_esmc_to_fp8_(esmc)

        self.esmc = esmc

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        load_esmc: bool = True,
        esmc_precision: str = "fp8",
        config: EsmFold2Config | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        local_dir = resolve_model_dir(pretrained_model_name_or_path, **kwargs)
        if config is None:
            config = EsmFold2Config.from_pretrained(
                local_dir, **default_module_flags(local_dir)
            )
            if cls is EsmFold2Model and config.type == "experimental":
                from esm.models.esmfold2.experimental import EsmFold2ExperimentalModel

                return EsmFold2ExperimentalModel.from_pretrained(
                    local_dir,
                    load_esmc=load_esmc,
                    esmc_precision=esmc_precision,
                    config=config,
                    device=device,
                    dtype=dtype,
                )
        # Built on the CPU and cast there: the bundled ESMC is stored in fp32, and
        # moving it first put the whole fp32 copy on the GPU.
        model = cls._load_pretrained(local_dir, config, device="cpu", dtype=dtype)
        # A bundled backbone came in with the trunk; only the separate-repo
        # arrangement needs a second fetch.
        if load_esmc and config.esmc_config is None:
            model.load_esmc(model.config.esmc_id, precision=esmc_precision)
        elif config.esmc_config is not None:
            model.set_esmc_precision(esmc_precision)
        return model.to(device)

    def set_kernel_backend(self, backend: str | None) -> None:
        """Select kernel backend.

        Args:
            backend: ``None`` (reference path), ``"fused"`` (vendored Triton
                kernels, bit-exact against the reference path), or ``"fast"``
                (those plus approximate, inference-only kernels: the MSA
                encoder's triangle multiplication, the fused diffusion and
                atom-transformer steps, and on sm_90 with Anthropic's
                ``opt_core`` installed, its pair-transition and TriMul
                kernels).

        Under ``"fast"`` the diffusion sampler stays fp32. Set
        ``ESMFOLD2_BF16_DIT_GEMM=1`` before importing this package to run the
        diffusion and atom-transformer GEMMs on bf16 operands, with the
        residual stream and normalisation statistics kept in fp32.
        """
        self.folding_trunk.set_kernel_backend(backend)
        self.inputs_embedder.set_kernel_backend(backend)
        if self.lm_encoder is not None:
            self.lm_encoder.set_kernel_backend(backend)
        self.parcae_coda.set_kernel_backend(backend)
        self.confidence_head.set_kernel_backend(backend)
        self.structure_head.set_kernel_backend(backend)
        if self.msa_encoder is not None:
            self.msa_encoder.set_kernel_backend(backend)

    def apply_torch_compile(
        self, mode: str = "fixed_seqlen", dynamic: bool | None = None
    ) -> None:
        """Compile L²-heavy blocks. ``mode='fixed_seqlen'`` recompiles per L; ``'dynamic_seqlen'`` compiles once.

        Does NOT stack with our Triton kernels — call ``set_kernel_backend(None)``
        before compiling. Raises if the fused backend is still selected.
        """
        import torch._dynamo

        from esm.models.esmfold2.layers import _KERNEL_BACKENDS

        stacked = [
            name
            for name, module in self.named_modules()
            if getattr(module, "_kernel_backend", None) in _KERNEL_BACKENDS
        ]
        if stacked:
            raise RuntimeError(
                "torch.compile does not stack with the fused Triton kernels: "
                f"{len(stacked)} module(s) still select the fused backend (e.g. "
                f"{stacked[0]!r}). Inductor cannot trace them - it fails with "
                "PendingUnbackedSymbolNotFound on some GPUs and silently diverges "
                "on others. Call set_kernel_backend(None) before compiling."
            )

        torch._dynamo.config.cache_size_limit = 512
        torch._dynamo.config.accumulated_cache_size_limit = 512
        # capture_scalar_outputs avoids graph breaks at .item() in atom-attention path.
        torch._dynamo.config.capture_scalar_outputs = True

        if dynamic is None:
            dynamic = mode == "dynamic_seqlen"
        kwargs: dict = {"dynamic": dynamic}

        from esm.models.esmfold2.layers import (
            DiffusionModule,
            DiffusionTransformer,
            PairUpdateBlock,
        )

        compile_targets = (
            PairUpdateBlock,
            DiffusionTransformer,
            DiffusionModule,
            MSAEncoderBlock,
        )

        def _maybe_compile(module: nn.Module) -> None:
            if isinstance(module, compile_targets):
                module.forward = torch.compile(module.forward, **kwargs)  # ty:ignore[invalid-assignment]

        self.apply(_maybe_compile)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)
        if self.lm_encoder is not None:
            self.lm_encoder.set_chunk_size(chunk_size)
        self.parcae_coda.set_chunk_size(chunk_size)
        self.confidence_head.set_chunk_size(chunk_size)
        if self.msa_encoder is not None:
            self.msa_encoder.set_chunk_size(chunk_size)

    def set_memory_mode(self, mode: str | None, *, shard_rows: bool = False) -> None:
        """Select the memory mode: ``None`` (default) or ``"big"``.

        "big" lowers peak GPU memory at some cost in speed, for inputs the
        other modes cannot hold. Inference only.

        ``shard_rows=True`` (with "big") splits every ``N x N`` pair tensor by
        rows across the ranks of an initialised process group, so the reachable
        input grows with the GPU count. Every rank then folds rank 0's input;
        without it each rank folds its own input whole (data parallel). It
        needs ``opt_core`` (see ``rowpair.OPT_CORE_HINT``), a process group
        (``torch.distributed.init_process_group``) and each rank on its own GPU
        (``torch.cuda.set_device(LOCAL_RANK)``). Sharded, a call folds one
        complex. Every rank returns the same coordinates and scores, but
        ``pae``, ``pde`` and ``distogram_logits`` exist on rank 0 only (``None``
        elsewhere), and ``pae_logits`` / ``pde_logits`` are not returned.
        """
        if mode not in _VALID_MEMORY_MODES:
            raise ValueError(
                f"memory mode must be one of {_VALID_MEMORY_MODES}, got {mode!r}"
            )
        if shard_rows and mode != MEMORY_BIG:
            raise ValueError('shard_rows needs set_memory_mode("big")')
        if shard_rows:
            rowpair.require_opt_core()
        self._memory_mode = mode
        self._shard_rows = shard_rows
        self.confidence_head.set_memory_mode(mode)
        self.structure_head.set_memory_mode(mode)

    def _park_esmc(self, n_tokens: int) -> bool:
        if self._memory_mode == MEMORY_BIG:
            return n_tokens >= _ESMC_OFFLOAD_MIN_TOKENS
        return self._offload_esmc

    def _lm_hidden_states(
        self,
        lm_hidden_states: Tensor | None,
        input_ids: Tensor | None,
        asym_id: Tensor,
        residue_index: Tensor,
        mol_type: Tensor,
        tok_mask: Tensor,
        lm_mask_pct: float | None,
    ) -> Tensor | None:
        """The given LM hidden states, else ESM-C's on ``input_ids``, else ``None``."""
        if lm_hidden_states is not None or input_ids is None or self.esmc is None:
            return lm_hidden_states
        return self._compute_lm_hidden_states(
            input_ids,
            asym_id,
            residue_index,
            mol_type,
            tok_mask,
            lm_mask_pct=(
                lm_mask_pct if lm_mask_pct is not None else self.config.lm_mask_pct
            ),
        )

    def _park_esmc_after_lm(self, n_tokens: int) -> None:
        """Move ESM-C to the host once the forward is done with it."""
        if not self._park_esmc(n_tokens) or self.esmc is None:
            return
        self.esmc.to("cpu")
        self._esmc_parked = True
        # Back to the driver, not the pool: cuSOLVER's handle allocates outside it.
        torch.cuda.empty_cache()

    def _msa_inputs(
        self,
        msa: Tensor | None,
        msa_attention_mask: Tensor | None,
        has_deletion: Tensor | None,
        deletion_value: Tensor | None,
        x_inputs: Tensor,
        msa_max_depth: int | None,
        msa_column_mask_rate: float | None,
    ) -> dict | None:
        """The MSA encoder's per-loop inputs, column-masked once; ``None`` without an MSA.

        ``None`` depth or rate means the checkpoint's value.
        """
        if self.msa_encoder is None or msa is None:
            return None
        msa_cfg = self.config.msa_encoder
        depth = msa_cfg.max_depth if msa_max_depth is None else msa_max_depth
        rate = (
            msa_cfg.column_mask_rate
            if msa_column_mask_rate is None
            else msa_column_mask_rate
        )
        return dict(
            msa=msa,
            msa_attention_mask=maybe_apply_msa_column_masking(
                msa_attention_mask, rate=rate
            ),
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            x_inputs=x_inputs,
            max_depth=depth,
            subsample_enabled=depth is not None,
        )

    def _compute_lm_hidden_states(
        self,
        input_ids: Tensor,
        asym_id: Tensor,
        residue_index: Tensor,
        mol_type: Tensor,
        tok_mask: Tensor,
        lm_mask_pct: float = 0.0,
    ) -> Tensor:
        assert self.esmc is not None
        if self._esmc_parked:
            self.esmc.to(self.device)
            self._esmc_parked = False
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return compute_lm_hidden_states(
                self.esmc,
                input_ids,
                asym_id,
                residue_index,
                mol_type,
                tok_mask,
                lm_mask_pct=lm_mask_pct,
            )

    def _discretized_dynamics(self) -> tuple[Tensor, Tensor]:
        delta = F.softplus(self.parcae_log_delta)
        a = torch.exp(-delta * torch.exp(self.parcae_log_a))
        b = delta[:, None] * self.parcae_b_cont
        return a, b

    def _inject_pair(
        self, z: Tensor, z_inject_pair: Tensor, a: Tensor, b_mat: Tensor
    ) -> Tensor:
        if not _memory_big(self):
            injected = self.parcae_input_norm(z_inject_pair)
            return a * z + F.linear(injected.to(z.dtype), b_mat)
        for lo, hi in row_bands(z.shape[1]):
            injected = self.parcae_input_norm(z_inject_pair[:, lo:hi])
            z[:, lo:hi] = a * z[:, lo:hi] + F.linear(injected.to(z.dtype), b_mat)
        return z

    def _distogram_logits(self, z: Tensor) -> Tensor:
        if not (_memory_big(self) and z.is_cuda):
            return self.distogram_head(z + z.transpose(-2, -3))
        logits: Tensor | None = None
        for lo, hi in row_bands(z.shape[1]):
            band = self.distogram_head(z[:, lo:hi] + z[:, :, lo:hi].transpose(-2, -3))
            if logits is None:
                logits = torch.empty(
                    (*z.shape[:2], *band.shape[2:]), dtype=band.dtype, pin_memory=True
                )
            logits[:, lo:hi].copy_(band)
        assert logits is not None
        return logits

    def _init_pair_state(self, ref: Tensor) -> Tensor:
        # trunc_normal_ by hand: torch after 2.11 switched it to rejection sampling,
        # which changes the draw for a given seed and so every seeded fold.
        std = math.sqrt(2.0 / (5.0 * ref.shape[-1]))
        bound = math.erf(3.0 / math.sqrt(2.0))
        state = torch.empty_like(ref, dtype=torch.float32).uniform_(-bound, bound)
        state.erfinv_().mul_(std * math.sqrt(2.0)).clamp_(-3 * std, 3 * std)
        return state.to(dtype=ref.dtype)

    def _init_pair_state_rows(
        self, layout: rowpair.Layout, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        """This rank's rows of ``_init_pair_state``, from rows of the same draw."""
        channels = self.config.pairwise_hidden_size
        std = math.sqrt(2.0 / (5.0 * channels))
        bound = math.erf(3.0 / math.sqrt(2.0))
        return rowpair.uniform_rows(
            layout,
            channels,
            -bound,
            bound,
            device,
            transform=lambda t: (
                t.erfinv_().mul_(std * math.sqrt(2.0)).clamp_(-3 * std, 3 * std)
            ),
            dtype=dtype,
        )

    def _lm_pair_rows(
        self, lm_tokens: Tensor, layout: rowpair.Layout, p: float, dtype: torch.dtype
    ) -> Tensor:
        """This rank's rows of the LM pair, dropped out at ``p``, in ``dtype``."""
        return rowpair.dropout_rows(
            lambda g0, g1: self.language_model.pair_from_tokens(
                lm_tokens, rows=(g0, g1)
            ).float(),
            layout,
            p,
            self.config.pairwise_hidden_size,
            dtype,
            lm_tokens.device,
        )

    def shards_pair_rows(self) -> bool:
        """True when ``forward`` splits pair rows over the process group's ranks.

        That is ``set_memory_mode("big", shard_rows=True)``. Inputs too small
        to split are still folded whole on every rank.
        """
        return self._shard_rows

    def _row_encoders(
        self,
        layout: rowpair.Layout,
        x_inputs: Tensor,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        token_index: Tensor,
        token_bonds: Tensor,
        use_amp: bool,
    ) -> tuple[RowsFn, RowsFn, RowsFn]:
        """``(rel_rows, bonds_rows, z_init_rows)``: local pair rows ``[lo, hi)``.

        Each reader rebuilds the band it needs rather than holding the shard;
        the bands run under the autocast ``forward`` builds them in.
        """
        r0 = layout.r0

        def amp() -> torch.autocast:
            return torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16)

        def rel_rows(lo: int, hi: int) -> Tensor:
            with amp():
                return self.rel_pos(
                    residue_index=residue_index,
                    asym_id=asym_id,
                    sym_id=sym_id,
                    entity_id=entity_id,
                    token_index=token_index,
                    rows=(r0 + lo, r0 + hi),
                )

        def bonds_rows(lo: int, hi: int) -> Tensor:
            with amp():
                return self.token_bonds(token_bonds[:, r0 + lo : r0 + hi].float())

        def z_init_rows(lo: int, hi: int) -> Tensor:
            with amp():
                z_init = self.z_init_1(x_inputs[:, r0 + lo : r0 + hi]).unsqueeze(
                    2
                ) + self.z_init_2(x_inputs).unsqueeze(1)
                return z_init + rel_rows(lo, hi) + bonds_rows(lo, hi)

        return rel_rows, bonds_rows, z_init_rows

    def _check_row_sharding(
        self, tok_mask: Tensor, include_embeddings: bool, *replicated: Tensor | None
    ) -> None:
        """Refuse what the row-sharded fold cannot do; prove the inputs match."""
        if tok_mask.shape[0] != 1:
            raise ValueError(
                "ESMFold2: multi-GPU 'big' folds one input per call, "
                f"got batch {tok_mask.shape[0]}."
            )
        if include_embeddings:
            raise NotImplementedError(
                "ESMFold2: include_embeddings needs the whole pair; "
                "it is not available on several GPUs."
            )
        if getattr(self, "_cp_recycle_engine", None) is not None:
            raise RuntimeError(
                "ESMFold2: a Fold-CP wrapped model cannot also row-shard; "
                "use one or the other."
            )
        for i, t in enumerate(replicated):
            if t is not None:
                rowpair.guard_replicated(t.contiguous(), f"input {i}")

    def _distogram_logits_rows(
        self, z: Tensor, layout: rowpair.Layout
    ) -> Tensor | None:
        """The whole distogram in rank 0's host memory; ``None`` on other ranks.

        ``z^T`` arrives one band at a time, so only a band of it is on a device.
        """
        R = z.shape[1]
        rows = rowpair.TRANSITION_ROWS
        logits: Tensor | None = None
        # Every rank joins every band exchange, so all walk the largest shard.
        for lo in range(0, layout.Rmax, rows):
            z_t = rowpair.transpose_band_rows(z, layout, lo, lo + rows)
            if z_t is None:
                continue
            hi = lo + z_t.shape[1]
            band = self.distogram_head(z[:, lo:hi].float() + z_t.float())
            del z_t
            if logits is None:
                logits = band.new_empty((1, R, *band.shape[2:]))
            logits[:, lo:hi] = band
        assert logits is not None
        return rowpair.gather_rows_to_rank0_host(logits, layout)

    def _msa_step_inputs(
        self, _msa_inputs: dict, tok_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(msa_oh, msa_attn, has_deletion, deletion_value)`` of one loop step.

        Fresh row subsample each iteration (column mask was applied once in
        forward, before the loop).
        """
        msa_i, mask_i, hd_i, dv_i = maybe_subsample_msa(
            _msa_inputs["msa"],
            _msa_inputs["msa_attention_mask"],
            _msa_inputs["has_deletion"],
            _msa_inputs["deletion_value"],
            max_depth=_msa_inputs["max_depth"],
            enabled=_msa_inputs["subsample_enabled"],
        )
        B_msa, M, L_msa = msa_i.shape
        msa_oh = F.one_hot(
            msa_i.permute(0, 2, 1).long(), num_classes=NUM_RES_TYPES
        ).float()
        msa_attn = (
            mask_i.permute(0, 2, 1).float()
            if mask_i is not None
            else tok_mask[:, :, None].expand(-1, -1, M).float()
        )
        # Bias-free MSAEncoder.embed requires zeroed padding.
        msa_oh = msa_oh * msa_attn.unsqueeze(-1)
        hd = (
            hd_i.permute(0, 2, 1).float()
            if hd_i is not None
            else torch.zeros(B_msa, L_msa, M, device=msa_i.device)
        )
        dv = (
            dv_i.permute(0, 2, 1).float()
            if dv_i is not None
            else torch.zeros(B_msa, L_msa, M, device=msa_i.device)
        )
        return msa_oh, msa_attn, hd, dv

    def _run_one_loop(
        self,
        z: Tensor,
        z_init: Tensor,
        lm_z: Tensor | None,
        _msa_inputs: dict | None,
        pair_mask: Tensor,
        a: Tensor,
        b_mat: Tensor,
        tok_mask: Tensor,
        total_steps: int,
    ) -> Tensor:
        # CP orchestrator seam: when a context-parallel recycle engine has been
        # installed (by ``wrap_model_with_cp``), delegate the whole loop to it so
        # the pair stays sharded across iterations instead of round-tripping
        # through ``full_tensor()`` at every module boundary. Plain ``getattr``
        # default keeps this file CP-agnostic (no distributed import). The engine
        # returns a gathered full tensor, so the caller is unchanged.
        _cp_engine = getattr(self, "_cp_recycle_engine", None)
        if _cp_engine is not None:
            return _cp_engine.run_loop(
                self,
                z=z,
                z_init=z_init,
                lm_z=lm_z,
                msa_inputs=_msa_inputs,
                pair_mask=pair_mask,
                a=a,
                b_mat=b_mat,
                tok_mask=tok_mask,
                total_steps=total_steps,
            )

        # Helper method (not inline) so per-iter locals free on return —
        # otherwise leaks ~2 GB L²×c_z into distogram/sample scope.
        # training=True forces dropout under eval(), matching the per-loop
        # dropout strategy used at train time.
        lm_cfg = self.config.lm_encoder
        _per_loop_lm_dropout = (
            lm_z is not None
            and getattr(lm_cfg, "per_loop_lm_dropout", False)
            and getattr(lm_cfg, "lm_dropout", 0.0) > 0.0
        )
        _lm_dropout_p = getattr(lm_cfg, "lm_dropout", 0.0)

        for _ in range(total_steps):
            if _per_loop_lm_dropout:
                assert lm_z is not None  # narrowed by _per_loop_lm_dropout
                lm_z_i: Tensor | None = F.dropout(lm_z, p=_lm_dropout_p, training=True)
            else:
                lm_z_i = lm_z

            refined_lm_z: Tensor | None = None
            if lm_z_i is not None and self.lm_encoder is not None:
                lm_in = lm_z_i.to(z_init.dtype)
                # The fp32 draw is dead once cast; holding it set the fold's peak.
                lm_z_i = None
                refined_lm_z = self.lm_encoder(lm_in, pair_attention_mask=pair_mask)
                del lm_in

            z_inject_pair = z_init
            if lm_z_i is not None and self.lm_encoder is None:
                z_inject_pair = z_inject_pair + lm_z_i.to(z_inject_pair.dtype)

            if self.msa_encoder is not None and _msa_inputs is not None:
                msa_oh, msa_attn, hd, dv = self._msa_step_inputs(_msa_inputs, tok_mask)
                msa_pair = self.msa_encoder(
                    x_pair=z_inject_pair,
                    x_inputs=_msa_inputs["x_inputs"],
                    msa_oh=msa_oh,
                    has_deletion=hd,
                    deletion_value=dv,
                    msa_attention_mask=msa_attn,
                ).to(z_inject_pair.dtype)
                z_inject_pair = (
                    msa_pair
                    if self.config.msa_encoder.overwrite
                    else (z_inject_pair + msa_pair)
                )

            if refined_lm_z is not None:
                z_inject_pair = z_inject_pair + refined_lm_z.to(z_inject_pair.dtype)

            z = self._inject_pair(z, z_inject_pair, a, b_mat)
            del z_inject_pair, refined_lm_z, lm_z_i
            z = self.folding_trunk(z, pair_attention_mask=pair_mask)

        return z

    def _run_one_loop_rows(
        self,
        z: Tensor,
        z_init_rows: RowsFn,
        lm_tokens: Tensor | None,
        _msa_inputs: dict | None,
        pair_mask: Tensor,
        a: Tensor,
        b_mat: Tensor,
        tok_mask: Tensor,
        total_steps: int,
        layout: rowpair.Layout,
    ) -> Tensor:
        """``_run_one_loop`` on this rank's rows, in the same order.

        ``z`` and ``pair_mask`` are rows; ``lm_tokens`` is the ``[B, L, d_z]``
        LM token state. Neither ``z_init`` nor the LM pair is held across the
        loop: each step rebuilds them per band, and the inject reads each band
        once, so the only pair shards alive are ``z`` and one step's LM rows.
        """
        lm_cfg = self.config.lm_encoder
        per_loop_dropout = (
            getattr(lm_cfg, "per_loop_lm_dropout", False)
            and getattr(lm_cfg, "lm_dropout", 0.0) > 0.0
        )
        p = getattr(lm_cfg, "lm_dropout", 0.0) if per_loop_dropout else 0.0
        R = z.shape[1]

        def whole(rows_fn: RowsFn) -> Tensor:
            bands = row_bands(R)
            first = rows_fn(*bands[0])
            out = first.new_empty((*first.shape[:1], R, *first.shape[2:]))
            for lo, hi in bands:
                out[:, lo:hi] = first if lo == 0 else rows_fn(lo, hi)
            return out

        for _ in range(total_steps):
            lm_rows = (
                None
                if lm_tokens is None
                else self._lm_pair_rows(lm_tokens, layout, p, z.dtype)
            )
            refined_lm_z: Tensor | None = None
            if lm_rows is not None and self.lm_encoder is not None:
                # The row trunk overwrites lm_rows: it becomes refined_lm_z.
                refined_lm_z = self.lm_encoder(
                    lm_rows, pair_attention_mask=pair_mask, inplace=True
                )
                lm_rows = None

            def base_rows(lo: int, hi: int, lm: Tensor | None = lm_rows) -> Tensor:
                band = z_init_rows(lo, hi)
                if lm is not None:
                    band = band + lm[:, lo:hi].to(band.dtype)
                return band

            msa_pair: Tensor | None = None
            if self.msa_encoder is not None and _msa_inputs is not None:
                msa_oh, msa_attn, hd, dv = self._msa_step_inputs(_msa_inputs, tok_mask)
                x_pair = whole(base_rows)
                msa_pair = self.msa_encoder(
                    x_pair=x_pair,
                    x_inputs=_msa_inputs["x_inputs"],
                    msa_oh=msa_oh,
                    has_deletion=hd,
                    deletion_value=dv,
                    msa_attention_mask=msa_attn,
                ).to(x_pair.dtype)
                del x_pair, msa_oh, msa_attn, hd, dv
            overwrite = self.config.msa_encoder.overwrite

            for lo, hi in row_bands(R):
                if msa_pair is not None and overwrite:
                    inject = msa_pair[:, lo:hi]
                else:
                    inject = base_rows(lo, hi)
                    if msa_pair is not None:
                        inject = inject + msa_pair[:, lo:hi]
                if refined_lm_z is not None:
                    inject = inject + refined_lm_z[:, lo:hi].to(inject.dtype)
                injected = self.parcae_input_norm(inject)
                z[:, lo:hi] = a * z[:, lo:hi] + F.linear(injected.to(z.dtype), b_mat)
                del inject, injected
            del refined_lm_z, msa_pair, lm_rows
            z = self.folding_trunk(z, pair_attention_mask=pair_mask, inplace=True)

        return z

    def _forward_rows(
        self,
        layout: rowpair.Layout,
        *,
        x_inputs: Tensor,
        use_amp: bool,
        total_steps: int,
        n_samples: int,
        token_index: Tensor,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        mol_type: Tensor,
        token_bonds: Tensor,
        tok_mask: Tensor,
        atm_mask: Tensor,
        disto_idx: Tensor,
        atom_to_token: Tensor,
        ref_pos: Tensor,
        ref_charge: Tensor,
        ref_element_oh: Tensor,
        ref_atom_name_chars_oh: Tensor,
        ref_space_uid: Tensor,
        msa: Tensor | None,
        has_deletion: Tensor | None,
        deletion_value: Tensor | None,
        msa_attention_mask: Tensor | None,
        msa_max_depth: int | None,
        msa_column_mask_rate: float | None,
        input_ids: Tensor | None,
        lm_hidden_states: Tensor | None,
        lm_mask_pct: float | None,
        num_sampling_steps: int | None,
        noise_scale: float | None,
        step_scale: float | None,
        max_inference_sigma: float | None,
    ) -> dict[str, Tensor]:
        """``forward`` from the input embedding on, with pair rows split over ranks.

        The statements and their random draws run in ``forward``'s order. No
        ``N x N`` tensor is whole on a device; ``distogram_logits`` lands in rank
        0's host memory and is absent on the other ranks.
        """
        # Kernels need not be deterministic; the replicated inputs must be.
        rowpair.broadcast_(x_inputs)
        rel_rows, bonds_rows, z_init_rows = self._row_encoders(
            layout,
            x_inputs,
            residue_index,
            asym_id,
            sym_id,
            entity_id,
            token_index,
            token_bonds,
            use_amp,
        )
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            lm_hidden_states = self._lm_hidden_states(
                lm_hidden_states,
                input_ids,
                asym_id,
                residue_index,
                mol_type,
                tok_mask,
                lm_mask_pct,
            )
            lm_tokens: Tensor | None = None
            if lm_hidden_states is not None:
                lm_tokens = self.language_model.token_state(
                    lm_hidden_states.detach()
                ).contiguous()
                rowpair.broadcast_(lm_tokens)
            del lm_hidden_states
            self._park_esmc_after_lm(tok_mask.shape[1])

            pair_mask = (
                rowpair.rows_of(tok_mask, layout)[:, :, None].float()
                * tok_mask[:, None, :].float()
            )
            z = self._init_pair_state_rows(
                layout, z_init_rows(0, 1).dtype, x_inputs.device
            )
            a, b = self._discretized_dynamics()
            a = a.view(1, 1, 1, -1).to(device=z.device, dtype=z.dtype)
            b_mat = b.to(device=z.device, dtype=z.dtype)

            _msa_inputs = self._msa_inputs(
                msa,
                msa_attention_mask,
                has_deletion,
                deletion_value,
                x_inputs,
                msa_max_depth,
                msa_column_mask_rate,
            )

            with rowpair.sharded(layout):
                z = self._run_one_loop_rows(
                    z,
                    z_init_rows,
                    lm_tokens,
                    _msa_inputs,
                    pair_mask,
                    a,
                    b_mat,
                    tok_mask,
                    total_steps,
                    layout,
                )
                del lm_tokens, _msa_inputs, a, b_mat
                for lo, hi in row_bands(z.shape[1]):
                    z[:, lo:hi] = self.parcae_readout(z[:, lo:hi])
                z = self.parcae_coda(z, pair_attention_mask=pair_mask, inplace=True)
        # z stays in the trunk's dtype; each reader upcasts its band, which is exact.

        distogram_logits = self._distogram_logits_rows(z, layout)

        head = self.structure_head
        N = layout.N
        route = rowpair.sampler_route(
            N * N * head.whole_sampler_bytes_per_pair(z.device), z.device
        )
        pair_conditioning: Tensor | None = None
        bias_rows: list[Tensor] | None = None
        if route == "whole":
            pair_conditioning = rowpair.all_gather_rows(
                head.pair_conditioning_rows(z, rel_rows), layout
            )
        else:
            with rowpair.sharded(layout):
                bias_rows = head.pair_bias_rows(z, rel_rows)
        sampler_pair = pair_conditioning if bias_rows is None else bias_rows[0]
        assert sampler_pair is not None
        # The whole route runs the single-device DiT, so it must not see the layout.
        with rowpair.sharded(layout) if bias_rows is not None else nullcontext():
            structure_output = self.structure_head.sample(
                z_trunk=sampler_pair,
                s_inputs=x_inputs,
                s_trunk=None,
                relative_position_encoding=sampler_pair,
                ref_pos=ref_pos,
                ref_charge=ref_charge,
                ref_mask=atm_mask,
                ref_element=ref_element_oh,
                ref_atom_name_chars=ref_atom_name_chars_oh,
                ref_space_uid=ref_space_uid,
                tok_idx=atom_to_token,
                asym_id=asym_id,
                residue_index=residue_index,
                entity_id=entity_id,
                token_index=token_index,
                sym_id=sym_id,
                token_attention_mask=tok_mask,
                num_diffusion_samples=n_samples,
                num_sampling_steps=num_sampling_steps,
                noise_scale=noise_scale,
                step_scale=step_scale,
                max_inference_sigma=max_inference_sigma,
                return_atom_repr=False,
                pair_conditioning=pair_conditioning,
                pair_bias_rows=bias_rows,
                replica_sync=rowpair.sync_replicated,
            )
        del pair_conditioning, bias_rows, sampler_pair
        sample_coords = structure_output["sample_atom_coords"]
        assert sample_coords is not None
        # Every rank sampled the same noise; rank 0's result is the one kept.
        sample_coords = rowpair.broadcast_(sample_coords.contiguous())

        with rowpair.sharded(layout):
            single, scores, interface = self.confidence_head.scores_rows(
                s_inputs=x_inputs.detach(),
                z=z,
                rel_rows=rel_rows,
                bonds_rows=bonds_rows,
                x_pred=sample_coords.detach(),
                distogram_atom_idx=disto_idx,
                token_attention_mask=tok_mask,
                asym_id=asym_id,
                mol_type=mol_type,
                num_diffusion_samples=n_samples,
                layout=layout,
            )
        # The pair rows are dead; the finish only reads per-token and small tensors.
        del z
        confidence_output = self.confidence_head._finish_scores(
            single=single,
            scores=scores,
            x_pred=sample_coords.detach(),
            distogram_atom_idx=disto_idx,
            token_attention_mask=tok_mask,
            atom_to_token=atom_to_token,
            atom_attention_mask=atm_mask,
            asym_id=asym_id,
            mol_type=mol_type,
            num_diffusion_samples=n_samples,
            interface_per_token=interface,
        )
        output: dict[str, Tensor] = {"sample_atom_coords": sample_coords}
        if distogram_logits is not None:
            output["distogram_logits"] = distogram_logits
        output.update(confidence_output)
        output["atom_pad_mask"] = (
            atm_mask.unsqueeze(0) if atm_mask.dim() == 1 else atm_mask
        )
        output["residue_index"] = residue_index
        output["entity_id"] = entity_id
        return output

    @torch.inference_mode()
    def forward(
        self,
        token_index: Tensor,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        mol_type: Tensor,
        res_type: Tensor,
        token_bonds: Tensor,
        token_attention_mask: Tensor,
        ref_pos: Tensor,
        ref_element: Tensor,
        ref_charge: Tensor,
        ref_atom_name_chars: Tensor,
        ref_space_uid: Tensor,
        atom_attention_mask: Tensor,
        atom_to_token: Tensor,
        distogram_atom_idx: Tensor,
        deletion_mean: Tensor | None = None,
        msa: Tensor | None = None,
        has_deletion: Tensor | None = None,
        deletion_value: Tensor | None = None,
        msa_attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        lm_hidden_states: Tensor | None = None,
        num_loops: int | None = None,
        num_diffusion_samples: int | None = None,
        num_sampling_steps: int | None = None,
        lm_mask_pct: float | None = None,
        msa_max_depth: int | None = None,
        msa_column_mask_rate: float | None = None,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        max_inference_sigma: float | None = 256.0,
        disto_cond: Tensor | None = None,
        disto_cond_mask: Tensor | None = None,
        include_embeddings: bool = False,
        **unused_features: Tensor,
    ) -> dict[str, Tensor]:
        unexpected = sorted(set(unused_features) - _IGNORED_FEATURE_KEYS)
        if unexpected:
            raise TypeError(
                f"{type(self).__name__}.forward() got unexpected keyword "
                f"argument(s) {unexpected}."
            )

        # Named rather than left to **kwargs so conditioning cannot be swallowed
        # and the fold silently run unconditioned. No released checkpoint carries
        # the disto_conditioning_proj weights that would consume these.
        if disto_cond_mask is not None and disto_cond_mask.any():
            raise NotImplementedError(
                "distogram conditioning is not implemented for ESMFold2; "
                "fold without it."
            )

        tok_mask = token_attention_mask
        atm_mask = atom_attention_mask
        disto_idx = distogram_atom_idx

        layout: rowpair.Layout | None = None
        if self.shards_pair_rows():
            rowpair.require_process_group(self.device)
            layout = rowpair.layout_for(tok_mask.shape[1])
        if layout is not None:
            self._check_row_sharding(
                tok_mask,
                include_embeddings,
                token_index,
                residue_index,
                asym_id,
                res_type,
                ref_pos,
                atom_to_token,
                msa,
                msa_attention_mask,
            )
            rowpair.sync_rng_state(ref_pos.device)

        n_loops: int = num_loops if num_loops is not None else self.config.num_loops
        n_samples: int = (
            num_diffusion_samples
            if num_diffusion_samples is not None
            else self.config.num_diffusion_samples
        )
        total_steps = max(1, n_loops + 1)

        if res_type.dim() == 2:
            res_type_oh = F.one_hot(res_type.long(), num_classes=NUM_RES_TYPES).float()
            res_type_oh = res_type_oh * tok_mask.unsqueeze(-1).float()
        else:
            res_type_oh = res_type.float()

        if msa is not None:
            msa_oh_profile = F.one_hot(msa.long(), num_classes=NUM_RES_TYPES).float()
            if msa_attention_mask is not None:
                mask_f = msa_attention_mask.float().unsqueeze(-1)
                msa_oh_profile = msa_oh_profile * mask_f
                valid_seq_count = msa_attention_mask.float().sum(dim=1).clamp(min=1)
                profile = msa_oh_profile.sum(dim=1) / valid_seq_count.unsqueeze(-1)
            else:
                profile = msa_oh_profile.mean(dim=1)
        else:
            profile = res_type_oh

        if deletion_mean is None:
            deletion_mean = torch.zeros(
                res_type.shape[0], res_type.shape[1], device=res_type.device
            )

        ref_element_oh = F.one_hot(
            ref_element.long(), num_classes=MAX_ATOMIC_NUMBER
        ).float()
        ref_atom_name_chars_oh = F.one_hot(
            ref_atom_name_chars.long(), num_classes=CHAR_VOCAB_SIZE
        ).float()
        # Bias-free downstream Linears require zeroed padding.
        atm_mask_f = atm_mask.float()
        ref_element_oh = ref_element_oh * atm_mask_f.unsqueeze(-1)
        ref_atom_name_chars_oh = ref_atom_name_chars_oh * atm_mask_f.unsqueeze(
            -1
        ).unsqueeze(-1)
        atom_to_token = atom_to_token * atm_mask.long()

        use_amp = ref_pos.device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            x_inputs = self.inputs_embedder(
                aatype=res_type_oh,
                profile=profile.float(),
                deletion_mean=deletion_mean.float(),
                ref_pos=ref_pos,
                atom_attention_mask=atm_mask,
                ref_space_uid=ref_space_uid,
                ref_charge=ref_charge,
                ref_element=ref_element_oh,
                ref_atom_name_chars=ref_atom_name_chars_oh,
                atom_to_token=atom_to_token,
            )

        if layout is not None:
            return self._forward_rows(
                layout,
                x_inputs=x_inputs,
                use_amp=use_amp,
                total_steps=total_steps,
                n_samples=n_samples,
                token_index=token_index,
                residue_index=residue_index,
                asym_id=asym_id,
                sym_id=sym_id,
                entity_id=entity_id,
                mol_type=mol_type,
                token_bonds=token_bonds,
                tok_mask=tok_mask,
                atm_mask=atm_mask,
                disto_idx=disto_idx,
                atom_to_token=atom_to_token,
                ref_pos=ref_pos,
                ref_charge=ref_charge,
                ref_element_oh=ref_element_oh,
                ref_atom_name_chars_oh=ref_atom_name_chars_oh,
                ref_space_uid=ref_space_uid,
                msa=msa,
                has_deletion=has_deletion,
                deletion_value=deletion_value,
                msa_attention_mask=msa_attention_mask,
                msa_max_depth=msa_max_depth,
                msa_column_mask_rate=msa_column_mask_rate,
                input_ids=input_ids,
                lm_hidden_states=lm_hidden_states,
                lm_mask_pct=lm_mask_pct,
                num_sampling_steps=num_sampling_steps,
                noise_scale=noise_scale,
                step_scale=step_scale,
                max_inference_sigma=max_inference_sigma,
            )

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            # CP (sharded pair-init): build z_init / rel_pos / token_bonds directly
            # as 2-D-sharded DTensors so the full L×L pair is never resident on any
            # rank (the init-phase peak that made CP OOM at short L). Installed with
            # the recycle engine; plain getattr keeps this file CP-agnostic.
            _cp_pair_init = getattr(self, "_cp_pair_init", None)
            if _cp_pair_init is not None:
                (
                    z_init,
                    relative_position_encoding,
                    token_bonds_encoding,
                    _pi_n_orig,
                ) = _cp_pair_init(
                    x_inputs,
                    residue_index,
                    asym_id,
                    sym_id,
                    entity_id,
                    token_index,
                    token_bonds,
                )
            else:
                z_init = self.z_init_1(x_inputs).unsqueeze(2) + self.z_init_2(
                    x_inputs
                ).unsqueeze(1)

                relative_position_encoding = self.rel_pos(
                    residue_index=residue_index,
                    asym_id=asym_id,
                    sym_id=sym_id,
                    entity_id=entity_id,
                    token_index=token_index,
                )
                token_bonds_encoding = self.token_bonds(token_bonds.float())
                z_init = z_init + relative_position_encoding + token_bonds_encoding
                _pi_n_orig = z_init.shape[1]

            lm_hidden_states = self._lm_hidden_states(
                lm_hidden_states,
                input_ids,
                asym_id,
                residue_index,
                mol_type,
                tok_mask,
                lm_mask_pct,
            )
            lm_z: Tensor | None = None
            if lm_hidden_states is not None:
                # CP (#14): when a distributed LM->pair builder is installed, emit
                # lm_z as a sharded (Shard0,Shard1,Shard2) DTensor so the full L×L
                # pair is never resident on any rank (the binding peak per the T0
                # phase sweep). Consumed by the recycle engine, installed together
                # with it. Plain getattr keeps this file CP-agnostic.
                # Only emit a sharded lm_z when the recycle engine (its sole
                # consumer) is active; the serial _run_one_loop expects a full
                # tensor. Keeps the gather-per-iteration fallback path valid.
                _cp_lm = getattr(self, "_cp_language_model", None)
                if (
                    _cp_lm is not None
                    and getattr(self, "_cp_recycle_engine", None) is not None
                ):
                    lm_z, _ = _cp_lm(lm_hidden_states.detach())
                else:
                    lm_z = self.language_model(lm_hidden_states.detach())
            del lm_hidden_states
            # ESM-C is done for this forward: free its ~6 GB (fp8) for the trunk and diffusion.
            self._park_esmc_after_lm(tok_mask.shape[1])

            # Pair attention mask: sharded block under CP (outer product of the
            # sliced row/col token-mask — never the full L×L), else the serial
            # full outer product.
            if _cp_pair_init is not None:
                pair_mask = _cp_pair_init.pair_mask(tok_mask)
            else:
                pair_mask = tok_mask[:, :, None].float() * tok_mask[:, None, :].float()

            # Initial pair state: sharded random block under CP (never full L×L),
            # else the serial full draw.
            if _cp_pair_init is not None:
                z = _cp_pair_init.init_pair_state(z_init, _pi_n_orig)
            else:
                z = self._init_pair_state(z_init)

            a, b = self._discretized_dynamics()
            a = a.view(1, 1, 1, -1).to(device=z.device, dtype=z.dtype)
            b_mat = b.to(device=z.device, dtype=z.dtype)

            # Inference-time MSA diversity: column mask is applied once here
            # (shared across recycling loops); row subsampling is deferred to
            # per-iter inside _run_one_loop (fresh subset per loop).
            _msa_inputs = self._msa_inputs(
                msa,
                msa_attention_mask,
                has_deletion,
                deletion_value,
                x_inputs,
                msa_max_depth,
                msa_column_mask_rate,
            )

            # CP tail (#7): when a recycle engine is installed, keep the pair
            # SHARDED through the recycle loop + parcae so the full L×L pair is
            # not resident during the expensive structure phase. The pair lives as
            # ``_z_dt`` (DTensor); it is gathered only transiently for the heads
            # that don't yet consume sharded input (distogram, confidence). Plain
            # ``getattr`` keeps this file CP-agnostic.
            _cp_engine = getattr(self, "_cp_recycle_engine", None)
            _z_dt = None
            # z may be a sharded DTensor (padded) under CP, so use the pre-pad
            # token count tracked at pair-init rather than z.shape[1].
            _n_orig = _pi_n_orig
            if _cp_engine is not None:
                _z_dt, _n_orig = _cp_engine.run_loop(
                    self,
                    z=z,
                    z_init=z_init,
                    lm_z=lm_z,
                    msa_inputs=_msa_inputs,
                    pair_mask=pair_mask,
                    a=a,
                    b_mat=b_mat,
                    tok_mask=tok_mask,
                    total_steps=total_steps,
                    gather=False,
                    n_orig=_pi_n_orig,
                )
                del z_init, lm_z, _msa_inputs, a, b_mat
                _z_dt = _cp_engine.parcae_finish(self, _z_dt, pair_mask)
                z = None  # pair lives sharded in _z_dt; gathered transiently below
            else:
                # Method call (not inline loop) frees per-iter L²×c_z locals.
                z = self._run_one_loop(
                    z=z,
                    z_init=z_init,
                    lm_z=lm_z,
                    _msa_inputs=_msa_inputs,
                    pair_mask=pair_mask,
                    a=a,
                    b_mat=b_mat,
                    tok_mask=tok_mask,
                    total_steps=total_steps,
                )
                del z_init, lm_z, _msa_inputs, a, b_mat

                z = self.parcae_readout(z)
                z = self.parcae_coda(z, pair_attention_mask=pair_mask)

                z = z.float()

        embeddings: dict[str, Tensor] = {}
        if include_embeddings:
            _z_emb = (
                _z_dt.full_tensor()[:, :_n_orig, :_n_orig, :].float()
                if _z_dt is not None
                else z
            )
            assert _z_emb is not None
            embeddings["output_embedding_pair_pooled"] = _z_emb.detach().mean(
                dim=1, dtype=torch.float32
            )
            del _z_emb

        _cp_disto = getattr(self, "_cp_distogram_head", None)
        if _z_dt is not None and _cp_disto is not None:
            # Distributed: symmetrize + head off the sharded pair, no full-z gather.
            distogram_logits = _cp_disto.forward_sharded(_z_dt, _n_orig)
        else:
            if _z_dt is not None:
                z = _z_dt.full_tensor()[:, :_n_orig, :_n_orig, :].float()
            distogram_logits = self._distogram_logits(z)  # ty:ignore[invalid-argument-type]
            if _z_dt is not None:
                del z  # free the full pair before the (sharded) structure phase

        structure_output = self.structure_head.sample(
            z_trunk=(_z_dt if _z_dt is not None else z),  # ty:ignore[invalid-argument-type]
            s_inputs=x_inputs,
            s_trunk=None,
            relative_position_encoding=relative_position_encoding,
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_mask=atm_mask,
            ref_element=ref_element_oh,
            ref_atom_name_chars=ref_atom_name_chars_oh,
            ref_space_uid=ref_space_uid,
            tok_idx=atom_to_token,
            asym_id=asym_id,
            residue_index=residue_index,
            entity_id=entity_id,
            token_index=token_index,
            sym_id=sym_id,
            token_attention_mask=tok_mask,
            num_diffusion_samples=n_samples,
            num_sampling_steps=num_sampling_steps,
            noise_scale=noise_scale,
            step_scale=step_scale,
            max_inference_sigma=max_inference_sigma,
            return_atom_repr=False,
        )

        sample_coords = structure_output["sample_atom_coords"]
        assert sample_coords is not None
        output: dict[str, Tensor] = {"distogram_logits": distogram_logits}
        output["sample_atom_coords"] = sample_coords

        # Confidence head (#6): on the CP tail, run it distributed straight off the
        # sharded pair (no full-L×L re-gather of z) when the wrapper is installed;
        # otherwise re-gather transiently and run serially. CP-agnostic getattr.
        _cp_conf = getattr(self, "_cp_confidence_head", None)
        if _z_dt is not None and _cp_conf is not None:
            confidence_output = _cp_conf.forward_sharded(
                _z_dt,
                _n_orig,
                s_inputs=x_inputs.detach(),
                x_pred=sample_coords.detach(),
                distogram_atom_idx=disto_idx,
                token_attention_mask=tok_mask,
                atom_to_token=atom_to_token,
                atom_attention_mask=atm_mask,
                asym_id=asym_id,
                mol_type=mol_type,
                num_diffusion_samples=n_samples,
                relative_position_encoding=relative_position_encoding.detach(),
                token_bonds_encoding=token_bonds_encoding.detach(),
            )
        else:
            if _z_dt is not None:
                z = _z_dt.full_tensor()[:, :_n_orig, :_n_orig, :].float()
            confidence_output = self.confidence_head(
                s_inputs=x_inputs.detach(),
                z=z.detach().float(),  # ty:ignore[unresolved-attribute]
                x_pred=sample_coords.detach(),
                distogram_atom_idx=disto_idx,
                token_attention_mask=tok_mask,
                atom_to_token=atom_to_token,
                atom_attention_mask=atm_mask,
                asym_id=asym_id,
                mol_type=mol_type,
                num_diffusion_samples=n_samples,
                relative_position_encoding=relative_position_encoding.detach(),
                token_bonds_encoding=token_bonds_encoding.detach(),
            )
        output.update(confidence_output)
        output["atom_pad_mask"] = (
            atm_mask.unsqueeze(0) if atm_mask.dim() == 1 else atm_mask
        )
        output["residue_index"] = residue_index
        output["entity_id"] = entity_id
        output.update(embeddings)
        return output

    @torch.no_grad()
    def infer_protein(self, seq: str, **forward_kwargs) -> dict:
        from esm.models.esmfold2.protein_utils import (
            OUTPUT_TO_PDB_FEATURE_KEYS,
            prepare_protein_features,
        )

        features = prepare_protein_features(seq)
        features = {k: v.to(self.device) for k, v in features.items()}
        output = self(**features, **forward_kwargs)
        for k in OUTPUT_TO_PDB_FEATURE_KEYS:
            output[k] = features[k]
        return output

    def infer_protein_as_pdb(self, seq: str, **forward_kwargs) -> str:
        return self.output_to_pdb(self.infer_protein(seq, **forward_kwargs))

    @staticmethod
    def output_to_pdb(output: dict) -> str:
        from esm.models.esmfold2.protein_utils import output_to_pdb as _output_to_pdb

        return _output_to_pdb(output)


class MSAEncoderBlock(nn.Module):
    """One MSA encoder block: OPM into pair, MSA pair-weighted averaging, triangle update."""

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_hidden: int,
        n_heads_msa: int,
        msa_head_width: int,
        is_final_block: bool = False,
    ) -> None:
        super().__init__()
        self.is_final_block = is_final_block
        self.outer_product_mean = OuterProductMean(d_msa, d_hidden, d_pair)
        if not is_final_block:
            self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
                d_msa, d_pair, n_heads_msa, msa_head_width
            )
            self.msa_transition = PairTransition(d_msa, expansion_ratio=4)
        self.tri_mul_out = TriangleMultiplicativeUpdate(dim=d_pair, _outgoing=True)
        self.tri_mul_in = TriangleMultiplicativeUpdate(dim=d_pair, _outgoing=False)
        self.pair_transition = PairTransition(d_pair, expansion_ratio=4)

    # Only the triangle updates take a backend: PairTransition chunks but has no
    # kernel to choose.
    def set_kernel_backend(self, backend: str | None) -> None:
        self.tri_mul_out.set_kernel_backend(backend)
        self.tri_mul_in.set_kernel_backend(backend)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.outer_product_mean.set_chunk_size(chunk_size)
        self.tri_mul_out.set_chunk_size(chunk_size)
        self.tri_mul_in.set_chunk_size(chunk_size)
        if not self.is_final_block:
            self.msa_transition.set_chunk_size(chunk_size)
        self.pair_transition.set_chunk_size(chunk_size)

    def _triangle_updates(self, pair: Tensor, pair_attention_mask: Tensor) -> Tensor:
        """Both triangle updates, fused under "fast", eager otherwise.

        The eager engine upcasts the L^2 x c_z contraction to fp32, which makes
        this the same operation as the trunk's at ~9x the cost. The kernel is
        bf16, so this is not bitwise and sits behind "fast", not "fused".
        Cast around the pair of updates the way ``FoldingTrunk`` casts around
        the whole trunk; the kernel's own residual keeps the intermediate bf16.
        """
        if not self.tri_mul_out.fast_residual_available(pair):
            if self.tri_mul_out.fast_backend_requested():
                warn_fast_trimul_refused(
                    "the MSA encoder", fast_trimul_refusal_reason(pair)
                )
            pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
            return pair + self.tri_mul_in(pair, mask=pair_attention_mask)

        orig_dtype = pair.dtype
        fused_pair = _to_fused_kernel_dtype(pair)
        # Bool runs too, but fp32 matches the trunk's mask, so the kernel keeps
        # one Triton specialization instead of two.
        mask = pair_attention_mask.to(torch.float32)
        for tri in (self.tri_mul_out, self.tri_mul_in):
            fused_pair = tri.fused_residual_update(
                fused_pair, residual=fused_pair, drop_mask=None, mask=mask
            )
        return fused_pair.to(orig_dtype)

    def _triangle_updates_rows(
        self, pair: Tensor, pair_attention_mask: Tensor, layout: rowpair.Layout
    ) -> Tensor:
        """Both triangle updates on this rank's rows, in ``_triangle_updates``' dtypes."""
        orig_dtype = pair.dtype
        if self.tri_mul_out.fast_backend_requested() and pair.is_cuda:
            pair = _to_fused_kernel_dtype(pair)
        pair = pair.contiguous()
        mask = pair_attention_mask.to(torch.float32)
        for tri in (self.tri_mul_out, self.tri_mul_in):
            tri.residual_update_rows_(pair, mask, layout)
        return pair.to(orig_dtype)

    def forward(
        self,
        m: Tensor,
        pair: Tensor,
        msa_attention_mask: Tensor,
        pair_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        layout = rowpair.current_layout()
        if layout is not None:
            return self._forward_rows(
                m, pair, msa_attention_mask, pair_attention_mask, layout
            )
        pair = pair + self.outer_product_mean(m, msa_attention_mask)
        if not self.is_final_block:
            m = m + self.msa_pair_weighted_averaging(m, pair, pair_attention_mask)
            m = m + self.msa_transition(m)
        pair = self._triangle_updates(pair, pair_attention_mask)
        pair = pair + self.pair_transition(pair)
        return m, pair

    def _forward_rows(
        self,
        m: Tensor,
        pair: Tensor,
        msa_attention_mask: Tensor,
        pair_attention_mask: Tensor,
        layout: rowpair.Layout,
    ) -> tuple[Tensor, Tensor]:
        """``pair`` and its mask are this rank's rows; ``m`` is replicated."""
        rows = (layout.r0, layout.r1)
        pair = pair + self.outer_product_mean(m, msa_attention_mask, rows=rows)
        if not self.is_final_block:
            m_rows = self.msa_pair_weighted_averaging(
                m, pair, pair_attention_mask, rows=rows
            )
            m = m + rowpair.all_gather_rows(m_rows, layout, dim=1)
            del m_rows
            m = m + self.msa_transition(m)
        pair = self._triangle_updates_rows(pair, pair_attention_mask, layout)
        pair = rowpair.row_blocks_(self.pair_transition, pair, add=True)
        return m, pair


class MSAEncoder(nn.Module):
    """Stack of [`MSAEncoderBlock`] layers that conditions the pair on an MSA."""

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_inputs: int,
        d_hidden: int = 32,
        n_layers: int = 4,
        n_heads_msa: int = 8,
        msa_head_width: int = 16,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(35, d_msa, bias=False)
        self.project_inputs = nn.Linear(d_inputs, d_msa, bias=False)
        self.blocks = nn.ModuleList(
            [
                MSAEncoderBlock(
                    d_msa=d_msa,
                    d_pair=d_pair,
                    d_hidden=d_hidden,
                    n_heads_msa=n_heads_msa,
                    msa_head_width=msa_head_width,
                    is_final_block=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )

    def set_kernel_backend(self, backend: str | None) -> None:
        for block in self.blocks:
            cast(MSAEncoderBlock, block).set_kernel_backend(backend)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        for block in self.blocks:
            cast(MSAEncoderBlock, block).set_chunk_size(chunk_size)

    def forward(
        self,
        x_pair: Tensor,
        x_inputs: Tensor,
        msa_oh: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        msa_attention_mask: Tensor,
    ) -> Tensor:
        # All inputs are pre-transposed to [B, L, M, ...] before calling.
        m_feat = torch.cat(
            [msa_oh, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1
        )
        m = self.embed(m_feat) + self.project_inputs(x_inputs).unsqueeze(2)
        tok_mask = msa_attention_mask[:, :, 0].bool()
        layout = rowpair.current_layout()
        tok_rows = tok_mask if layout is None else rowpair.rows_of(tok_mask, layout)
        pair_attention_mask = tok_rows.unsqueeze(2) & tok_mask.unsqueeze(1)
        for block in self.blocks:
            m, x_pair = block(m, x_pair, msa_attention_mask, pair_attention_mask)
        return x_pair
