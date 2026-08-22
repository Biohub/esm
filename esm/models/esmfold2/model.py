"""PyTorch ESMFold2 model — the standard released architecture.

Quickstart::

    from transformers import EsmFold2Model

    model = EsmFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
    open("ubq.pdb", "w").write(model.infer_protein_as_pdb("MQIFVKTLTGKT..."))

For multi-chain / ligand / MSA inputs see ``ESMFold2InputBuilder`` in the
companion ``esm`` package.
"""

import math
from contextlib import contextmanager
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format

    TE_AVAILABLE = True
except ImportError:
    te: Any = None
    DelayedScaling: Any = None
    Format: Any = None
    TE_AVAILABLE = False

from esm.models.esmc import EsmcModel
from esm.models.esmc.checkpoint_layout import published_to_native_subtree
from esm.models.esmfold2.config import EsmFold2Config, default_module_flags
from esm.models.esmfold2.hf_checkpoint import hf_state_dict_to_native, is_hf_layout
from esm.models.esmfold2.layers import (
    CHAR_VOCAB_SIZE,
    MAX_ATOMIC_NUMBER,
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
    compute_lm_hidden_states,
    gather_rep_atom_coords,
    gather_token_to_atom,
    maybe_apply_msa_column_masking,
    maybe_subsample_msa,
)
from esm.models.hub import HubPreTrainedModel, resolve_model_dir

_EPS = 1e-6
_NONPOLYMER_ID = 4

# Default for the triangle / OPM / pair-transition L² ops. Caps peak memory
# so L≈2k folds on an 80 GB GPU (~76 GB peak at chunk=128 for L=1438;
# chunk=64 leaves headroom for the largest foldbench targets). Override via
# ``model.set_chunk_size(...)``; pass None to disable chunking (faster for
# short L but OOM-prone past ~600).
_DEFAULT_CHUNK_SIZE = 64

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

    def set_kernel_backend(self, backend: str | None) -> None:
        self.folding_trunk.set_kernel_backend(backend)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)

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

        z_base = self.z_norm(z)
        if relative_position_encoding is not None:
            z_base = z_base + relative_position_encoding
        if token_bonds_encoding is not None:
            z_base = z_base + token_bonds_encoding
        z_base = z_base + self.s_to_z(s_inputs_normed).unsqueeze(2)
        z_base = z_base + self.s_to_z_transpose(s_inputs_normed).unsqueeze(1)
        z_base = z_base + self.s_to_z_prod_out(
            self.s_to_z_prod_in1(s_inputs_normed)[:, :, None, :]
            * self.s_to_z_prod_in2(s_inputs_normed)[:, None, :, :]
        )

        pair = self._repeat_batch(z_base, num_diffusion_samples)
        x_pred_flat = self._flatten_sample_axis(x_pred)
        atom_to_token_m = self._repeat_batch(atom_to_token, num_diffusion_samples)
        atom_mask_m = self._repeat_batch(atom_attention_mask, num_diffusion_samples)
        rep_idx_m = self._repeat_batch(distogram_atom_idx, num_diffusion_samples).long()
        mask = self._repeat_batch(token_attention_mask, num_diffusion_samples)
        Bm = pair.shape[0]

        rep_coords = gather_rep_atom_coords(x_pred_flat, rep_idx_m)
        rep_distances = torch.cdist(
            rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist"
        )
        distogram_bins = (
            (rep_distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        )
        pair = pair + self.dist_bin_pairwise_embed(distogram_bins)

        pair_mask = mask[:, :, None].float() * mask[:, None, :].float()

        # FoldingTrunk handles the bf16 cast internally during inference so
        # each block's fused trimul engages. In-place residual avoids an
        # extra fp32 pair allocation.
        with torch.amp.autocast("cuda", enabled=pair.is_cuda, dtype=torch.bfloat16):
            pair_delta = self.folding_trunk(pair, pair_attention_mask=pair_mask)
        pair.add_(pair_delta.float())
        del pair_delta
        single = self.row_attention_pooling(pair, mask)

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
        is_ligand = (expanded_type == _NONPOLYMER_ID).float()
        inter_chain = (
            expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)
        ).float()
        near_contact = (rep_distances < 8).float()
        interface_per_token = (
            near_contact * inter_chain * (1.0 - is_ligand).unsqueeze(-1)
        ).amax(dim=-1)
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

        # PAE
        pae_logits = self.pae_head(self.pae_ln(pair))
        pae = _categorical_mean(pae_logits, start=0.0, end=32.0).detach()

        # PDE
        pde_logits = self.pde_head(self.pde_ln(pair))
        pde = _categorical_mean(pde_logits, start=0.0, end=32.0).detach()

        # Resolved (per-atom binary).
        s_at_atoms_res = self.resolved_ln(s_at_atoms)
        w_res = self.resolved_weight[intra_idx]
        resolved_logits = torch.einsum("...c,...cb->...b", s_at_atoms_res, w_res)

        # pTM / ipTM from pae_logits.
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

        pair_mask_2d = mask_f.unsqueeze(-1) * mask_f.unsqueeze(-2)
        ptm_per_row = (tm_expected * pair_mask_2d).sum(dim=-1) / (
            pair_mask_2d.sum(dim=-1) + _EPS
        )
        ptm = ptm_per_row.max(dim=-1).values

        inter_chain_mask = (
            expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)
        ).float() * pair_mask_2d
        iptm_per_row = (tm_expected * inter_chain_mask).sum(dim=-1) / (
            inter_chain_mask.sum(dim=-1) + _EPS
        )
        iptm = iptm_per_row.max(dim=-1).values

        max_chain_id = int(expanded_asym.max().item()) if Bm > 0 else 0
        n_chains = max_chain_id + 1
        pair_chains_iptm = torch.zeros(
            Bm, n_chains, n_chains, device=tm_expected.device, dtype=tm_expected.dtype
        )
        # pair_chains_iptm[c1, c2] = max over rows i in chain c2 of the mean over
        # columns j in chain c1 of tm_expected[i, j] (max-of-row-mean, as in the
        # global iptm above), so iptm equals the max off-diagonal entry.
        for c1 in range(n_chains):
            chain_c1 = (expanded_asym == c1).float() * mask_f
            if chain_c1.sum() == 0:
                continue
            col_mask = chain_c1.unsqueeze(-2)
            avg_tm = (tm_expected * col_mask).sum(dim=-1) / (
                col_mask.sum(dim=-1) + _EPS
            )
            for c2 in range(n_chains):
                chain_c2 = (expanded_asym == c2).float() * mask_f
                row_vals = avg_tm.masked_fill(chain_c2 == 0, float("-inf"))
                pair_chains_iptm[:, c1, c2] = row_vals.max(dim=-1).values.clamp(min=0.0)

        return {
            "plddt_logits": plddt_logits,
            "plddt": plddt.detach(),
            "plddt_per_atom": plddt_per_atom.detach(),
            "plddt_ca": plddt_ca.detach(),
            "complex_plddt": complex_plddt.detach(),
            "complex_iplddt": complex_iplddt.detach(),
            "pae_logits": pae_logits,
            "pae": pae,
            "pde_logits": pde_logits,
            "pde": pde,
            "resolved_logits": resolved_logits,
            "ptm": ptm.detach(),
            "iptm": iptm.detach(),
            "pair_chains_iptm": pair_chains_iptm.detach(),
        }


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


def _convert_te_modules_to_fp8_inplace(module: nn.Module) -> None:
    """Re-init each TE module via quantized_model_init so weights live as fp8.

    Must be called inside torch.no_grad(); covers nn.Linear, te.Linear,
    te.LayerNormLinear, te.LayerNormMLP — the last two hold 99% of ESMC weight.
    """
    if not TE_AVAILABLE:
        raise RuntimeError("transformer_engine is not available; cannot use fp8.")
    from transformer_engine.pytorch import quantized_model_init

    def _walk(mod: nn.Module) -> None:
        for name, child in list(mod.named_children()):
            replaced = False
            if isinstance(child, nn.Linear):
                in_f, out_f = child.in_features, child.out_features
                has_bias = child.bias is not None
                device = child.weight.device
                dtype = child.weight.dtype
                w = child.weight.data
                b = child.bias.data if has_bias else None
                setattr(mod, name, nn.Identity())
                del child
                torch.cuda.empty_cache()
                with quantized_model_init(enabled=True):
                    new_mod = te.Linear(
                        in_f, out_f, bias=has_bias, params_dtype=dtype
                    ).to(device)
                new_mod.weight.quantize_(w)  # ty:ignore[call-non-callable, unresolved-attribute]
                if has_bias:
                    assert b is not None
                    new_mod.bias.data.copy_(b)  # ty:ignore[call-non-callable]
                del w, b
                replaced = True
            elif isinstance(child, te.Linear):
                # te.Linear with bf16 weight → re-init inside quantized_model_init for fp8.
                in_f, out_f = child.in_features, child.out_features
                has_bias = child.bias is not None
                device = child.weight.device
                dtype = (
                    child.weight.dtype
                    if not hasattr(child.weight, "_data")
                    else torch.bfloat16
                )
                state = {k: v.detach().clone() for k, v in child.state_dict().items()}
                setattr(mod, name, nn.Identity())
                del child
                torch.cuda.empty_cache()
                with quantized_model_init(enabled=True):
                    new_mod = te.Linear(
                        in_f, out_f, bias=has_bias, params_dtype=dtype
                    ).to(device)  # ty:ignore[no-matching-overload]
                new_mod.load_state_dict(state, strict=False)
                replaced = True
            elif hasattr(te, "LayerNormLinear") and isinstance(
                child, te.LayerNormLinear
            ):
                state = {k: v.detach().clone() for k, v in child.state_dict().items()}
                hidden_size = child.in_features
                out_features = child.out_features
                has_bias = child.use_bias
                device = next(child.parameters()).device
                setattr(mod, name, nn.Identity())
                del child
                torch.cuda.empty_cache()
                with quantized_model_init(enabled=True):
                    new_mod = te.LayerNormLinear(
                        hidden_size,
                        out_features,
                        bias=has_bias,
                        params_dtype=torch.bfloat16,
                    ).to(device)
                new_mod.load_state_dict(state, strict=False)
                replaced = True
            elif hasattr(te, "LayerNormMLP") and isinstance(child, te.LayerNormMLP):
                state = {k: v.detach().clone() for k, v in child.state_dict().items()}
                fc1_weight: Tensor = child.fc1_weight
                hidden_size = int(fc1_weight.shape[1])
                # fc1 packed as (2*ffn_hidden_size, hidden_size) for swiglu.
                ffn_hidden_size = int(fc1_weight.shape[0]) // 2
                has_bias = (
                    getattr(child, "fc1_bias", None) is not None
                    and child.fc1_bias is not None
                )
                device = fc1_weight.device
                setattr(mod, name, nn.Identity())
                del child
                torch.cuda.empty_cache()
                with quantized_model_init(enabled=True):
                    new_mod = te.LayerNormMLP(
                        hidden_size=hidden_size,
                        ffn_hidden_size=ffn_hidden_size,
                        bias=has_bias,
                        activation="swiglu",
                        params_dtype=torch.bfloat16,
                    ).to(device)
                new_mod.load_state_dict(state, strict=False)
                replaced = True

            if replaced:
                # Freeze via .eval()+.requires_grad_(False); per-param ops would unwrap Float8Tensor.
                new_mod.eval().requires_grad_(False)
                setattr(mod, name, new_mod)
                torch.cuda.empty_cache()
            else:
                _walk(child)

    _walk(module)
    torch.cuda.empty_cache()


@contextmanager
def _lm_precision_context(fp8: bool):
    """bf16 autocast (+ optional TE fp8 autocast) around the LM forward.

    te.autocast keeps te.Linear outputs bf16 instead of the fp32 default
    (~425 MB at L=1024 in the hidden-state cache).
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if fp8 and TE_AVAILABLE:
            fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID,
                amax_history_len=1,
                amax_compute_algo="most_recent",
            )
            with te.autocast(enabled=True, recipe=fp8_recipe):
                yield
        else:
            yield


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
    * ``model.set_kernel_backend(None | "fused" | "cuequivariance")``:
      select kernel backend (None = reference path).
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
        self._esmc_fp8: bool = False  # set by load_esmc(fp8=True)

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

    def load_esmc(self, esmc_model_path: str, precision: str = "bf16") -> None:
        """Fetch the ESMC LM from a separate repo and attach it.

        Only needed when the backbone is *not* bundled into this checkpoint; see
        ``EsmFold2Config.esmc_config``.
        """
        from esm.models.esmc import EsmcModel

        self.esmc = EsmcModel.from_pretrained(esmc_model_path)
        self.set_esmc_precision(precision)

    def set_esmc_precision(self, precision: str = "bf16") -> None:
        """Cast the attached ESMC LM and freeze it.

        ``precision``: ``"bf16"`` (default), ``"fp32"``, or ``"fp8"``.
        ``"fp8"`` requires H100 + TransformerEngine ≥ 2.x and quantizes
        every TE module's weights to fp8 storage.
        """
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp8": torch.bfloat16,  # underlying weights stay bf16, TE re-quantizes to fp8
        }
        if precision not in dtype_map:
            raise ValueError(
                f"precision must be one of {list(dtype_map)}, got {precision!r}"
            )
        if self.esmc is None:
            raise RuntimeError("no ESMC LM is attached; nothing to cast.")

        esmc = self.esmc.to(device=self.device, dtype=dtype_map[precision]).eval()
        for p in esmc.parameters():
            p.requires_grad_(False)

        if precision == "fp8":
            if not TE_AVAILABLE:
                raise RuntimeError(
                    "transformer_engine is not available; cannot use fp8."
                )
            with torch.no_grad():
                _convert_te_modules_to_fp8_inplace(esmc)
            self._esmc_fp8 = True
        else:
            self._esmc_fp8 = False

        self.esmc = esmc

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        load_esmc: bool = True,
        esmc_precision: str = "bf16",
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
        model = cls._load_pretrained(local_dir, config, device=device, dtype=dtype)
        # A bundled backbone came in with the trunk; only the separate-repo
        # arrangement needs a second fetch.
        if load_esmc and config.esmc_config is None:
            model.load_esmc(model.config.esmc_id, precision=esmc_precision)
        elif config.esmc_config is not None:
            model.set_esmc_precision(esmc_precision)
        return model

    def set_kernel_backend(self, backend: str | None) -> None:
        """Select kernel backend.

        Args:
            backend: ``None`` (reference path), ``"fused"`` (vendored Triton
                kernels), or ``"cuequivariance"`` (cuequivariance kernels
                where applicable; vanilla python fallback otherwise).
        """
        self.folding_trunk.set_kernel_backend(backend)
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
        before compiling.
        """
        import torch._dynamo

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
        # fp8 TE kernels require prod(shape[:-1]) % 8 == 0.
        pad_to = 8 if self._esmc_fp8 else None
        with _lm_precision_context(self._esmc_fp8):
            return compute_lm_hidden_states(
                self.esmc,
                input_ids,
                asym_id,
                residue_index,
                mol_type,
                tok_mask,
                pad_to_multiple=pad_to,
                lm_mask_pct=lm_mask_pct,
            )

    def _discretized_dynamics(self) -> tuple[Tensor, Tensor]:
        delta = F.softplus(self.parcae_log_delta)
        a = torch.exp(-delta * torch.exp(self.parcae_log_a))
        b = delta[:, None] * self.parcae_b_cont
        return a, b

    def _init_pair_state(self, ref: Tensor) -> Tensor:
        std = math.sqrt(2.0 / (5.0 * ref.shape[-1]))
        state = torch.empty_like(ref, dtype=torch.float32)
        nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return state.to(dtype=ref.dtype)

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
                refined_lm_z = self.lm_encoder(
                    lm_z_i.to(z_init.dtype), pair_attention_mask=pair_mask
                )

            z_inject_pair = z_init
            if lm_z_i is not None and self.lm_encoder is None:
                z_inject_pair = z_inject_pair + lm_z_i.to(z_inject_pair.dtype)

            if self.msa_encoder is not None and _msa_inputs is not None:
                # Fresh row subsample each iteration (column mask was applied
                # once in forward, before this loop).
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

            injected_pair = self.parcae_input_norm(z_inject_pair)
            z = a * z + F.linear(injected_pair.to(z.dtype), b_mat)
            z = self.folding_trunk(z, pair_attention_mask=pair_mask)

        return z

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

            if (
                lm_hidden_states is None
                and input_ids is not None
                and self.esmc is not None
            ):
                lm_hidden_states = self._compute_lm_hidden_states(
                    input_ids,
                    asym_id,
                    residue_index,
                    mol_type,
                    tok_mask,
                    lm_mask_pct=(
                        lm_mask_pct
                        if lm_mask_pct is not None
                        else self.config.lm_mask_pct
                    ),
                )
            lm_z: Tensor | None = None
            if lm_hidden_states is not None:
                lm_z = self.language_model(lm_hidden_states.detach())
            del lm_hidden_states

            pair_mask = tok_mask[:, :, None].float() * tok_mask[:, None, :].float()

            z = self._init_pair_state(z_init)

            a, b = self._discretized_dynamics()
            a = a.view(1, 1, 1, -1).to(device=z.device, dtype=z.dtype)
            b_mat = b.to(device=z.device, dtype=z.dtype)

            # Inference-time MSA diversity: column mask is applied once here
            # (shared across recycling loops); row subsampling is deferred to
            # per-iter inside _run_one_loop (fresh subset per loop).
            _msa_inputs: dict | None = None
            if self.msa_encoder is not None and msa is not None:
                # ``None`` means "use the checkpoint's value".
                msa_cfg = self.config.msa_encoder
                depth = msa_cfg.max_depth if msa_max_depth is None else msa_max_depth
                rate = (
                    msa_cfg.column_mask_rate
                    if msa_column_mask_rate is None
                    else msa_column_mask_rate
                )
                msa_attention_mask = maybe_apply_msa_column_masking(
                    msa_attention_mask, rate=rate
                )
                _msa_inputs = dict(
                    msa=msa,
                    msa_attention_mask=msa_attention_mask,
                    has_deletion=has_deletion,
                    deletion_value=deletion_value,
                    x_inputs=x_inputs,
                    max_depth=depth,
                    subsample_enabled=depth is not None,
                )

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
        distogram_logits = self.distogram_head(z + z.transpose(-2, -3))

        structure_output = self.structure_head.sample(
            z_trunk=z,
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

        confidence_output = self.confidence_head(
            s_inputs=x_inputs.detach(),
            z=z.detach().float(),
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

    def forward(
        self,
        m: Tensor,
        pair: Tensor,
        msa_attention_mask: Tensor,
        pair_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pair = pair + self.outer_product_mean(m, msa_attention_mask)
        if not self.is_final_block:
            m = m + self.msa_pair_weighted_averaging(m, pair, pair_attention_mask)
            m = m + self.msa_transition(m)
        pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, mask=pair_attention_mask)
        pair = pair + self.pair_transition(pair)
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
        pair_attention_mask = tok_mask.unsqueeze(2) & tok_mask.unsqueeze(1)
        for block in self.blocks:
            m, x_pair = block(m, x_pair, msa_attention_mask, pair_attention_mask)
        return x_pair
