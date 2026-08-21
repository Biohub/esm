"""Present this package's ESMFold2 API on top of the upstream HuggingFace port.

:class:`EsmFold2HFAdapter` wraps ``transformers.models.esmfold2.EsmFold2Model`` so
that ``ESMFold2InputBuilder``, ``MolecularComplexResult``, ``output_to_pdb`` and the
``infer_protein`` family keep working against it.

The direction is ours-on-theirs: their input side is single-chain protein only,
while ours has CCD ligands, MSA and multi-chain, so wrapping ours around theirs
keeps the richer half. Their trunk/head split is re-exposed as
:meth:`EsmFold2HFAdapter.trunk`.

The port landed in ``transformers`` 5.16.0.dev0 and this repo pins 4.x, so nothing
upstream is imported at module scope; :func:`upstream_available` gates the two
entry points that need it.
"""

import contextlib
import importlib.util
from typing import Any

import torch
from torch import Tensor

from esm.models.esmfold2.layers import (
    maybe_apply_msa_column_masking,
    maybe_subsample_msa,
)
from esm.models.esmfold2.model import _IGNORED_FEATURE_KEYS

#: Import path of the upstream port. Absent from ``transformers`` < 5.16.
_UPSTREAM_MODULE = "transformers.models.esmfold2.modeling_esmfold2"

#: An older Biohub/transformers fork can occupy the same import path without these.
_PORT_MARKERS = ("EsmFold2AtomInputs", "EsmFold2FoldingMixin")

_UPSTREAM_MISSING_MESSAGE = (
    f"{_UPSTREAM_MODULE} does not expose the ESMFold2 port, which landed in "
    "transformers 5.16.0.dev0. Install upstream transformers to use "
    "EsmFold2HFAdapter.from_pretrained / .trunk."
)

#: Featurizer keys no model consumes; their ``fold`` would swallow them silently.
_FEATURIZER_ONLY_KEYS = _IGNORED_FEATURE_KEYS

#: The only feature-key rename between the two sides; the other 22 are identical.
_FEATURE_RENAMES = {"token_attention_mask": "attention_mask"}

#: Flat in a ``fold`` call; their ``forward`` takes them bundled as ``EsmFold2AtomInputs``.
_ATOM_INPUT_KEYS = (
    "ref_pos",
    "ref_charge",
    "atom_attention_mask",
    "ref_element",
    "ref_atom_name_chars",
    "ref_space_uid",
    "atom_to_token",
)


def _import_upstream():
    """The upstream modeling module, or ``None`` if this env does not have it."""
    try:
        if importlib.util.find_spec(_UPSTREAM_MODULE) is None:
            return None
        module = importlib.import_module(_UPSTREAM_MODULE)
    except (ImportError, ValueError):
        return None
    if not all(hasattr(module, marker) for marker in _PORT_MARKERS):
        return None
    return module


def upstream_available() -> bool:
    """Whether the upstream ESMFold2 port is usable in this environment."""
    return _import_upstream() is not None


def _upstream():
    """The upstream modeling module, or an ImportError explaining its absence."""
    module = _import_upstream()
    if module is None:
        raise ImportError(_UPSTREAM_MISSING_MESSAGE)
    return module


def translate_features(features: dict[str, Any]) -> dict[str, Any]:
    """Our feature dict -> the kwargs upstream ``EsmFold2Model.fold`` takes."""
    out: dict[str, Any] = {}
    for key, value in features.items():
        if key in _FEATURIZER_ONLY_KEYS:
            continue
        if key in ("disto_cond", "disto_cond_mask"):
            # The port has no conditioning path, so drop-after-checking matches ours.
            if key == "disto_cond_mask" and value is not None and value.any():
                raise NotImplementedError(
                    "distogram conditioning is not implemented for ESMFold2"
                )
            continue
        out[_FEATURE_RENAMES.get(key, key)] = value
    return out


@contextlib.contextmanager
def _sampler_config_overrides(
    config: Any,
    *,
    noise_scale: float | None,
    step_scale: float | None,
    max_inference_sigma: float | None,
):
    """Scope our per-call sampler kwargs onto ``config.structure_head``.

    Upstream takes no sampler arguments and reads these off the config at call
    time. Mutating shared state is **not** thread-safe, unlike passing an argument.
    """
    head = config.structure_head
    saved = (head.noise_scale, head.step_scale, head.inference_sigma_cap)
    if noise_scale is not None:
        head.noise_scale = noise_scale
    if step_scale is not None:
        head.step_scale = step_scale
    if max_inference_sigma is not None:
        head.inference_sigma_cap = max_inference_sigma
    try:
        yield
    finally:
        head.noise_scale, head.step_scale, head.inference_sigma_cap = saved


class EsmFold2HFAdapter(torch.nn.Module):
    """Our :class:`~...esmfold2.model.EsmFold2Model` surface over the HF port.

    Construct from an already-built upstream model, or from a converted checkpoint
    via :meth:`from_pretrained`::

        result = ESMFold2InputBuilder().fold(EsmFold2HFAdapter(hf_model), spi)
    """

    def __init__(self, hf_model: Any) -> None:
        super().__init__()
        self.model = hf_model

    # -- attribute passthrough ------------------------------------------------

    @property
    def config(self) -> Any:
        """*Their* config: ``num_diffusion_samples``, ``msa_encoder.max_depth``,
        ``lm_mask_pct``, ``type`` and ``disable_msa_features`` do not exist there."""
        return self.model.config

    @property
    def device(self) -> torch.device:
        return self.model.device

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        load_esmc: bool = True,
        esmc_precision: str = "bf16",
        config: Any = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        **kwargs,
    ) -> "EsmFold2HFAdapter":
        """Our ``from_pretrained`` signature, backed by theirs.

        The checkpoint must already be in the upstream layout; ``biohub/ESMFold2``
        needs their offline converter run over it first.
        """
        if not load_esmc:
            raise NotImplementedError(
                "the HF port always bundles ESMC under esmc.*; load_esmc=False "
                "has no counterpart"
            )
        upstream = _upstream()
        if config is not None:
            kwargs["config"] = config
        model = upstream.EsmFold2Model.from_pretrained(
            pretrained_model_name_or_path, dtype=dtype, **kwargs
        )
        # No "fp8": ours quantizes TransformerEngine modules in place, and the
        # port's ESMC is plain PyTorch with none for that walk to find.
        esmc_dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
        if esmc_precision not in esmc_dtypes:
            raise ValueError(
                f"esmc_precision must be one of {list(esmc_dtypes)}, "
                f"got {esmc_precision!r}"
            )
        model.esmc.requires_grad_(False)
        model.esmc = model.esmc.to(dtype=esmc_dtypes[esmc_precision])
        return cls(model.to(device).eval())

    def set_chunk_size(self, chunk_size: int | None) -> None:
        """Set the token-axis chunk on every module that holds one.

        Upstream reads ``config.chunk_size`` once, in ``__init__``, so setting the
        config after load does nothing and the tree has to be walked instead.
        """
        self.model.config.chunk_size = chunk_size
        for module in self.model.modules():
            if hasattr(module, "chunk_size"):
                module.chunk_size = chunk_size

    # -- the call -------------------------------------------------------------

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
        """Our end-to-end ``forward``, dispatched to their ``fold``.

        Their ``forward`` is the trunk alone: same name, different meaning.
        Signature and output dict match ours key for key.
        """
        # Their fold swallows unknown keys silently; ours raises. Keep ours.
        unexpected = sorted(set(unused_features) - _FEATURIZER_ONLY_KEYS)
        if unexpected:
            raise TypeError(
                f"{type(self).__name__}.forward() got unexpected keyword "
                f"argument(s) {unexpected}."
            )

        if disto_cond_mask is not None and disto_cond_mask.any():
            raise NotImplementedError(
                "distogram conditioning is not implemented for ESMFold2; "
                "fold without it."
            )

        if lm_mask_pct:
            input_ids = self._mask_input_ids(input_ids, lm_mask_pct)

        if msa is not None:
            # Row subsampling is per-fold here, not per trunk loop as ours is: the
            # loop is inside their forward. Prefer msa_max_depth=None on the port.
            if msa_column_mask_rate:
                msa_attention_mask = maybe_apply_msa_column_masking(
                    msa_attention_mask, rate=msa_column_mask_rate
                )
            if msa_max_depth is not None:
                msa, msa_attention_mask, has_deletion, deletion_value = (
                    maybe_subsample_msa(
                        msa,
                        msa_attention_mask,
                        has_deletion,
                        deletion_value,
                        max_depth=msa_max_depth,
                        enabled=True,
                    )
                )

        features = translate_features(
            {
                "token_index": token_index,
                "residue_index": residue_index,
                "asym_id": asym_id,
                "sym_id": sym_id,
                "entity_id": entity_id,
                "mol_type": mol_type,
                "res_type": res_type,
                "token_bonds": token_bonds,
                "token_attention_mask": token_attention_mask,
                "ref_pos": ref_pos,
                "ref_element": ref_element,
                "ref_charge": ref_charge,
                "ref_atom_name_chars": ref_atom_name_chars,
                "ref_space_uid": ref_space_uid,
                "atom_attention_mask": atom_attention_mask,
                "atom_to_token": atom_to_token,
                "distogram_atom_idx": distogram_atom_idx,
                "deletion_mean": deletion_mean,
                "msa": msa,
                "has_deletion": has_deletion,
                "deletion_value": deletion_value,
                "msa_attention_mask": msa_attention_mask,
                "input_ids": input_ids,
                "lm_hidden_states": lm_hidden_states,
            }
        )
        # Their defaults are keyword-absent, not None.
        features = {k: v for k, v in features.items() if v is not None}

        with _sampler_config_overrides(
            self.config,
            noise_scale=noise_scale,
            step_scale=step_scale,
            max_inference_sigma=max_inference_sigma,
        ):
            hf_output = self.model.fold(
                num_diffusion_samples=num_diffusion_samples,
                num_sampling_steps=num_sampling_steps,
                num_loops=num_loops,
                **features,
            )

        # Absent from their EsmFold2Output, and read by ESMFold2InputBuilder.decode.
        output = {k: v for k, v in hf_output.items() if v is not None}
        output["atom_pad_mask"] = (
            atom_attention_mask.unsqueeze(0)
            if atom_attention_mask.dim() == 1
            else atom_attention_mask
        )
        output["residue_index"] = residue_index
        output["entity_id"] = entity_id
        return output

    def _mask_input_ids(self, input_ids: Tensor | None, lm_mask_pct: float) -> Tensor:
        """Ours masks after the BOS/EOS wrap; with no upstream hook this masks
        before it -- same rate, different positions."""
        if input_ids is None:
            raise ValueError("lm_mask_pct requires input_ids")
        esmc_config = self.config.esmc_config
        special = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_id in (
            esmc_config.bos_token_id,
            esmc_config.eos_token_id,
            esmc_config.pad_token_id,
        ):
            special |= input_ids == special_id
        do_mask = (
            torch.rand(input_ids.shape, device=input_ids.device) < lm_mask_pct
        ) & ~special
        return input_ids.masked_fill(do_mask, esmc_config.mask_token_id)

    # -- their extra surface, re-exposed --------------------------------------

    def trunk(self, **features: Any) -> Any:
        """Their ``forward``: the trunk alone. We have no counterpart."""
        upstream = _upstream()
        features = translate_features(features)
        atom_inputs = upstream.EsmFold2AtomInputs(
            **{key: features.pop(key) for key in _ATOM_INPUT_KEYS}
        )
        # Confidence-head only; dropped so their trunk cannot swallow it.
        features.pop("distogram_atom_idx", None)
        return self.model(atom_inputs=atom_inputs, **features)

    # -- convenience entry points ---------------------------------------------

    @torch.no_grad()
    def infer_protein(self, seq: str, **forward_kwargs: Any) -> dict:
        """Fold a single-chain sequence, with the ``output_to_pdb`` keys attached.

        Uses our featurizer, so the returned dict keeps our key spellings.
        """
        from esm.models.esmfold2.protein_utils import (
            OUTPUT_TO_PDB_FEATURE_KEYS,
            prepare_protein_features,
        )

        features = {
            k: v.to(self.device) for k, v in prepare_protein_features(seq).items()
        }
        output = self(**features, **forward_kwargs)
        for key in OUTPUT_TO_PDB_FEATURE_KEYS:
            output[key] = features[key]
        return output

    def infer_protein_as_pdb(self, seq: str, **forward_kwargs: Any) -> str:
        return self.output_to_pdb(self.infer_protein(seq, **forward_kwargs))

    @staticmethod
    def output_to_pdb(output: dict) -> str:
        """Ours, which renders sample 0; theirs defaults to the best-ranked one."""
        from esm.models.esmfold2.protein_utils import output_to_pdb as _output_to_pdb

        return _output_to_pdb(output)


__all__ = ["EsmFold2HFAdapter", "translate_features", "upstream_available"]
