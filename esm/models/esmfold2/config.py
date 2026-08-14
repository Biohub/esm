# Copyright 2026 Biohub. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ESMFold2 model configuration."""

from __future__ import annotations

import copy
import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from esm.models.esmc.config import EsmcConfig

CONFIG_NAME = "config.json"

# Published ESMFold2 checkpoints on the HuggingFace Hub.
ESMFOLD2_HF_REPO = "biohub/ESMFold2"
ESMFOLD2_EXPERIMENTAL_HF_REPO = "biohub/ESMFold2-Experimental"

# ---------------------------------------------------------------------------
# Nested dataclass configs
# ---------------------------------------------------------------------------

_DEFAULT_ESMC_HF_REPO = "biohub/ESMC-6B"

# Dotted source path in a pre-alignment ``config.json`` -> dotted path now. Field
# names follow the HuggingFace port so one config.json describes the model to
# both implementations. Delete once every ``biohub/ESMFold2*`` repo has been
# re-published; until then a strict read would reject every live checkpoint.
_LEGACY_PATHS = {
    "d_single": "hidden_size",
    "d_pair": "pairwise_hidden_size",
    "inputs.d_inputs": "single_inputs_size",
    "inputs.atom_encoder.swa_window_size": "sliding_window",
    "inputs.atom_encoder.d_atom": "atom_encoder.hidden_size",
    "inputs.atom_encoder.d_token": "atom_encoder.output_dim",
    "inputs.atom_encoder.n_blocks": "atom_encoder.num_hidden_layers",
    "inputs.atom_encoder.n_heads": "atom_encoder.num_attention_heads",
    "inputs.atom_encoder.expansion_ratio": "atom_encoder.expansion_ratio",
    "inputs.atom_encoder.spatial_rope_base_frequency": (
        "atom_encoder.spatial_rope_base_frequency"
    ),
    "inputs.atom_encoder.n_spatial_rope_pairs_per_axis": (
        "atom_encoder.n_spatial_rope_pairs_per_axis"
    ),
    "inputs.atom_encoder.n_uid_rope_pairs": "atom_encoder.n_uid_rope_pairs",
    "inputs.atom_encoder.uid_rope_base_frequency": (
        "atom_encoder.uid_rope_base_frequency"
    ),
    "folding_trunk.n_layers": "folding_trunk_num_hidden_layers",
    "folding_trunk.n_heads": "folding_trunk_num_attention_heads",
    "folding_trunk.dropout": "folding_trunk_dropout",
    "parcae.coda_n_layers": "parcae_num_coda_layers",
    "msa_encoder_overwrite": "msa_encoder.overwrite",
    "msa_encoder.d_msa": "msa_encoder.hidden_size",
    "msa_encoder.d_hidden": "msa_encoder.outer_hidden_size",
    "msa_encoder.n_layers": "msa_encoder.num_hidden_layers",
    "msa_encoder.n_heads_msa": "msa_encoder.num_attention_heads",
    "msa_encoder.msa_head_width": "msa_encoder.head_width",
    "lm_encoder.n_layers": "lm_encoder.num_hidden_layers",
    "confidence_head.folding_trunk.n_layers": "confidence_head.num_hidden_layers",
    "confidence_head.folding_trunk.n_heads": "confidence_head.num_attention_heads",
    "confidence_head.folding_trunk.dropout": "confidence_head.dropout",
    "structure_head.diffusion_module.c_atom": (
        "structure_head.diffusion_module.atom_encoder.hidden_size"
    ),
    "structure_head.diffusion_module.atom_num_blocks": (
        "structure_head.diffusion_module.atom_encoder.num_hidden_layers"
    ),
    "structure_head.diffusion_module.atom_num_heads": (
        "structure_head.diffusion_module.atom_encoder.num_attention_heads"
    ),
    "structure_head.diffusion_module.c_token": (
        "structure_head.diffusion_module.token_hidden_size"
    ),
}


def _pop_path(tree: dict, path: str):
    """Remove and return ``path`` from a nested dict; ``_MISSING`` when absent."""
    parts = path.split(".")
    node = tree
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        return _MISSING
    return node.pop(parts[-1])


def _set_path(tree: dict, path: str, value) -> None:
    parts = path.split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node.setdefault(parts[-1], value)


_MISSING = object()


def _prune_empty(tree: dict) -> None:
    """Drop dicts left empty by remapping, innermost first.

    The pre-alignment wrappers (``inputs``, ``folding_trunk``,
    ``confidence_head.folding_trunk``) held nothing but remapped leaves, so once
    those move out the wrapper is noise that would otherwise be kept as a stray
    attribute and re-serialised.
    """
    for key in list(tree):
        value = tree[key]
        if not isinstance(value, dict):
            continue
        _prune_empty(value)
        if not value:
            del tree[key]


def _resolve_legacy_keys(kwargs: dict) -> dict:
    """Map a pre-alignment ``config.json`` onto the current field names."""
    if "num_recycles" in kwargs:
        raise ValueError(
            "config.json uses 'num_recycles'; ESMFold2 calls this 'num_loops'. "
            "Re-export the checkpoint's config with the current field name "
            "rather than letting the trunk silently fall back to the default "
            "loop count."
        )
    tree = copy.deepcopy(kwargs)
    moved = []
    for old, new in _LEGACY_PATHS.items():
        value = _pop_path(tree, old)
        if value is _MISSING:
            continue
        _set_path(tree, new, value)
        moved.append(old)
    _prune_empty(tree)
    if moved:
        warnings.warn(
            f"config.json uses {len(moved)} pre-alignment ESMFold2 field "
            f"path(s), e.g. {sorted(moved)[:3]}; see EsmFold2Config for the "
            f"current names.",
            FutureWarning,
            stacklevel=3,
        )
    return tree


@dataclass
class EsmFold2MsaEncoderConfig:
    """Config for the optional MSA encoder module (Large MSA models only)."""

    enabled: bool = False
    # If True, MSA encoder output replaces the pair stream; if False, it is added.
    overwrite: bool = True
    hidden_size: int = 128
    outer_hidden_size: int = 32
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    head_width: int = 32
    # Inference-time MSA diversity: in config, so a checkpoint ships the values
    # it was tuned for rather than a call signature deciding them.
    max_depth: int | None = 1024
    column_mask_rate: float = 0.1


@dataclass
class EsmFold2ParcaeConfig:
    """Release-only config for the parcae diffusion-loop scheduler."""

    enabled: bool = True
    poisson_mean: float = 3.0
    min_steps: int = 1
    max_steps: int | None = 6


@dataclass
class EsmFold2LmEncoderConfig:
    """Release-only config for the LM-side pair encoder."""

    enabled: bool = True
    num_hidden_layers: int = 4
    lm_dropout: float = 0.25
    per_loop_lm_dropout: bool = True


@dataclass
class EsmFold2AtomEncoderConfig:
    """Config for SWA atom encoder/decoder with 3D RoPE.

    The attention window lives on the parent as ``sliding_window``, shared with
    the structure head's atom encoder.
    """

    hidden_size: int = 128
    output_dim: int = 768
    num_hidden_layers: int = 3
    num_attention_heads: int = 4
    expansion_ratio: int = 2
    # 3D RoPE config
    spatial_rope_base_frequency: float = 20.0
    n_spatial_rope_pairs_per_axis: int = 2
    n_uid_rope_pairs: int = 10
    uid_rope_base_frequency: float = 10000.0


@dataclass
class EsmFold2DiffusionAtomEncoderConfig:
    """Widths of the diffusion module's own atom encoder."""

    hidden_size: int = 128
    num_hidden_layers: int = 3
    num_attention_heads: int = 4


@dataclass
class EsmFold2DiffusionModuleConfig:
    """Config for the DiffusionModule."""

    sigma_data: float = 16.0
    token_hidden_size: int = 768
    fourier_dim: int = 256
    token_num_blocks: int = 12
    token_num_heads: int = 16
    transition_multiplier: int = 2
    atom_encoder: EsmFold2DiffusionAtomEncoderConfig = field(
        default_factory=EsmFold2DiffusionAtomEncoderConfig
    )
    # Re-derived by the port from the parent's widths; kept explicit here.
    c_z: int = 256
    c_s_inputs: int = 451
    relpos_r_max: int = 32
    relpos_s_max: int = 2

    def __post_init__(self):
        if isinstance(self.atom_encoder, dict):
            self.atom_encoder = EsmFold2DiffusionAtomEncoderConfig(
                **cast(dict[str, Any], self.atom_encoder)
            )


@dataclass
class EsmFold2StructureHeadConfig:
    """Config for the diffusion-based structure prediction head."""

    diffusion_module: EsmFold2DiffusionModuleConfig = field(
        default_factory=EsmFold2DiffusionModuleConfig
    )
    distogram_bins: int = 128

    # Training noise: sigma ~ sigma_data * exp(mu + sigma * N(0,1))
    train_noise_log_mean: float = -1.2
    train_noise_log_std: float = 1.5

    # Sampling defaults (ODE)
    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.0
    step_scale: float = 1.0

    # Inference schedule defaults
    inference_s_max: float = 160.0
    inference_s_min: float = 4e-4
    inference_p: float = 8.0
    inference_num_steps: int = 68

    def __post_init__(self):
        if isinstance(self.diffusion_module, dict):
            self.diffusion_module = EsmFold2DiffusionModuleConfig(
                **self.diffusion_module  # ty:ignore[invalid-argument-type]
            )


@dataclass
class EsmFold2ConfidenceHeadConfig:
    enabled: bool = True
    num_hidden_layers: int = 4
    num_plddt_bins: int = 50
    num_pde_bins: int = 64
    num_pae_bins: int = 64
    min_dist: float = 2.0
    max_dist: float = 52.0
    distogram_bins: int = 128
    # Not carried by the port, which fixes both; kept so a checkpoint can set them.
    num_attention_heads: int = 8
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class EsmFold2Config:
    """
    Configuration for the ESMFold2 structure prediction model.

    Uses SWA atom encoders with 3D RoPE, a diffusion transformer,
    a folding trunk, and an ESMC 6B PLM backbone.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    the model outputs. Read the documentation from [`PretrainedConfig`] for more
    information.

    Args:
        d_single (`int`, defaults to 384):
            Dimensionality of single (per-residue) representations.
        d_pair (`int`, defaults to 256):
            Dimensionality of pair (residue-residue) representations.
        n_relative_residx_bins (`int`, defaults to 32):
            Number of bins for relative residue index encoding.
        n_relative_chain_bins (`int`, defaults to 2):
            Number of bins for relative chain encoding.
        num_loops (`int`, defaults to 20):
            Number of trunk loops for iterative refinement.
        num_diffusion_samples (`int`, defaults to 8):
            Number of parallel structure predictions to generate.
        lm_dropout (`float`, defaults to 0.0):
            Dropout probability on LM pair embeddings. When > 0, dropout is
            applied with ``training=True`` (including at inference) to match
            the experimental training recipe used by binder design.
        force_lm_dropout_during_inference (`bool`, defaults to False):
            When True, apply ``lm_dropout`` even when ``model.eval()`` and
            ``lm_dropout`` > 0. Binder-design loads set this to True.
        lm_mask_pct (`float`, defaults to 0.0):
            Fraction of sequence residues randomly replaced with the LM mask
            token before running the PLM backbone, matching the training-time
            input corruption. Single-sequence checkpoints set this to 0.1.
        disable_msa_features (`bool`, defaults to False):
            When True, zero out MSA-derived ``profile`` and ``deletion_mean``
            before the inputs embedder (experimental medium/large checkpoints).
        single_inputs_size (`int`, defaults to 451):
            Width of the per-residue input features.
        sliding_window (`int`, defaults to 128):
            Atom-attention window, shared by the inputs embedder and the
            structure head's atom encoder.
        folding_trunk_num_hidden_layers (`int`, defaults to 24):
            Number of pairformer layers in the folding trunk.
        parcae_num_coda_layers (`int`, defaults to 2):
            Number of coda layers in the release-only parcae scheduler.
        atom_encoder (`EsmFold2AtomEncoderConfig`):
            Configuration for the inputs embedder's atom encoder.
        structure_head (`EsmFold2StructureHeadConfig`):
            Configuration for the diffusion-based structure prediction head.
        confidence_head (`EsmFold2ConfidenceHeadConfig`):
            Configuration for the confidence prediction head.

    Examples:

    ```python
    >>> from transformers import EsmFold2Config, EsmFold2ExperimentalModel

    >>> # Initializing an ESMFold2 configuration
    >>> configuration = EsmFold2Config(type="experimental")

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = EsmFold2ExperimentalModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "esmfold2"
    has_no_defaults_at_init = True

    def __init__(self, **kwargs):
        kwargs = _resolve_legacy_keys(kwargs)
        # PretrainedConfig kept unrecognised kwargs as attributes; preserve that
        # so nothing in a published config.json is silently discarded. The
        # explicit assignments below then take precedence.
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Default "release" so a bare ``EsmFold2Config()`` (used internally by
        # HF's ``save_pretrained``) doesn't raise ``KeyError('type')``.
        self.type: str = kwargs.get("type", "release")
        if self.type not in ("release", "experimental"):
            raise ValueError(
                f"EsmFold2Config.type must be 'release' or 'experimental', "
                f"got {self.type!r}"
            )

        # Top-level scalar fields
        self.hidden_size: int = kwargs.get("hidden_size", 384)
        self.pairwise_hidden_size: int = kwargs.get("pairwise_hidden_size", 256)
        self.single_inputs_size: int = kwargs.get("single_inputs_size", 451)
        # Shared by the inputs embedder and the structure head's atom encoder.
        self.sliding_window: int = kwargs.get("sliding_window", 128)
        self.folding_trunk_num_hidden_layers: int = kwargs.get(
            "folding_trunk_num_hidden_layers", 24
        )
        self.parcae_num_coda_layers: int = kwargs.get("parcae_num_coda_layers", 2)
        # Not carried by the port, which fixes both; kept so a checkpoint can set them.
        self.folding_trunk_num_attention_heads: int = kwargs.get(
            "folding_trunk_num_attention_heads", 8
        )
        self.folding_trunk_dropout: float = kwargs.get("folding_trunk_dropout", 0.0)
        self.n_relative_residx_bins: int = kwargs.get("n_relative_residx_bins", 32)
        self.n_relative_chain_bins: int = kwargs.get("n_relative_chain_bins", 2)
        self.num_loops: int = kwargs.get("num_loops", 20)
        self.num_diffusion_samples: int = kwargs.get("num_diffusion_samples", 8)
        # If True, ``profile`` / ``deletion_mean`` are zeroed before the inputs
        # embedder.
        self.disable_msa_features: bool = kwargs.get("disable_msa_features", False)
        self.lm_dropout: float = kwargs.get("lm_dropout", 0.0)
        self.force_lm_dropout_during_inference: bool = kwargs.get(
            "force_lm_dropout_during_inference", False
        )
        self.lm_mask_pct: float = kwargs.get("lm_mask_pct", 0.0)

        self.lm_d_model: int = kwargs.get("lm_d_model", 2560)
        self.lm_num_layers: int = kwargs.get("lm_num_layers", 80)
        # ``esmc_config`` describes a backbone bundled into the same checkpoint,
        # so it loads in one pass. ``esmc_id`` is the older arrangement, where the
        # backbone is a separate repo fetched afterwards by ``load_esmc``.
        esmc_config = kwargs.get("esmc_config")
        if isinstance(esmc_config, dict):
            esmc_config = EsmcConfig.from_dict(esmc_config)
        self.esmc_config: EsmcConfig | None = esmc_config
        self.esmc_id: str = kwargs.get("esmc_id", _DEFAULT_ESMC_HF_REPO)

        def _init_nested(cls, val):
            if isinstance(val, cls):
                return val
            if isinstance(val, dict):
                return cls(**val)
            return cls()

        self.atom_encoder = _init_nested(
            EsmFold2AtomEncoderConfig, kwargs.get("atom_encoder")
        )
        self.structure_head = _init_nested(
            EsmFold2StructureHeadConfig, kwargs.get("structure_head")
        )
        self.confidence_head = _init_nested(
            EsmFold2ConfidenceHeadConfig, kwargs.get("confidence_head")
        )
        self.msa_encoder = _init_nested(
            EsmFold2MsaEncoderConfig, kwargs.get("msa_encoder")
        )
        # Release-only modules — ignored when ``type == "experimental"``.
        self.parcae = _init_nested(EsmFold2ParcaeConfig, kwargs.get("parcae"))
        self.lm_encoder = _init_nested(
            EsmFold2LmEncoderConfig, kwargs.get("lm_encoder")
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs: Any
    ) -> "EsmFold2Config":
        """Read ``config.json`` from a local directory or the HuggingFace Hub."""
        from esm.models.hub import resolve_model_dir

        local_dir = resolve_model_dir(
            pretrained_model_name_or_path,
            revision=kwargs.pop("revision", None),
            cache_dir=kwargs.pop("cache_dir", None),
            token=kwargs.pop("token", None),
            local_files_only=kwargs.pop("local_files_only", False),
            force_download=kwargs.pop("force_download", False),
        )
        with open(Path(local_dir) / CONFIG_NAME) as f:
            raw = json.load(f)
        raw.pop("model_type", None)
        # Caller overrides win, matching PretrainedConfig.from_pretrained.
        raw.update(kwargs)
        return cls(**raw)

    def save_pretrained(self, save_directory: str | os.PathLike) -> None:
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / CONFIG_NAME, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    def to_dict(self):
        output = {k: v for k, v in self.__dict__.items()}
        output["model_type"] = self.model_type
        if self.esmc_config is not None:
            output["esmc_config"] = self.esmc_config.to_dict()
        for name in (
            "atom_encoder",
            "structure_head",
            "confidence_head",
            "msa_encoder",
            "parcae",
            "lm_encoder",
        ):
            output[name] = asdict(getattr(self, name))
        return output


__all__ = [
    "EsmFold2AtomEncoderConfig",
    "EsmFold2Config",
    "EsmFold2ConfidenceHeadConfig",
    "EsmFold2DiffusionAtomEncoderConfig",
    "EsmFold2DiffusionModuleConfig",
    "EsmFold2LmEncoderConfig",
    "EsmFold2MsaEncoderConfig",
    "EsmFold2ParcaeConfig",
    "EsmFold2StructureHeadConfig",
]
