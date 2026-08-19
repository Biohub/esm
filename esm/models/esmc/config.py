"""Configuration for the ESMC models."""

import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_NAME = "config.json"

# Published ESMC checkpoints on the HuggingFace Hub.
ESMC_300M_HF_REPO = "biohub/ESMC-300M"
ESMC_600M_HF_REPO = "biohub/ESMC-600M"
ESMC_6B_HF_REPO = "biohub/ESMC-6B"

# Every published ESMC uses this SwiGLU expansion. It stays a constant rather
# than a config field because ``intermediate_size`` is what actually fixes the
# shapes; this only derives a default for checkpoints that omit it.
ESMC_EXPANSION_RATIO = 8 / 3

# Field names as spelled by checkpoints published before the field names were
# aligned with ``transformers``. Delete once every ``biohub/ESMC-*`` repo has
# been re-published; until then a strict read would reject every live checkpoint.
_LEGACY_KEYS = {
    "d_model": "hidden_size",
    "n_heads": "num_attention_heads",
    "n_layers": "num_hidden_layers",
}


def esmc_intermediate_size(hidden_size: int) -> int:
    """Round the SwiGLU hidden dim up to a multiple of 256 after expansion."""
    return int(((ESMC_EXPANSION_RATIO * hidden_size) + 255) // 256 * 256)


def _resolve_legacy_keys(raw: dict) -> dict:
    """Map pre-alignment field names onto their current spelling."""
    stale = {old: new for old, new in _LEGACY_KEYS.items() if old in raw}
    if not stale:
        return raw
    resolved = dict(raw)
    for old, new in stale.items():
        resolved.setdefault(new, resolved.pop(old))
    warnings.warn(
        f"{CONFIG_NAME} uses pre-alignment ESMC field names {sorted(stale)}; "
        f"the current names are {sorted(stale.values())}.",
        FutureWarning,
        stacklevel=3,
    )
    return resolved


@dataclass
class EsmcConfig:
    """Configuration for the ESMC models.

    Field names match ``transformers``' ``EsmcConfig`` so that one ``config.json``
    describes the model to both implementations.

    Parameters
    ----------
    vocab_size : int
        Number of amino acid tokens representable by ``input_ids``.
    hidden_size : int
        Dimensionality of the encoder layers.
    num_attention_heads : int
        Number of attention heads per layer.
    num_hidden_layers : int
        Number of transformer blocks.
    intermediate_size : int | None
        Width of the SwiGLU feed-forward hidden layer. Derived from
        ``hidden_size`` via :func:`esmc_intermediate_size` when ``None``.
    pad_token_id : int
        Index of the padding token (``"<pad>"``).
    mask_token_id : int
        Index of the mask token (``"<mask>"``).
    initializer_range : float
        Standard deviation of the normal weight initialiser.
    classifier_dropout : float
        Dropout ratio for the classification heads.
    num_labels : int
        Number of labels for the classification heads.
    problem_type : str | None
        One of ``"regression"``, ``"single_label_classification"`` or
        ``"multi_label_classification"``; inferred at loss time when ``None``.
    output_hidden_states : bool
        Default value for ``output_hidden_states`` at forward time.
    output_attentions : bool
        Default value for ``output_attentions`` at forward time.
    return_dict : bool
        Default value for ``return_dict`` at forward time.
    attn_implementation : str
        Attention backend: ``"sdpa"``, ``"eager"`` or ``"flash_attention_2"``.
        A property of the run rather than of the weights, so it is honoured when
        a ``config.json`` carries it but never written back — matching
        ``transformers``, which keeps it off the serialised config entirely.
    """

    model_type = "esmc"

    vocab_size: int = 64
    hidden_size: int = 2560
    num_attention_heads: int = 40
    num_hidden_layers: int = 80
    intermediate_size: int | None = None
    pad_token_id: int = 1
    mask_token_id: int = 32
    initializer_range: float = 0.02
    classifier_dropout: float = 0.1
    num_labels: int = 2
    problem_type: str | None = None
    output_hidden_states: bool = False
    output_attentions: bool = False
    return_dict: bool = True
    attn_implementation: str = "sdpa"

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) is not a multiple of "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        derived = esmc_intermediate_size(self.hidden_size)
        if self.intermediate_size is None:
            self.intermediate_size = derived
        elif self.intermediate_size != derived:
            # The feed-forward width is not a free parameter here: the layers
            # derive it from hidden_size, so honouring a different value would
            # need a different FFN. Refuse rather than build the wrong shapes.
            raise ValueError(
                f"intermediate_size ({self.intermediate_size}) does not match the "
                f"value ESMC derives from hidden_size ({derived})."
            )

    @property
    def use_return_dict(self) -> bool:
        return self.return_dict

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, raw: dict) -> "EsmcConfig":
        """Build a config from a parsed ``config.json``.

        Keys this class does not read are ignored. ``transformers`` serialises
        knobs ESMC never varies — ``expansion_ratio``, ``scale_residue``,
        ``hidden_act``, ``rope_parameters``, ``attention_bias``, ``mlp_bias``,
        ``head_dim``, ``num_key_value_heads``, ``max_position_embeddings``, the
        parallelism plans — and this implementation fixes all of them.

        Raises
        ------
        KeyError
            If a field the architecture depends on is absent. Substituting a
            default there would silently build a different model and load
            garbage into it.
        """
        raw = _resolve_legacy_keys(raw)

        def required(key: str):
            if key not in raw:
                raise KeyError(
                    f"{CONFIG_NAME} is missing required ESMC field {key!r}; "
                    f"present keys are {sorted(raw)}"
                )
            return raw[key]

        if "num_labels" in raw:
            num_labels = raw["num_labels"]
        elif "id2label" in raw:
            num_labels = len(raw["id2label"])
        else:
            num_labels = 2
        # hidden_size=1 keeps __post_init__'s divisibility check happy; only the
        # non-architectural defaults are read off this instance.
        defaults = cls(hidden_size=1, num_attention_heads=1)
        return cls(
            vocab_size=required("vocab_size"),
            hidden_size=required("hidden_size"),
            num_attention_heads=required("num_attention_heads"),
            num_hidden_layers=required("num_hidden_layers"),
            intermediate_size=raw.get("intermediate_size"),
            pad_token_id=required("pad_token_id"),
            mask_token_id=required("mask_token_id"),
            initializer_range=raw.get("initializer_range", defaults.initializer_range),
            classifier_dropout=raw.get(
                "classifier_dropout", defaults.classifier_dropout
            ),
            num_labels=num_labels,
            problem_type=raw.get("problem_type"),
            output_hidden_states=raw.get(
                "output_hidden_states", defaults.output_hidden_states
            ),
            output_attentions=raw.get("output_attentions", defaults.output_attentions),
            return_dict=raw.get("return_dict", defaults.return_dict),
            attn_implementation=raw.get(
                "attn_implementation", defaults.attn_implementation
            ),
        )

    @classmethod
    def from_pretrained(cls, directory: str | os.PathLike) -> "EsmcConfig":
        with open(Path(directory) / CONFIG_NAME) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        raw = asdict(self)
        del raw["attn_implementation"]
        return {"model_type": self.model_type, **raw}

    def save_pretrained(self, save_directory: str | os.PathLike) -> None:
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / CONFIG_NAME, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
