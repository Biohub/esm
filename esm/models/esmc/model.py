"""ESMC encoder and task heads.

ESMC is a protein language model trained by EvolutionaryScale with a
masked-token objective over amino acid sequences. The architecture is a
standard Transformer encoder with RoPE positional embeddings, QK LayerNorm and
SwiGLU feed-forward networks.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
import torch.nn as nn
from safetensors.torch import save_file
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from esm.models.esmc.checkpoint_layout import (
    native_to_published,
    published_to_native,
)
from esm.models.esmc.config import EsmcConfig
from esm.models.esmc.kernels import (
    FLASH_ATTN_INSTALLED,
    TE_INSTALLED,
    pad_input,
    unpad_input,
)
from esm.models.esmc.layers import (
    EsmcRotaryEmbedding,
    EsmcTransformerStack,
)
from esm.models.esmc.sae import EsmcSaeLayer
from esm.models.hub import (
    HubPreTrainedModel,
    resolve_model_dir,
)

_SAFETENSORS_INDEX = "model.safetensors.index.json"
_SAFETENSORS_SINGLE = "model.safetensors"


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EsmcOutput:
    """Output of :class:`EsmcModel`.

    Attributes
    ----------
    last_hidden_state : torch.Tensor | None
        Hidden states at the output of the last layer, after layer norm, of
        shape ``(batch, seq_len, d_model)``.
    hidden_states : torch.Tensor | None
        Stacked hidden states of shape
        ``(n_layers + 1, batch, seq_len, d_model)`` - one entry per block input
        plus the final post-LayerNorm output; populated when
        ``output_hidden_states=True``.
    sae_outputs : dict[str, torch.Tensor] | None
        SAE feature magnitudes keyed by SAE model name (sparse tensors).
    attentions : tuple[torch.Tensor, ...] | None
        Per-layer attention weights, populated when ``output_attentions=True``.
    """

    last_hidden_state: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    sae_outputs: dict[str, torch.Tensor] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    last_hidden_state_prenorm: torch.Tensor | None = None


@dataclass
class EsmcMaskedLMOutput:
    """Output of :class:`EsmcForMaskedLM`."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    sae_outputs: dict[str, torch.Tensor] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    last_hidden_state_prenorm: torch.Tensor | None = None


@dataclass
class EsmcTokenClassifierOutput:
    """Output of :class:`EsmcForTokenClassification`."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    sae_outputs: dict[str, torch.Tensor] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


@dataclass
class EsmcSequenceClassifierOutput:
    """Output of :class:`EsmcForSequenceClassification`."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    sae_outputs: dict[str, torch.Tensor] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


# ---------------------------------------------------------------------------
# State-dict / weight loading helpers
# ---------------------------------------------------------------------------


def _drop_te_extra_state(
    module: nn.Module, state_dict: dict, prefix: str, local_metadata: dict
) -> None:
    """Strip Transformer Engine's non-tensor ``_extra_state`` entries; TE
    returns these as ``io.BytesIO`` objects the checkpoint writer cannot
    dtype-infer."""
    for key in list(state_dict):
        if key.endswith("_extra_state"):
            del state_dict[key]


def _ignore_te_extra_state(module: nn.Module, incompatible_keys) -> None:
    """The load-side counterpart of :func:`_drop_te_extra_state`.

    Without it a model cannot even reload its own ``state_dict``, since the keys
    stripped on the way out are still expected on the way in. They hold nothing
    but fp8 scaling metadata, which is empty until fp8 runs and is recomputed
    when it does.
    """
    incompatible_keys.missing_keys[:] = [
        key
        for key in incompatible_keys.missing_keys
        if not key.endswith("_extra_state")
    ]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class EsmcPreTrainedModel(HubPreTrainedModel):
    """ESMC weight initialisation on top of the shared Hub loader."""

    config_class = EsmcConfig
    _keys_to_ignore_on_load_unexpected = [r"\._extra_state$"]

    def _init_weights(
        self, module: nn.Module, device: torch.device | str | None = None
    ):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, EsmcRotaryEmbedding):
            # `post_init` reaches here through `nn.Module.apply`, which passes no
            # device, so fall back to where the buffers already are. Defaulting to
            # the CPU moved them off a CUDA model and broke the next forward.
            module.reset_parameters(
                device=module.inv_freq.device if device is None else device
            )

    def _materialize_uninitialized(self, device: torch.device | str = "cpu") -> None:
        """Materialize and initialize parameters/buffers still on the meta
        device after a partial checkpoint load (e.g. a freshly-added head or
        the non-persistent RoPE buffers)."""
        for module in self.modules():
            direct = list(module._parameters.values()) + list(module._buffers.values())
            if any(t is not None and t.is_meta for t in direct):
                module.to_empty(device=device, recurse=False)
                self._init_weights(module, device=device)

    @classmethod
    def _adapt_checkpoint_keys(
        cls, raw: dict[str, torch.Tensor], model_keys: set[str]
    ) -> dict[str, torch.Tensor]:
        """Align published-checkpoint keys onto a target module tree.

        The published checkpoints are saved from ``EsmcForMaskedLM``, so
        backbone keys carry an ``esmc.`` prefix and the head keys are
        ``lm_head.*``. Keys are first translated out of the published layout
        (see :mod:`.checkpoint_layout`), then matched against the target tree:
        this keeps keys that already match, strips the ``esmc.`` prefix when
        loading the bare encoder, drops Transformer-Engine ``_extra_state``
        entries, and ignores keys with no counterpart (e.g. ``lm_head.*`` when
        loading :class:`EsmcModel`).
        """
        adapted: dict[str, torch.Tensor] = {}
        for key, value in raw.items():
            if key.endswith("_extra_state"):
                continue
            if key in model_keys:
                adapted[key] = value
            elif key.startswith("esmc.") and key[len("esmc.") :] in model_keys:
                adapted[key[len("esmc.") :]] = value
        return adapted

    @classmethod
    def _normalize_checkpoint_layout(
        cls, raw: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return published_to_native(raw)

    def save_pretrained(self, save_directory: str | os.PathLike) -> None:
        """Write ``config.json`` and ``model.safetensors`` in published layout.

        The in-memory state dict keeps Transformer Engine's fused parameters, so
        it is translated on the way out; see :mod:`.checkpoint_layout`.
        """
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(directory)
        state = {k: v for k, v in self.state_dict().items() if v is not None}
        save_file(
            native_to_published(state),
            str(directory / "model.safetensors"),
            metadata={"format": "pt"},
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        attn_implementation: str | None = None,
        revision: str | None = None,
        cache_dir: str | os.PathLike | None = None,
        token: str | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> Self:
        """Load an ESMC model from a local directory or the HuggingFace Hub.

        ``device`` defaults to CUDA when one is present, and selects the fused
        kernels as well as the placement: they are CUDA-only, so a model built
        for the CPU must not use them.
        """
        local_dir = resolve_model_dir(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
        )
        config = cls.config_class.from_pretrained(local_dir)
        if attn_implementation is not None:
            config.attn_implementation = attn_implementation
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Sets the default device the encoder reads to choose its kernels.
        with torch.device(device):
            return cls._load_pretrained(local_dir, config, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Base encoder
# ---------------------------------------------------------------------------


class EsmcModel(EsmcPreTrainedModel):
    """The bare ESMC encoder outputting raw hidden states.

    ``EsmcModel(config)`` is fully constructible from config alone with random
    weights and performs no network or disk I/O, so it can be embedded as a
    sub-module and populated later; checkpoint loading lives in
    :meth:`from_pretrained`. Its ``state_dict`` has two top-level module groups,
    ``embed.*`` (the token embedding) and ``transformer.*`` (the
    :class:`EsmcTransformerStack`: ``transformer.blocks.<i>.*`` and
    ``transformer.norm.weight``); the non-persistent RoPE buffers are omitted.
    Per-layer hidden states are exposed via ``forward(..., output_hidden_states=True)``
    as :attr:`EsmcOutput.hidden_states`, shape
    ``(n_layers + 1, batch, seq_len, d_model)``.
    """

    _SAE_KEY_RE = re.compile(r"layer(\d+)")
    _keys_to_ignore_on_load_unexpected = [r"\._extra_state$", r"^lm_head\."]

    def __init__(self, config: EsmcConfig):
        super().__init__(config)
        # Every fused kernel is CUDA-only, so all of them hang off one condition:
        # the device this model's parameters are being created on. Deriving them
        # separately is what let the LayerNorm and attention gates disagree.
        on_cuda = torch.get_default_device().type == "cuda"
        self._use_flash_attn = (
            on_cuda
            and FLASH_ATTN_INSTALLED
            and config.attn_implementation == "flash_attention_2"
        )
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer = EsmcTransformerStack(
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            use_flash_attn=self._use_flash_attn,
            use_te=on_cuda and TE_INSTALLED,
        )
        self._sae_models: nn.ModuleDict = nn.ModuleDict()
        self._register_state_dict_hook(_drop_te_extra_state)
        self.register_load_state_dict_post_hook(_ignore_te_extra_state)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed = value

    def add_sae_models(self, sae_models: list[EsmcSaeLayer]) -> None:
        """Register one or more SAEs obtained from an :class:`EsmcSaeModel`.

        Each is keyed by ``f"layer{N}"`` (the backbone-layer index ``N`` the
        SAE is trained against). Attaching two SAEs for the same backbone layer
        raises - only one SAE per layer can be active.
        """
        for layer in sae_models:
            assert isinstance(layer, EsmcSaeLayer), (
                f"Expected an SAE layer (model.layers['<idx>']), got "
                f"{type(layer).__name__}."
            )
            key = f"layer{int(layer.layer)}"
            if key in self._sae_models:
                raise ValueError(
                    f"An SAE is already registered at {key!r}. Only one SAE "
                    "per backbone layer can be active - pick a different layer "
                    "on one of them, or attach in a fresh model."
                )
            self._sae_models[key] = layer

    def _get_sae_layer_num_requested(self, model_name: str) -> int:
        match = self._SAE_KEY_RE.fullmatch(model_name)
        assert match is not None, (
            f"Unexpected SAE key {model_name!r}; expected 'layer{{N}}'."
        )
        return int(match.group(1))

    def _validate_sae_inputs(self, input_ids: torch.Tensor) -> None:
        assert torch.all(input_ids != self.config.mask_token_id), (
            "SAE inputs must not contain mask tokens. "
            "SAEs were trained on unmasked sequences."
        )

    def _get_sae_outputs(
        self,
        hidden_states: torch.Tensor,
        layers_to_collect: list[int],
        token_mask: torch.Tensor,
        normalize_sae: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run all registered SAEs and return their feature magnitudes."""
        layer_to_idx = {layer: idx for idx, layer in enumerate(layers_to_collect)}
        sae_outputs: dict[str, torch.Tensor] = {}

        for model_name, sae_module in self._sae_models.items():
            assert isinstance(sae_module, EsmcSaeLayer)
            layer = sae_module
            requested_layer = self._get_sae_layer_num_requested(model_name)
            layer_idx = layer_to_idx[requested_layer]
            layer_states = hidden_states[layer_idx].clone().to(self.device)

            sae_out = layer.get_sae_output(layer_states, token_mask)
            features = sae_out.feature_magnitudes.detach()

            if normalize_sae:
                features = (features / layer.max) * layer.idf

            sae_outputs[model_name] = features.to_sparse()

        return sae_outputs

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        compute_sae: bool = True,
        normalize_sae: bool = False,
    ) -> tuple[torch.Tensor, ...] | EsmcOutput:
        """Encode ``input_ids`` and return hidden states.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token ids of shape ``(batch, seq_len)``.
        attention_mask : torch.Tensor | None
            Boolean/int mask of real tokens; inferred from ``pad_token_id`` when
            omitted and ``sequence_id`` is not given.
        sequence_id : torch.Tensor | None
            Integer chain-ID tensor for chain-aware attention masking; tokens
            sharing a non-negative value attend to each other, padding is
            ``-1``. May be combined with ``attention_mask``, which is folded in
            by marking the masked positions ``-1``. The ``flash_attention_2``
            backend only supports single-chain inputs.
        output_hidden_states, output_attentions, return_dict : bool | None
            Override the corresponding config defaults.
        compute_sae : bool
            Run any SAEs registered via :meth:`add_sae_models`.
        normalize_sae : bool
            Scale SAE feature magnitudes by ``idf / max``.
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        output_sae = compute_sae and len(self._sae_models) > 0

        # Determine which intermediate layers to collect. When SAEs are
        # registered we must collect at least the layers they target.
        if output_hidden_states:
            layers_to_collect: list[int] = list(
                range(self.config.num_hidden_layers + 1)
            )
        elif output_sae:
            layers_to_collect = sorted(
                {self._get_sae_layer_num_requested(name) for name in self._sae_models}
            )
        else:
            layers_to_collect = []

        user_supplied_sequence_id = sequence_id is not None
        if sequence_id is not None:
            # Padding is spelled -1 in a sequence_id, so fold any mask the caller
            # also passed into it rather than dropping one of the two. Without
            # this, padding keeps a real chain id and stays visible to queries in
            # that chain, which corrupts their output rather than just the pad
            # rows.
            if attention_mask is not None:
                sequence_id = sequence_id.masked_fill(~attention_mask.bool(), -1)
            bool_mask = sequence_id >= 0
        else:
            if attention_mask is None:
                attention_mask = input_ids != self.config.pad_token_id  # ty:ignore[invalid-assignment]
            assert attention_mask is not None
            bool_mask = attention_mask.bool()

        x = self.embed(input_ids)
        b, l_ = x.shape[:2]

        if self._use_flash_attn:
            if sequence_id is not None and (sequence_id > 0).any():
                raise ValueError(
                    "Multi-chain ``sequence_id`` (any value > 0) is not "
                    "supported with attn_implementation='flash_attention_2'. "
                    "Re-load the model with attn_implementation='sdpa' (or "
                    "'eager') for chain-aware attention masking."
                )
            assert unpad_input is not None
            x, indices, *_ = unpad_input(x, bool_mask)
        else:
            indices = None

        # One of chain ids or padding decides the mask, never both - the same
        # split `transformers` makes. Chain ids compare query against key;
        # padding hides a key from every query, so it broadcasts over the query
        # axis. ``True`` means attend.
        if self._use_flash_attn:
            trans_seq_id = bool_mask
        elif user_supplied_sequence_id:
            assert sequence_id is not None
            trans_seq_id = (
                sequence_id.unsqueeze(-1) == sequence_id.unsqueeze(-2)
            ).unsqueeze(1)
        elif bool_mask.all() and not output_attentions:
            # No padding to mask, so the fused kernels can take it;
            # output_attentions forces the manual branch regardless.
            trans_seq_id = None
        else:
            trans_seq_id = bool_mask[:, None, None, :]

        last_hidden_state, prenorm, collected, attentions = self.transformer(
            x,
            sequence_id=trans_seq_id,
            layers_to_collect=layers_to_collect,
            output_attentions=output_attentions,
        )

        if self._use_flash_attn:
            assert indices is not None and pad_input is not None
            last_hidden_state = pad_input(last_hidden_state, indices, b, l_)
            prenorm = pad_input(prenorm, indices, b, l_)
            collected = tuple(pad_input(h, indices, b, l_) for h in collected)

        collected_tensor: torch.Tensor | None = (
            torch.stack(collected, dim=0) if collected else None
        )

        sae_outputs: dict[str, torch.Tensor] | None = None
        if output_sae and collected_tensor is not None:
            assert input_ids is not None
            self._validate_sae_inputs(input_ids)
            sae_outputs = self._get_sae_outputs(
                collected_tensor, layers_to_collect, bool_mask, normalize_sae
            )

        hidden_states_tensor = collected_tensor if output_hidden_states else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    hidden_states_tensor,
                    sae_outputs,
                    attentions,
                ]
                if v is not None
            )  # ty:ignore[invalid-return-type]

        return EsmcOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states_tensor,
            sae_outputs=sae_outputs,
            attentions=attentions,
            last_hidden_state_prenorm=prenorm if output_hidden_states else None,
        )


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


def _esmc_lm_head(
    d_model: int, output_dim: int, hidden_dim: int | None = None
) -> nn.Sequential:
    """Linear -> GELU -> LayerNorm -> Linear projection head for masked LM."""
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )


class EsmcForMaskedLM(EsmcPreTrainedModel):
    """ESMC with a masked language modelling head."""

    def __init__(self, config: EsmcConfig):
        super().__init__(config)
        self.esmc = EsmcModel(config)
        self.lm_head = _esmc_lm_head(config.hidden_size, config.vocab_size)

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head[-1]  # ty:ignore[invalid-return-type]

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head[-1] = new_embeddings

    def add_sae_models(self, sae_models: list[EsmcSaeLayer]) -> None:
        """Proxy to :meth:`EsmcModel.add_sae_models`."""
        self.esmc.add_sae_models(sae_models)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.Tensor | None = None,
        compute_sae: bool = True,
        normalize_sae: bool = False,
    ) -> tuple[torch.Tensor, ...] | EsmcMaskedLMOutput:
        """Run the encoder and project to per-token vocabulary logits.

        Parameters
        ----------
        labels : torch.Tensor | None
            Target token ids for the masked-LM loss; positions with label
            ``-100`` are ignored.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_outputs = self.esmc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            compute_sae=compute_sae,
            normalize_sae=normalize_sae,
        )

        logits = self.lm_head(encoder_outputs.last_hidden_state)

        loss: torch.Tensor | None = None
        if labels is not None:
            loss = CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    logits,
                    encoder_outputs.last_hidden_state,
                    encoder_outputs.hidden_states,
                    encoder_outputs.sae_outputs,
                    encoder_outputs.attentions,
                ]
                if v is not None
            )

        return EsmcMaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            sae_outputs=encoder_outputs.sae_outputs,
            attentions=encoder_outputs.attentions,
            last_hidden_state_prenorm=encoder_outputs.last_hidden_state_prenorm,
        )


class _EsmcClassificationHead(nn.Module):
    """Dense classification head applied to the ``<cls>`` token."""

    def __init__(self, config: EsmcConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states[:, 0, :]  # <cls> token
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class EsmcForSequenceClassification(EsmcPreTrainedModel):
    """ESMC with a sequence-level classification head over the ``<cls>``
    token. Supports regression, single-label and multi-label classification.
    """

    _keys_to_ignore_on_load_unexpected = [r"\._extra_state$", r"^lm_head\."]
    _keys_to_ignore_on_load_missing = [r"^classifier\."]

    def __init__(self, config: EsmcConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.esmc = EsmcModel(config)
        self.classifier = _EsmcClassificationHead(config)

    def add_sae_models(self, sae_models: list[EsmcSaeLayer]) -> None:
        """Proxy to :meth:`EsmcModel.add_sae_models`."""
        self.esmc.add_sae_models(sae_models)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.Tensor | None = None,
        compute_sae: bool = True,
        normalize_sae: bool = False,
    ) -> tuple[torch.Tensor, ...] | EsmcSequenceClassifierOutput:
        """Run the encoder and classify each sequence from its ``<cls>``
        representation.

        Parameters
        ----------
        labels : torch.Tensor | None
            Class indices (or float targets for regression).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_outputs = self.esmc(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            compute_sae=compute_sae,
            normalize_sae=normalize_sae,
        )
        logits = self.classifier(encoder_outputs.last_hidden_state)

        loss: torch.Tensor | None = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(
                    logits.squeeze() if self.num_labels == 1 else logits,
                    labels.squeeze() if self.num_labels == 1 else labels,
                )
            elif self.config.problem_type == "single_label_classification":
                loss = CrossEntropyLoss()(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss = BCEWithLogitsLoss()(logits, labels)

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    logits,
                    encoder_outputs.last_hidden_state,
                    encoder_outputs.hidden_states,
                    encoder_outputs.sae_outputs,
                    encoder_outputs.attentions,
                ]
                if v is not None
            )

        return EsmcSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            sae_outputs=encoder_outputs.sae_outputs,
            attentions=encoder_outputs.attentions,
        )


class EsmcForTokenClassification(EsmcPreTrainedModel):
    """ESMC with a per-token classification head."""

    _keys_to_ignore_on_load_unexpected = [r"\._extra_state$", r"^lm_head\."]
    _keys_to_ignore_on_load_missing = [r"^classifier\."]

    def __init__(self, config: EsmcConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.esmc = EsmcModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def add_sae_models(self, sae_models: list[EsmcSaeLayer]) -> None:
        """Proxy to :meth:`EsmcModel.add_sae_models`."""
        self.esmc.add_sae_models(sae_models)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.Tensor | None = None,
        compute_sae: bool = True,
        normalize_sae: bool = False,
    ) -> tuple[torch.Tensor, ...] | EsmcTokenClassifierOutput:
        """Run the encoder and classify every token.

        Parameters
        ----------
        labels : torch.Tensor | None
            Per-token class indices; positions with ``-100`` are ignored.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_outputs = self.esmc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            compute_sae=compute_sae,
            normalize_sae=normalize_sae,
        )

        sequence_output = self.dropout(encoder_outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss: torch.Tensor | None = None
        if labels is not None:
            loss = CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.num_labels), labels.to(logits.device).view(-1)
            )

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    logits,
                    encoder_outputs.last_hidden_state,
                    encoder_outputs.hidden_states,
                    encoder_outputs.sae_outputs,
                    encoder_outputs.attentions,
                ]
                if v is not None
            )

        return EsmcTokenClassifierOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            sae_outputs=encoder_outputs.sae_outputs,
            attentions=encoder_outputs.attentions,
        )
