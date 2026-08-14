"""Deprecated ``ESMC`` surface, kept so existing code keeps running.

``ESMC`` used to be one class holding an encoder, an LM head, a tokenizer and a
local implementation of ``ESMCInferenceClient``. Those are now four separate
things (:class:`EsmcModel`, :class:`EsmcForMaskedLM`, :class:`EsmcTokenizer` and
``esmc_client``). :class:`ESMC` here wraps a native
:class:`EsmcForMaskedLM` - the piece that owns the LM head, so
``sequence_logits`` is still available - and re-exposes the old method and
attribute names.

``hidden_states`` is realigned back onto the old layout - one entry per block
output, before the final LayerNorm - because the native model follows the
``transformers`` convention instead; see :func:`_legacy_hidden_states`.

"""

import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast

from esm.models.esmc.config import (
    ESMC_6B_HF_REPO,
    ESMC_300M_HF_REPO,
    ESMC_600M_HF_REPO,
    EsmcConfig,
)
from esm.models.esmc.model import EsmcForMaskedLM
from esm.models.esmc.tokenizer import EsmcTokenizer
from esm.sdk.api import (
    ESMProtein,
    ESMProteinTensor,
    ForwardTrackData,
    LogitsConfig,
    LogitsOutput,
)
from esm.utils.constants.models import (
    ESMC_6B,
    ESMC_300M,
    ESMC_600M,
)
from esm.utils.sampling import _BatchedESMProteinTensor

_DEPRECATION_MESSAGE = (
    "ESMC is deprecated and now wraps the native EsmcForMaskedLM. Use "
    "EsmcModel / EsmcForMaskedLM with EsmcTokenizer directly, or esmc_client(...) "
    "for the remote inference API."
)


@dataclass
class ESMCOutput:
    """Deprecated. Superseded by ``EsmcMaskedLMOutput``."""

    sequence_logits: torch.Tensor
    embeddings: torch.Tensor | None
    hidden_states: torch.Tensor | None
    attentions: tuple[torch.Tensor, ...] | None = None

    @property
    def logits(self) -> torch.Tensor:
        """Alias, so code written against ``EsmcMaskedLMOutput`` also works."""
        return self.sequence_logits

    def __getitem__(self, key: str):
        """Mapping access, so ``out["logits"]`` works as it does on the native
        output."""
        return self.logits if key == "logits" else getattr(self, key)


def _legacy_hidden_states(
    stacked: torch.Tensor | None, last_prenorm: torch.Tensor | None
) -> torch.Tensor | None:
    """Rebuild the old ``hidden_states`` stack from the native one.

    The old stack held one entry per block *output*, before the final
    LayerNorm: ``(out_0, ..., out_{n-1})``, so ``hidden_states[i]`` was block
    ``i``'s output. The native stack instead follows the ``transformers``
    convention of one entry per block *input* plus the normalised final state:
    ``(embeddings, out_0, ..., out_{n-2}, norm(out_{n-1}))``.

    Dropping the leading embeddings realigns the indices, and the final
    pre-LayerNorm state replaces the normalised one -- LayerNorm is not
    invertible, so it has to be carried out of the model rather than recovered.
    """
    if stacked is None:
        return None
    if last_prenorm is None:
        raise ValueError(
            "the native model returned hidden_states without "
            "last_hidden_state_prenorm, so the legacy stack cannot be rebuilt"
        )
    return torch.cat([stacked[1:-1], last_prenorm[None]], dim=0)


def _legacy_name_to_repo(model_name: str) -> str:
    """Map a pre-rename local model name onto its HuggingFace repo id.

    Anything unrecognised is passed through, so a repo id works directly.
    """
    return {
        ESMC_300M: ESMC_300M_HF_REPO,
        ESMC_600M: ESMC_600M_HF_REPO,
        ESMC_6B: ESMC_6B_HF_REPO,
    }.get(model_name, model_name)


class ESMC(nn.Module):
    """Deprecated wrapper around :class:`EsmcForMaskedLM`.

    Accepts the old constructor arguments, or wraps an already-built native
    model via ``model=``.
    """

    def __init__(
        self,
        d_model: int | None = None,
        n_heads: int | None = None,
        n_layers: int | None = None,
        # Pre-port callers passed an EsmSequenceTokenizer, so accept any fast
        # tokenizer: only __call__, pad_token_id and decode are used.
        tokenizer: PreTrainedTokenizerFast | None = None,
        use_flash_attn: bool = True,
        *,
        model: EsmcForMaskedLM | None = None,
    ):
        super().__init__()
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)

        if model is None:
            defaults = EsmcConfig()
            model = EsmcForMaskedLM(
                EsmcConfig(
                    hidden_size=defaults.hidden_size if d_model is None else d_model,
                    num_attention_heads=defaults.num_attention_heads
                    if n_heads is None
                    else n_heads,
                    num_hidden_layers=defaults.num_hidden_layers
                    if n_layers is None
                    else n_layers,
                    attn_implementation="flash_attention_2"
                    if use_flash_attn
                    else "sdpa",
                )
            )
        self.model = model
        self.tokenizer = EsmcTokenizer() if tokenizer is None else tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = ESMC_600M,
        device: torch.device | str | None = None,
        use_flash_attn: bool = True,
    ) -> "ESMC":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        model = EsmcForMaskedLM.from_pretrained(
            _legacy_name_to_repo(model_name),
            device=device,
            # The old loader cast to bf16 whenever it left the CPU.
            dtype=torch.bfloat16 if device.type != "cpu" else None,
            attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
        )
        return cls(model=model)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def raw_model(self) -> "ESMC":
        return self

    def state_dict(self, *args, **kwargs):
        """Delegate so checkpoint keys match the unwrapped native model."""
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.model.load_state_dict(state_dict, *args, **kwargs)

    def _tokenize(self, sequence: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(sequence, return_tensors="pt", padding=True)
        return encoded["input_ids"].to(self.device)

    def _detokenize(self, sequence: torch.Tensor) -> list[str]:
        pad = self.tokenizer.pad_token_id
        assert pad is not None
        assert sequence.ndim == 2
        return [
            self.tokenizer.decode(row[row != pad][1:-1]).replace(" ", "")
            for row in sequence
        ]

    def forward(
        self,
        sequence_tokens: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> ESMCOutput:
        # ``input_ids`` is accepted so a caller written against the native model
        # works here too, and no caller needs to branch on the model class.
        if input_ids is not None:
            if sequence_tokens is not None:
                raise ValueError("pass either sequence_tokens or input_ids, not both")
            sequence_tokens = input_ids

        # The old ``sequence_id`` defaulted to a boolean padding mask, which is
        # what ``attention_mask`` means now; a non-bool tensor is the integer
        # chain id the native model still calls ``sequence_id``.
        attention_mask = None
        if sequence_id is not None and sequence_id.dtype == torch.bool:
            attention_mask, sequence_id = sequence_id, None

        out = self.model(
            input_ids=sequence_tokens,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            # The old forward always stacked and returned hidden states.
            output_hidden_states=True,
            output_attentions=bool(output_attentions),
            return_dict=True,
        )
        return ESMCOutput(
            sequence_logits=out.logits,
            embeddings=out.last_hidden_state,
            hidden_states=_legacy_hidden_states(
                out.hidden_states, out.last_hidden_state_prenorm
            ),
            attentions=out.attentions,
        )

    def encode(self, input: ESMProtein) -> ESMProteinTensor:
        tokens = None if input.sequence is None else self._tokenize([input.sequence])[0]
        return ESMProteinTensor(
            sequence=tokens,
            potential_sequence_of_concern=input.potential_sequence_of_concern,
        ).to(self.device)

    def decode(self, input: ESMProteinTensor) -> ESMProtein:
        tokens = input.sequence
        assert tokens is not None
        batched = tokens[None] if tokens.ndim == 1 else tokens
        return ESMProtein(sequence=self._detokenize(batched)[0])

    def logits(
        self,
        input: ESMProteinTensor | _BatchedESMProteinTensor,
        config: LogitsConfig = LogitsConfig(),
    ) -> LogitsOutput:
        if not isinstance(input, _BatchedESMProteinTensor):
            input = _BatchedESMProteinTensor.from_protein_tensor(input)
        device = torch.device(input.device)

        with (
            torch.no_grad(),
            torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ),
        ):
            out = self(sequence_tokens=input.sequence)

        hidden = out.hidden_states
        if hidden is not None and config.ith_hidden_layer != -1:
            layer = config.ith_hidden_layer
            hidden = hidden[layer : layer + 1]
        return LogitsOutput(
            logits=ForwardTrackData(
                sequence=out.sequence_logits if config.sequence else None
            ),
            embeddings=out.embeddings if config.return_embeddings else None,
            hidden_states=hidden if config.return_hidden_states else None,
        )
