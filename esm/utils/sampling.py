import warnings
from typing import Literal

import attr
import torch
import torch.nn.functional as F

from esm.sdk.api import ESMProteinTensor, SamplingConfig, SamplingTrackConfig
from esm.tokenization import TokenizerCollectionProtocol, get_invalid_tokenizer_ids
from esm.tokenization.function_tokenizer import InterProQuantizedTokenizer
from esm.utils.constants.esm3 import (
    MAX_RESIDUE_ANNOTATIONS,
    SASA_DISCRETIZATION_BOUNDARIES,
)


def _non_batched_dims(k: str, v: torch.Tensor):
    match k:
        case "sequence":
            return 1
        case "structure":
            if v.is_floating_point():
                # This is the one hot soft structure token.
                return 2
            else:
                # This is the normal int structure token.
                return 1
        case "secondary_structure":
            return 1
        case "sasa":
            return 1
        case "function":
            return 2
        case "residue_annotations":
            return 2
        case "coordinates":
            return 3
        case _:
            raise ValueError(f"Unknown dim for track {k}")


class _BatchedESMProteinTensor(ESMProteinTensor):
    @staticmethod
    def from_protein_tensor(protein: ESMProteinTensor):
        def _maybe_unsqueeze(x: torch.Tensor | None):
            return x.unsqueeze(0) if x is not None else None

        return _BatchedESMProteinTensor(
            sequence=_maybe_unsqueeze(protein.sequence),
            structure=_maybe_unsqueeze(protein.structure),
            secondary_structure=_maybe_unsqueeze(protein.secondary_structure),
            sasa=_maybe_unsqueeze(protein.sasa),
            function=_maybe_unsqueeze(protein.function),
            residue_annotations=_maybe_unsqueeze(protein.residue_annotations),
            coordinates=_maybe_unsqueeze(protein.coordinates),
        )

    def __len__(self) -> int:
        def get_len(k, v) -> int:
            assert len(v.shape) == _non_batched_dims(k, v) + 1
            return v.size(1)

        l = self._detect_attribute(get_len, "length")
        return l if l is not None else 0

    @property
    def batch_size(self) -> int:
        def get_batch_size(k, v) -> int:
            assert len(v.shape) == _non_batched_dims(k, v) + 1
            return v.size(0)

        d = self._detect_attribute(get_batch_size, "batch size")
        assert d is not None
        return d

    def slice(self, i: int, sequence_len: int | None = None) -> ESMProteinTensor:
        def _maybe_slice(x: torch.Tensor | None):
            if x is None:
                return None
            row = x[i]
            if sequence_len is not None:
                row = row[:sequence_len]
            return row

        return ESMProteinTensor(
            sequence=_maybe_slice(self.sequence),
            structure=_maybe_slice(self.structure),
            secondary_structure=_maybe_slice(self.secondary_structure),
            sasa=_maybe_slice(self.sasa),
            function=_maybe_slice(self.function),
            residue_annotations=_maybe_slice(self.residue_annotations),
            coordinates=_maybe_slice(self.coordinates),
        )

    def set_slice(self, i: int, slice: ESMProteinTensor):
        """Update the i-th slice of this tensor data class."""
        for f in attr.fields(ESMProteinTensor):
            s = getattr(self, f.name)
            v = getattr(slice, f.name)

            assert v is None or (
                v is not None and s is not None
            ), f"Trying to set a slice on None tensor ({f.name})."

            if v is not None:
                s[i, ...] = v


def get_default_sampling_config(
    tokenizers: TokenizerCollectionProtocol,
) -> SamplingConfig:
    tracks = [f.name for f in attr.fields(SamplingConfig)]
    sampling_config = SamplingConfig()
    for current_track in tracks:
        setattr(
            sampling_config,
            current_track,
            SamplingTrackConfig(
                invalid_ids=get_invalid_tokenizer_ids(
                    getattr(tokenizers, current_track)
                ),
                temperature=1.0,
                top_p=1.0,
                # TODO: Add different mask and padding tokens for all tracks
                # Some tracks have the same pad and mask, which causes ambiguity when sampling
                only_sample_masked_tokens=current_track
                not in ["secondary_structure", "sasa", "function"],
            ),
        )
    return sampling_config


def validate_sampling_config(
    sampling_config: SamplingConfig, on_invalid: Literal["raise", "warn"] = "warn"
):
    # Check that all tracks have topk_logprobs less or equal to MAX_TOP_K
    for track in attr.fields(SamplingConfig):
        track: attr.Attribute
        track_config = getattr(sampling_config, track.name, None)
        if isinstance(track_config, SamplingTrackConfig):
            max_topk = track.metadata["max_topk"]
            if track_config.topk_logprobs > max_topk:
                msg = (
                    f"Sampling track {track.name} has topk_logprobs={track_config.topk_logprobs} "
                    f"greater than MAX_TOPK={max_topk}."
                )
                if on_invalid == "raise":
                    raise AssertionError(msg)
                else:
                    warnings.warn(msg)


def sample_logits(
    logits: torch.Tensor,
    temperature: float | torch.Tensor,
    valid_ids: list[int] = [],
    top_p: float | torch.Tensor = 1.0,
    mask_logits_of_invalid_ids: bool = True,
):
    """Default sampling from logits.

    Args:
        logits is shape (..., vocab_size)
        temperature is broadcastable to (...)
    """
    if len(valid_ids) == 0:
        raise ValueError(
            "Can not sample logits if there are no valid ids to sample from."
        )

    if top_p < 1.0:
        logits = top_p_logits(logits, top_p=top_p)

    temperature = _tensorize_like(temperature, logits)
    batch_dims = logits.size()[:-1]
    logits = logits.reshape(-1, logits.shape[-1])

    # Only sample from valid ids
    # the /logits endpoint should receive unmodified logits
    if mask_logits_of_invalid_ids:
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[..., valid_ids] = False
        logits[mask] = -torch.inf

    if torch.all(temperature == 0):
        ids = logits.argmax(-1)
        return ids.reshape(*batch_dims)

    assert not torch.any(temperature == 0), "Partial temperature 0 not supported."

    # Sample from all logits
    probs = F.softmax(logits / temperature[..., None], dim=-1)
    ids = torch.multinomial(probs, 1).squeeze(1)

    ids = ids.reshape(*batch_dims)
    return ids


def sample_function_logits(
    logits: torch.Tensor,
    tokenizer: InterProQuantizedTokenizer,
    top_p: float | torch.Tensor = 1.0,
    temperature: float | torch.Tensor = 1.0,
    p_none_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Works with inputs that have batch dimension."""
    [B, L, D, V] = logits.shape
    assert D == tokenizer.depth

    if top_p < 1.0:
        logits = top_p_logits(logits, top_p=top_p)

    temperature = torch.ones_like(logits[..., 0]) * temperature

    log_p = F.log_softmax(logits / temperature[..., None], dim=-1)  # (B, L, D, V)

    # Choose which positions have no predicted function.
    none_index = tokenizer.vocab_to_index["<none>"]
    log_p_nones = log_p[..., none_index]  # (B, L, D)
    p_none = torch.exp(log_p_nones).mean(dim=-1)  # "Ensemble of <none> predictions"
    where_none = p_none > p_none_threshold  # (B, L)

    # Set probability of <none> to 0 for all not-none positions
    batch_size, seq_len, depth = log_p.shape[:-1]
    expanded_where_not_none = ~where_none.unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)
    expanded_where_not_none = expanded_where_not_none.expand(
        batch_size, seq_len, depth, 1
    )  # (B, L, D, 1)
    indices = torch.arange(log_p.shape[-1], device=log_p.device)  # (V,)
    mask = indices == none_index  # (V,)
    mask = expanded_where_not_none & mask  # (B, L, D, 1) x (V,) -> (B, L, D, V)
    log_p[mask] = -torch.inf

    ids = torch.argmax(log_p, dim=-1)  # (B, L, D)
    ids[where_none, :] = tokenizer.vocab_to_index["<none>"]

    return ids, log_p


def sample_residue_annotation_logits(
    logits: torch.Tensor, annotation_threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    # Take top residue annotations
    top_residue_annotations_idx = logits.argsort(dim=-1, descending=True)[
        ..., :MAX_RESIDUE_ANNOTATIONS
    ]  # (B, L, MAX_R)
    top_residue_annotations_logprobs = torch.gather(
        F.logsigmoid(logits), -1, top_residue_annotations_idx
    )  # (B, L, MAX_R)
    top_residue_annotations_probs = top_residue_annotations_logprobs.exp()
    # Keep only positive predictions
    is_negative = top_residue_annotations_probs < annotation_threshold
    top_residue_annotations_idx[is_negative] = 0

    top_residue_annotations_logprobs = top_residue_annotations_logprobs

    return top_residue_annotations_idx, top_residue_annotations_logprobs


def sample_sasa_logits(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    sampling_track_config: SamplingTrackConfig,
    mask_idx: int,
    valid_ids: list[int],
    mask_logits_of_invalid_ids: bool = True,
) -> torch.Tensor:
    # Only sample from valid ids
    # the /logits endpoint should receive unmodified logits
    if mask_logits_of_invalid_ids:
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[..., valid_ids] = False
        logits[mask] = -torch.inf

    sasa_probs = torch.nn.functional.softmax(logits, dim=-1)
    max_prob_idx = torch.argmax(sasa_probs, dim=-1)
    sasa_bins = torch.tensor([0] + SASA_DISCRETIZATION_BOUNDARIES, dtype=torch.float)
    sasa_bins = (sasa_bins[:-1] + sasa_bins[1:]) / 2
    sasa_bins = sasa_bins.to(sasa_probs.device)

    sampling_mask = get_sampling_mask(tokens, sampling_track_config, mask_idx)
    # Adjust sasa_values based on max_prob_idx conditions
    sasa_value = torch.sum(sasa_probs[..., 3:-1] * sasa_bins, dim=-1)
    sasa_value[max_prob_idx == 18] = float("inf")
    sasa_value[~sampling_mask] = float("inf")

    # Set BOS and EOS tokens to 0
    sasa_value[..., 0] = 0.0
    sasa_value[..., -1] = 0.0
    return sasa_value


def top_p_logits(logits: torch.Tensor, top_p: float | torch.Tensor) -> torch.Tensor:
    top_p = _tensorize_like(top_p, logits)

    batch_dims = logits.size()[:-1]
    logits = logits.reshape(-1, logits.shape[-1])

    # Sort logits in descending order and extract the mask for the top_p
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumsum_logits = sorted_logits.softmax(-1).cumsum(-1)
    top_p_mask = cumsum_logits <= top_p[:, None]

    # Make sure at least one token is sampled
    top_p_mask[:, 0] = True

    # Mask out the logits that are not in the top_p
    batch_indices_to_mask, _ = torch.where(~top_p_mask)
    vocab_indices_to_mask = sorted_indices[~top_p_mask]
    logits[batch_indices_to_mask, vocab_indices_to_mask] = torch.finfo(logits.dtype).min

    return logits.reshape(*batch_dims, -1)


def _tensorize_like(value: int | float | torch.Tensor, logits: torch.Tensor):
    if isinstance(value, (float, int)):
        value = torch.full_like(logits[..., 0], value, dtype=logits.dtype)
    return value.to(logits.device).expand_as(logits[..., 0]).reshape(-1)


def get_sampling_mask(
    tokens: torch.Tensor, sampling_track_config: SamplingTrackConfig, mask_idx: int
):
    # Do not sample at BOS and EOS tokens
    sampling_mask = torch.ones_like(tokens, dtype=torch.bool)  # (B, L, )
    sampling_mask[:, 0] = False
    sampling_mask[:, -1] = False

    # Do not sample at special token positions but allow sampling at mask token
    special_minus_mask = list(set(sampling_track_config.invalid_ids) - {mask_idx})
    if len(special_minus_mask) > 0:
        special_tokens = torch.tensor(special_minus_mask, device=tokens.device)
        assert special_tokens.numel() > 0
        sampling_mask = sampling_mask & (
            tokens[..., None] != special_tokens[None, :]
        ).all(-1)

    # Keep only samples from masked positions (if specified)
    if sampling_track_config.only_sample_masked_tokens:
        masked_tokens = tokens == mask_idx
        sampling_mask = sampling_mask & masked_tokens
    return sampling_mask
