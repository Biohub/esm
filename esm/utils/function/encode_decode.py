import re
from typing import Sequence

import torch

from esm.models.function_decoder import (
    FunctionTokenDecoder,
    merge_annotations,
)
from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer,
)
from esm.tokenization.residue_tokenizer import (
    ResidueAnnotationsTokenizer,
)
from esm.utils.constants import esm3 as C
from esm.utils.types import FunctionAnnotation


def encode_function_annotations(
    sequence: str,
    function_annotations: Sequence[FunctionAnnotation],
    function_tokens_tokenizer: InterProQuantizedTokenizer,
    residue_annotations_tokenizer: ResidueAnnotationsTokenizer,
    add_special_tokens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(residue_annotations_tokenizer, ResidueAnnotationsTokenizer), (
        "residue_annotations_tokenizer must be of type ResidueAnnotationsTokenizer"
    )

    # Split the user's annotations by type
    ft_annotations: list[FunctionAnnotation] = []
    ra_annotations: list[FunctionAnnotation] = []
    for fa in function_annotations:
        assert 1 <= fa.start <= fa.end <= len(sequence), (
            f"Invalid (start, end) in function annotation {fa}. Indices 1-indexed and [inclusive, inclusive]"
        )

        supported_label = False

        # Is it an InterPro label?
        if match := re.search(r"IPR\d+", fa.label):
            if match.group() in function_tokens_tokenizer.interpro_to_index:
                ft_annotations.append(fa)
                supported_label = True

        # Is it a function keyword?
        if fa.label in function_tokens_tokenizer._tfidf.vocab_to_index:
            ft_annotations.append(fa)
            supported_label = True

        # Is it a residue annotation?
        if fa.label in residue_annotations_tokenizer._labels:
            ra_annotations.append(fa)
            supported_label = True

        if not supported_label:
            raise ValueError(f"Unknown label in FunctionAnnotation: {fa.label}")

    # Convert function token FunctionAnnotations -> Tensor
    function_tokens = function_tokens_tokenizer.tokenize(
        annotations=ft_annotations, seqlen=len(sequence)
    )
    function_token_ids = function_tokens_tokenizer.encode(
        function_tokens, add_special_tokens=add_special_tokens
    )

    # Convert residue annotation FunctionAnnotations -> Tensor
    if ra_annotations:
        descriptions, starts, ends = zip(
            *[(anot.label, anot.start, anot.end) for anot in ra_annotations]
        )
    else:
        descriptions = starts = ends = None
    ra_tokens = residue_annotations_tokenizer.tokenize(
        {
            "interpro_site_descriptions": descriptions,
            "interpro_site_starts": starts,
            "interpro_site_ends": ends,
        },
        sequence=sequence,
        fail_on_mismatch=True,
    )
    residue_annotation_ids = residue_annotations_tokenizer.encode(
        ra_tokens, add_special_tokens=add_special_tokens
    )

    return function_token_ids, residue_annotation_ids


def decode_function_tokens(
    function_token_ids: torch.Tensor,
    function_token_decoder: FunctionTokenDecoder,
    function_tokens_tokenizer: InterProQuantizedTokenizer,
    decoder_annotation_threshold: float = 0.1,
    annotation_min_length: int | None = 5,
    annotation_gap_merge_max: int | None = 3,
) -> list[FunctionAnnotation]:
    """Decodes model prediction logits into function predictions.

    Merges function token and residue annotation predictions into a single
    set of FunctionAnnotation predictions.

    Args:
        function_token_ids: Tensor <float>[length, depth] of
            function token ids.
        residue_annotation_logits: Tensor  <float>[length, RA-vocab] of residue
            annotation binary classification logits.
        function_tokens_tokenizer: InterPro annotation tokenizer.
        residue_annotation_threshold: tokenizer of residue annotations.
        residue_annotation_threshold: predicted probability threshold for emitting
            a predicted residue annotation.
    Returns:
        Predicted function annotations merged from both predictions.
    """
    assert function_token_ids.ndim == 2, (
        "function_token_ids must be of shape (length, depth)"
    )

    annotations: list[FunctionAnnotation] = []

    # Function Annotations from predicted function tokens.
    decoded = function_token_decoder.decode(
        function_token_ids,
        tokenizer=function_tokens_tokenizer,
        annotation_threshold=decoder_annotation_threshold,
        annotation_min_length=annotation_min_length,
        annotation_gap_merge_max=annotation_gap_merge_max,
    )

    # Convert predicted InterPro annotation to FunctionAnnotation.
    annotations.extend(decoded["function_keywords"])
    for annotation in decoded["interpro_annotations"]:
        annotation: FunctionAnnotation
        label = function_tokens_tokenizer.format_annotation(annotation)
        annotations.append(
            FunctionAnnotation(label=label, start=annotation.start, end=annotation.end)
        )

    return annotations


def decode_residue_annotation_tokens(
    residue_annotations_token_ids: torch.Tensor,
    residue_annotations_tokenizer: ResidueAnnotationsTokenizer,
    annotation_min_length: int | None = 5,
    annotation_gap_merge_max: int | None = 3,
) -> list[FunctionAnnotation]:
    """Decodes residue annotation tokens into FunctionAnnotations.

    Args:
        tokens: Tensor <int>[length, MAX_RESIDUE_ANNOTATIONS] of residue annotation tokens.
        residue_annotations_tokenizer: Tokenizer of residue annotations.
        threshold: predicted probability threshold for emitting a predicted residue
            annotation.
    Returns:
        Predicted residue annotations.
    """
    assert residue_annotations_token_ids.ndim == 2, (
        "logits must be of shape (length, MAX_RESIDUE_ANNOTATIONS)"
    )

    annotations: list[FunctionAnnotation] = []

    for depth in range(0, C.MAX_RESIDUE_ANNOTATIONS):
        token_ids = residue_annotations_token_ids[:, depth]
        nonzero_indices = torch.nonzero(token_ids).squeeze(dim=1).cpu().numpy()
        if len(nonzero_indices) == 0:
            continue
        for loc in nonzero_indices:
            vocab_index: int = token_ids[loc].item()  # type: ignore
            label = residue_annotations_tokenizer.vocabulary[vocab_index]
            if label not in [*residue_annotations_tokenizer.special_tokens, "<none>"]:
                annotation = FunctionAnnotation(label=label, start=loc, end=loc)
                annotations.append(annotation)

    annotations = merge_annotations(annotations, merge_gap_max=annotation_gap_merge_max)

    # Drop very small annotations.
    if annotation_min_length is not None:
        annotations = [
            annotation
            for annotation in annotations
            if annotation.end - annotation.start + 1 >= annotation_min_length
        ]

    return annotations
