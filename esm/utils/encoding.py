from typing import Sequence

import torch
import torch.nn.functional as F

from esm.models.vqvae import StructureTokenEncoder
from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer as EsmFunctionTokenizer,
)
from esm.tokenization.residue_tokenizer import ResidueAnnotationsTokenizer
from esm.tokenization.sasa_tokenizer import SASADiscretizingTokenizer
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.tokenization.ss_tokenizer import SecondaryStructureTokenizer
from esm.tokenization.structure_tokenizer import StructureTokenizer
from esm.utils.constants import esm3 as C
from esm.utils.function.encode_decode import encode_function_annotations
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation


# Raw Defaults
def get_default_sequence(sequence_length: int) -> str:
    return C.MASK_STR_SHORT * sequence_length


def get_default_secondary_structure(sequence_length: int) -> str:
    return C.MASK_STR_SHORT * sequence_length


def get_default_sasa(sequence_length: int) -> Sequence[float | str | None]:
    return [None] * sequence_length


# Tokenization
def tokenize_sequence(
    sequence: str,
    sequence_tokenizer: EsmSequenceTokenizer,
    add_special_tokens: bool = True,
) -> torch.Tensor:
    sequence = sequence.replace(C.MASK_STR_SHORT, sequence_tokenizer.mask_token)
    sequence_tokens = sequence_tokenizer.encode(
        sequence, add_special_tokens=add_special_tokens
    )
    sequence_tokens = torch.tensor(sequence_tokens, dtype=torch.int64)
    return sequence_tokens


def tokenize_structure(
    coordinates: torch.Tensor,
    structure_encoder: StructureTokenEncoder,
    structure_tokenizer: StructureTokenizer,
    reference_sequence: str = "",
    add_special_tokens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(structure_encoder.parameters()).device
    chain = ProteinChain.from_atom37(
        coordinates, sequence=reference_sequence if reference_sequence else None
    )

    # Setup padding
    if reference_sequence and len(reference_sequence) != coordinates.size(0):
        raise ValueError(
            f"Reference sequence length ({len(reference_sequence)}) does not match the number of residues in the coordinates ({coordinates.size(0)})"
        )

    left_pad = 0
    right_pad = 0

    if add_special_tokens:
        left_pad += 1  # Add space for BOS token
        right_pad += 1  # Add space for EOS token

    coordinates, plddt, residue_index = chain.to_structure_encoder_inputs()
    coordinates = coordinates.to(device)  # (1, L, 37, 3)
    plddt = plddt.to(device)  # (1, L)
    residue_index = residue_index.to(device)  # (1, L)
    _, structure_tokens = structure_encoder.encode(
        coordinates, residue_index=residue_index
    )
    coordinates = torch.squeeze(coordinates, dim=0)  # (L, 37, 3)
    plddt = torch.squeeze(plddt, dim=0)  # (L,)
    structure_tokens = torch.squeeze(structure_tokens, dim=0)  # (L,)

    # Add space for BOS and EOS tokens
    if add_special_tokens:
        coordinates = F.pad(
            coordinates, (0, 0, 0, 0, left_pad, right_pad), value=torch.inf
        )
        plddt = F.pad(plddt, (left_pad, right_pad), value=0)
        structure_tokens = F.pad(
            structure_tokens,
            (left_pad, right_pad),
            value=structure_tokenizer.mask_token_id,
        )
        structure_tokens[0] = structure_tokenizer.bos_token_id
        structure_tokens[-1] = structure_tokenizer.eos_token_id
    return coordinates, plddt, structure_tokens


def tokenize_secondary_structure(
    secondary_structure: str | Sequence[str],
    secondary_structure_tokenizer: SecondaryStructureTokenizer,
    add_special_tokens: bool = True,
) -> torch.Tensor:
    if isinstance(secondary_structure, str):
        # Ensure only one char per token
        secondary_structure = secondary_structure.replace(
            secondary_structure_tokenizer.mask_token, C.MASK_STR_SHORT
        )

    # Input as list of chars
    secondary_structure = [char for char in secondary_structure]

    # Use tokenizer's mask token
    secondary_structure = [
        secondary_structure_tokenizer.mask_token if char == C.MASK_STR_SHORT else char
        for char in secondary_structure
    ]

    secondary_structure_tokens = secondary_structure_tokenizer.encode(
        secondary_structure, add_special_tokens=add_special_tokens
    )
    return secondary_structure_tokens


def tokenize_sasa(
    sasa: Sequence[float | str | None],
    sasa_tokenizer: SASADiscretizingTokenizer,
    add_special_tokens: bool = True,
):
    sasa_tokens = sasa_tokenizer.encode(
        [sasa_tokenizer.mask_token if value is None else value for value in sasa],
        add_special_tokens=add_special_tokens,
    )
    return sasa_tokens


def tokenize_function_annotations(
    function_annotations: Sequence[FunctionAnnotation],
    reference_sequence: str,
    function_tokenizer: EsmFunctionTokenizer,
    residue_annotation_tokenizer: ResidueAnnotationsTokenizer,
    add_special_tokens: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    function_tokens, residue_annotation_tokens = encode_function_annotations(
        sequence=reference_sequence,
        function_annotations=function_annotations,
        function_tokens_tokenizer=function_tokenizer,
        residue_annotations_tokenizer=residue_annotation_tokenizer,
        add_special_tokens=add_special_tokens,
    )
    return function_tokens, residue_annotation_tokens


# Tokenized Defaults
def get_default_sequence_tokens(
    sequence_length: int, sequence_tokenizer: EsmSequenceTokenizer
) -> torch.Tensor:
    assert sequence_tokenizer.mask_token_id is not None
    assert sequence_tokenizer.bos_token_id is not None
    assert sequence_tokenizer.eos_token_id is not None

    sequence_tokens = torch.full(
        (sequence_length + 2,), sequence_tokenizer.mask_token_id
    )
    sequence_tokens[0] = sequence_tokenizer.bos_token_id
    sequence_tokens[-1] = sequence_tokenizer.eos_token_id

    return sequence_tokens


def get_default_structure_tokens(
    sequence_length: int, structure_tokenizer: StructureTokenizer
) -> torch.Tensor:
    structure_tokens = (
        torch.ones((sequence_length + 2,), dtype=torch.int64)
        * structure_tokenizer.mask_token_id
    )
    # Always include BOS and EOS tokens
    structure_tokens[0] = structure_tokenizer.bos_token_id
    structure_tokens[-1] = structure_tokenizer.eos_token_id
    return structure_tokens


def get_default_secondary_structure_tokens(
    sequence_length: int, secondary_structure_tokenizer: SecondaryStructureTokenizer
) -> torch.Tensor:
    ss8_tokens = torch.full(
        (sequence_length + 2,), secondary_structure_tokenizer.mask_token_id
    )
    ss8_tokens[0] = secondary_structure_tokenizer.bos_token_id
    ss8_tokens[-1] = secondary_structure_tokenizer.eos_token_id

    return ss8_tokens


def get_default_sasa_tokens(
    sequence_length: int, sasa_tokenizer: SASADiscretizingTokenizer
) -> torch.Tensor:
    sasa_tokens = torch.full((sequence_length + 2,), sasa_tokenizer.mask_token_id)
    sasa_tokens[0] = sasa_tokenizer.bos_token_id
    sasa_tokens[-1] = sasa_tokenizer.eos_token_id
    return sasa_tokens


def get_default_function_tokens(
    sequence_length: int, function_tokenizer: EsmFunctionTokenizer
) -> torch.Tensor:
    function_tokens = (
        torch.ones((sequence_length + 2, function_tokenizer.depth), dtype=torch.int64)
        * function_tokenizer.pad_token_id
    )
    # Always include BOS and EOS tokens
    function_tokens[0] = function_tokenizer.bos_token_id
    function_tokens[-1] = function_tokenizer.eos_token_id
    return function_tokens


def get_default_residue_annotation_tokens(
    sequence_length: int, residue_annotation_tokenizer: ResidueAnnotationsTokenizer
) -> torch.Tensor:
    residue_annotation_tokens = (
        torch.ones((sequence_length + 2, C.MAX_RESIDUE_ANNOTATIONS), dtype=torch.int64)
        * residue_annotation_tokenizer.pad_token_id
    )
    # Always include BOS and EOS tokens
    residue_annotation_tokens[0] = residue_annotation_tokenizer.bos_token_id
    residue_annotation_tokens[-1] = residue_annotation_tokenizer.eos_token_id
    return residue_annotation_tokens
