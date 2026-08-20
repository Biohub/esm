import pickle
import warnings
from typing import Any, Mapping, cast

import attr
import torch
from requests import Response

from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import StructureTokenDecoder
from esm.sdk.api import ESMProtein, ESMProteinTensor
from esm.tokenization import TokenizerCollectionProtocol
from esm.tokenization.function_tokenizer import InterProQuantizedTokenizer
from esm.tokenization.residue_tokenizer import ResidueAnnotationsTokenizer
from esm.tokenization.sasa_tokenizer import SASADiscretizingTokenizer
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.tokenization.ss_tokenizer import SecondaryStructureTokenizer
from esm.tokenization.structure_tokenizer import StructureTokenizer
from esm.tokenization.tokenizer_base import EsmTokenizerBase
from esm.utils.constants import api as api_constants
from esm.utils.constants import esm3 as C
from esm.utils.function.encode_decode import (
    decode_function_tokens,
    decode_residue_annotation_tokens,
)
from esm.utils.misc import maybe_list
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation


def decode_protein_tensor(
    input: ESMProteinTensor,
    tokenizers: TokenizerCollectionProtocol,
    structure_token_decoder: StructureTokenDecoder,
    function_token_decoder: FunctionTokenDecoder | None = None,
) -> ESMProtein:
    input = attr.evolve(input)  # Make a copy

    sequence = None
    secondary_structure = None
    sasa = None
    function_annotations = []

    coordinates = None

    # If all pad tokens, set to None
    for track in attr.fields(ESMProteinTensor):
        tokens: torch.Tensor | None = getattr(input, track.name)
        if track.name == "coordinates" or track.name == "potential_sequence_of_concern":
            continue
        if tokens is not None:
            tokens = tokens[1:-1]  # Remove BOS and EOS tokens
            tokens = tokens.flatten()  # For multi-track tensors
            track_tokenizer = getattr(tokenizers, track.name)
            if torch.all(tokens == track_tokenizer.pad_token_id):
                setattr(input, track.name, None)
            # If structure track has any mask tokens, do not decode.
            if track.name == "structure" and torch.any(
                tokens == track_tokenizer.mask_token_id
            ):
                setattr(input, track.name, None)

    if input.sequence is not None:
        sequence = decode_sequence(input.sequence, tokenizers.sequence)

    plddt, ptm, pae = None, None, None
    if input.structure is not None:
        # Note: We give priority to the structure tokens over the coordinates when decoding
        coordinates, plddt, ptm, pae = decode_structure(
            structure_tokens=input.structure,
            structure_decoder=structure_token_decoder,
            structure_tokenizer=tokenizers.structure,
            sequence=sequence,
        )
    elif input.coordinates is not None:
        coordinates = input.coordinates[1:-1, ...]

    if input.secondary_structure is not None:
        secondary_structure = decode_secondary_structure(
            input.secondary_structure, tokenizers.secondary_structure
        )
    if input.sasa is not None:
        sasa = decode_sasa(input.sasa, tokenizers.sasa)
    if input.function is not None:
        if function_token_decoder is None:
            raise ValueError(
                "Cannot decode function annotations without a function token decoder"
            )
        function_track_annotations = decode_function_annotations(
            input.function,
            function_token_decoder=function_token_decoder,
            function_tokenizer=tokenizers.function,
        )
        function_annotations.extend(function_track_annotations)
    if input.residue_annotations is not None:
        residue_annotations = decode_residue_annotations(
            input.residue_annotations, tokenizers.residue_annotations
        )
        function_annotations.extend(residue_annotations)

    return ESMProtein(
        sequence=sequence,
        secondary_structure=secondary_structure,
        sasa=sasa,  # type: ignore
        function_annotations=function_annotations if function_annotations else None,
        coordinates=coordinates,
        plddt=plddt,
        ptm=ptm,
        pae=pae,
        potential_sequence_of_concern=input.potential_sequence_of_concern,
    )


def _bos_eos_warn(msg: str, tensor: torch.Tensor, tok: EsmTokenizerBase):
    if tensor[0] != tok.bos_token_id:
        warnings.warn(
            f"{msg} does not start with BOS token, token is ignored. BOS={tok.bos_token_id} vs {tensor}"
        )
    if tensor[-1] != tok.eos_token_id:
        warnings.warn(
            f"{msg} does not end with EOS token, token is ignored. EOS='{tok.eos_token_id}': {tensor}"
        )


def decode_sequence(
    sequence_tokens: torch.Tensor, sequence_tokenizer: EsmSequenceTokenizer, **kwargs
) -> str:
    _bos_eos_warn("Sequence", sequence_tokens, sequence_tokenizer)
    sequence = sequence_tokenizer.decode(sequence_tokens, **kwargs)
    sequence = sequence.replace(" ", "")
    sequence = sequence.replace(sequence_tokenizer.mask_token, C.MASK_STR_SHORT)
    sequence = sequence.replace(sequence_tokenizer.cls_token, "")
    sequence = sequence.replace(sequence_tokenizer.eos_token, "")

    return sequence


def decode_structure(
    structure_tokens: torch.Tensor,
    structure_decoder: StructureTokenDecoder,
    structure_tokenizer: StructureTokenizer,
    sequence: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    is_singleton = len(structure_tokens.size()) == 1
    if is_singleton:
        structure_tokens = structure_tokens.unsqueeze(0)
    else:
        raise ValueError(
            f"Only one structure can be decoded at a time, got structure tokens of shape {structure_tokens.size()}"
        )
    _bos_eos_warn("Structure", structure_tokens[0], structure_tokenizer)

    decoder_output = structure_decoder.decode(structure_tokens)
    bb_coords: torch.Tensor = decoder_output["bb_pred"][
        0, 1:-1, ...
    ]  # Remove BOS and EOS tokens
    bb_coords = bb_coords.detach().cpu()

    plddt = decoder_output.get("plddt", None)
    if plddt is not None:
        plddt = plddt[0, 1:-1].detach().cpu()

    ptm = decoder_output.get("ptm", None)

    pae = decoder_output.get("predicted_aligned_error", None)

    chain = ProteinChain.from_backbone_atom_coordinates(bb_coords, sequence=sequence)
    chain = chain.infer_oxygen()
    return torch.tensor(chain.atom37_positions), plddt, ptm, pae


def decode_secondary_structure(
    secondary_structure_tokens: torch.Tensor, ss_tokenizer: SecondaryStructureTokenizer
) -> str:
    _bos_eos_warn("Secondary structure", secondary_structure_tokens, ss_tokenizer)
    secondary_structure_tokens = secondary_structure_tokens[1:-1]
    secondary_structure = ss_tokenizer.decode(secondary_structure_tokens)
    return secondary_structure


def decode_sasa(
    sasa_tokens: torch.Tensor, sasa_tokenizer: SASADiscretizingTokenizer
) -> list[float]:
    if sasa_tokens[0] != 0:
        raise ValueError("SASA does not start with 0 corresponding to BOS token")
    if sasa_tokens[-1] != 0:
        raise ValueError("SASA does not end with 0 corresponding to EOS token")
    sasa_tokens = sasa_tokens[1:-1]
    if sasa_tokens.dtype in [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.long,
    ]:
        # Decode if int
        # handles turning NaN's into None's
        sasa = sasa_tokenizer.decode_float(sasa_tokens)
    else:
        # If already float, just convert to list
        sasa = cast(list[float], maybe_list(sasa_tokens, convert_nan_to_none=True))

    return sasa


def decode_function_annotations(
    function_annotation_tokens: torch.Tensor,
    function_token_decoder: FunctionTokenDecoder,
    function_tokenizer: InterProQuantizedTokenizer,
    **kwargs,
) -> list[FunctionAnnotation]:
    # No need to check for BOS/EOS as function annotations are not affected

    function_annotations = decode_function_tokens(
        function_annotation_tokens,
        function_token_decoder=function_token_decoder,
        function_tokens_tokenizer=function_tokenizer,
        **kwargs,
    )
    return function_annotations


def decode_residue_annotations(
    residue_annotation_tokens: torch.Tensor,
    residue_annotation_decoder: ResidueAnnotationsTokenizer,
) -> list[FunctionAnnotation]:
    # No need to check for BOS/EOS as function annotations are not affected

    residue_annotations = decode_residue_annotation_tokens(
        residue_annotations_token_ids=residue_annotation_tokens,
        residue_annotations_tokenizer=residue_annotation_decoder,
    )
    return residue_annotations


def assemble_message(headers: Mapping[str, str], response: Response) -> dict[str, Any]:
    content_type = headers.get("Content-Type", "application/json")
    if content_type == api_constants.MIMETYPE_ES_PICKLE:
        return pickle.loads(response.content)
    elif "application/json" in content_type:
        # Can handle something like "application/json; charset=utf-8"
        return response.json()
    raise ValueError(f"Unknown Content-Type: {content_type}")
