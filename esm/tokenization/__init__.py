from dataclasses import dataclass
from typing import Protocol

from esm.utils.constants.models import ESM3_OPEN_SMALL, normalize_model_name

from .function_tokenizer import InterProQuantizedTokenizer
from .residue_tokenizer import ResidueAnnotationsTokenizer
from .sasa_tokenizer import SASADiscretizingTokenizer
from .sequence_tokenizer import EsmSequenceTokenizer
from .ss_tokenizer import SecondaryStructureTokenizer
from .structure_tokenizer import StructureTokenizer
from .tokenizer_base import EsmTokenizerBase


class TokenizerCollectionProtocol(Protocol):
    sequence: EsmSequenceTokenizer
    structure: StructureTokenizer
    secondary_structure: SecondaryStructureTokenizer
    sasa: SASADiscretizingTokenizer
    function: InterProQuantizedTokenizer
    residue_annotations: ResidueAnnotationsTokenizer


@dataclass
class TokenizerCollection:
    sequence: EsmSequenceTokenizer
    structure: StructureTokenizer
    secondary_structure: SecondaryStructureTokenizer
    sasa: SASADiscretizingTokenizer
    function: InterProQuantizedTokenizer
    residue_annotations: ResidueAnnotationsTokenizer


def get_esm3_model_tokenizers(model: str = ESM3_OPEN_SMALL) -> TokenizerCollection:
    if normalize_model_name(model) == ESM3_OPEN_SMALL:
        return TokenizerCollection(
            sequence=EsmSequenceTokenizer(),
            structure=StructureTokenizer(),
            secondary_structure=SecondaryStructureTokenizer(kind="ss8"),
            sasa=SASADiscretizingTokenizer(),
            function=InterProQuantizedTokenizer(),
            residue_annotations=ResidueAnnotationsTokenizer(),
        )
    else:
        raise ValueError(f"Unknown model: {model}")


def get_esmc_model_tokenizers() -> EsmSequenceTokenizer:
    return EsmSequenceTokenizer()


def get_invalid_tokenizer_ids(tokenizer: EsmTokenizerBase) -> list[int]:
    if isinstance(tokenizer, EsmSequenceTokenizer):
        return [
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.eos_token_id,
        ]
    else:
        return [
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        ]
