"""Batched forward passes with a tiny random ESM3."""

import pytest
import torch

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProteinTensor
from esm.tokenization import (
    InterProQuantizedTokenizer,
    ResidueAnnotationsTokenizer,
    SASADiscretizingTokenizer,
    SecondaryStructureTokenizer,
    StructureTokenizer,
    TokenizerCollection,
)
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils import encoding
from esm.utils.generation import _batch_forward, _stack_protein_tensors


@pytest.fixture
def tokenizers(tmp_path) -> TokenizerCollection:
    # The default residue-annotation vocabulary is fetched from the Hub; a one-label
    # CSV keeps these tests offline.
    csv = tmp_path / "residue_annotations.csv"
    csv.write_text("label,label_clean,count\nsite,site,1\n")
    return TokenizerCollection(
        sequence=EsmSequenceTokenizer(),
        structure=StructureTokenizer(),
        secondary_structure=SecondaryStructureTokenizer(kind="ss8"),
        sasa=SASADiscretizingTokenizer(),
        function=InterProQuantizedTokenizer(),
        residue_annotations=ResidueAnnotationsTokenizer(csv_path=str(csv)),
    )


@pytest.fixture
def model(tokenizers) -> ESM3:
    torch.manual_seed(0)
    return ESM3(
        d_model=32,
        n_heads=2,
        v_heads=2,
        n_layers=2,
        structure_encoder_fn=lambda _: None,  # ty:ignore[invalid-argument-type]
        structure_decoder_fn=lambda _: None,  # ty:ignore[invalid-argument-type]
        function_decoder_fn=lambda _: None,  # ty:ignore[invalid-argument-type]
        tokenizers=tokenizers,
    ).eval()


def _prompt(model: ESM3, sequence: str, with_coords: bool) -> ESMProteinTensor:
    tokens = encoding.tokenize_sequence(sequence, model.tokenizers.sequence)
    coordinates = None
    if with_coords:
        generator = torch.Generator().manual_seed(len(sequence))
        coordinates = 10 * torch.randn(len(tokens), 37, 3, generator=generator)
        coordinates[[0, -1]] = torch.inf  # BOS / EOS have no structure
    return ESMProteinTensor(sequence=tokens, coordinates=coordinates)


@pytest.mark.parametrize("with_coords", [False, True], ids=["sequence", "structure"])
def test_batched_logits_do_not_depend_on_padding(model, with_coords):
    short = _prompt(model, "MKTAYIAK", with_coords)
    long_ = _prompt(model, "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGT", with_coords)

    def logits(prompts):
        batch = _stack_protein_tensors(
            prompts, [len(p) for p in prompts], model.tokenizers, "cpu"
        )
        return _batch_forward(model, batch).logits

    alone = logits([short])
    padded = logits([short, long_])

    n = len(short)
    for track in ("sequence", "structure", "secondary_structure", "sasa", "function"):
        torch.testing.assert_close(
            getattr(padded, track)[0, :n], getattr(alone, track)[0], msg=track
        )
