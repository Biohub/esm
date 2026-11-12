"""ESM3 forward and batched inference on a tiny randomly initialised model."""

import pytest
import torch
import torch.nn.functional as F

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
from esm.utils.constants import esm3 as C
from esm.utils.generation import _batch_forward, _stack_protein_tensors

OUTPUTS = (
    "sequence_logits",
    "structure_logits",
    "secondary_structure_logits",
    "sasa_logits",
    "function_logits",
    "residue_logits",
    "embeddings",
)


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


def _padded_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """A (1, 12) row of residues, and the same row right-padded to length 40."""
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(
        C.SEQUENCE_STANDARD_AA_MIN_TOKEN,
        C.SEQUENCE_STANDARD_AA_MAX_TOKEN,
        (1, 12),
        generator=generator,
    )
    return tokens, F.pad(tokens, (0, 28), value=C.SEQUENCE_PAD_TOKEN)


@torch.no_grad()
def test_forward_output_does_not_depend_on_padding(model):
    tokens, padded = _padded_pair()

    alone = model(sequence_tokens=tokens)
    with_padding = model(sequence_tokens=padded)

    for name in OUTPUTS:
        torch.testing.assert_close(
            getattr(with_padding, name)[:, : tokens.shape[1]],
            getattr(alone, name),
            msg=name,
        )


@torch.no_grad()
def test_forward_uses_an_explicit_sequence_id_as_given(model):
    tokens, padded = _padded_pair()
    n = tokens.shape[1]
    default = model(sequence_tokens=padded).sequence_logits[:, :n]

    # Passing the padding mask explicitly is the same as the default...
    is_real = (padded != C.SEQUENCE_PAD_TOKEN).long()
    explicit = model(sequence_tokens=padded, sequence_id=is_real)
    torch.testing.assert_close(explicit.sequence_logits[:, :n], default)

    # ...and a caller's own sequence_id is not replaced by it.
    one_segment = model(sequence_tokens=padded, sequence_id=torch.zeros_like(padded))
    assert not torch.allclose(one_segment.sequence_logits[:, :n], default)


def _prompt(model: ESM3, kind: str, length: int) -> ESMProteinTensor:
    generator = torch.Generator().manual_seed(length)
    sequence = coordinates = None
    if kind != "coordinates only":
        residues = "".join("ACDEFGHIKLMNPQRSTVWY"[i % 20] for i in range(length))
        sequence = encoding.tokenize_sequence(residues, model.tokenizers.sequence)
    if kind != "sequence only":
        coordinates = 10 * torch.randn(length + 2, 37, 3, generator=generator)
        coordinates[[0, -1]] = torch.inf  # BOS / EOS have no structure
    return ESMProteinTensor(sequence=sequence, coordinates=coordinates)


@pytest.mark.parametrize(
    "kind", ["sequence only", "sequence and coordinates", "coordinates only"]
)
def test_batched_logits_do_not_depend_on_padding(model, kind):
    short, long_ = _prompt(model, kind, 8), _prompt(model, kind, 44)

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
