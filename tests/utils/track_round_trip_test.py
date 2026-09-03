"""Decode -> encode round trips must return the tokens they came from."""

import pytest
import torch

from esm.tokenization.sasa_tokenizer import SASADiscretizingTokenizer
from esm.tokenization.ss_tokenizer import SecondaryStructureTokenizer
from esm.utils.decoding import decode_sasa, decode_secondary_structure
from esm.utils.encoding import tokenize_sasa, tokenize_secondary_structure


def test_sasa_decode_encode_round_trip():
    tokenizer = SASADiscretizingTokenizer()
    # BOS, <pad>, a bin, <motif>, <unk>, top bin, EOS
    tokens = torch.tensor([0, 0, 10, 1, 2, 18, 0])

    values = decode_sasa(tokens, tokenizer)
    round_tripped = tokenize_sasa(values, tokenizer, add_special_tokens=True)

    assert torch.equal(round_tripped, tokens)


def test_sasa_decode_float_specials():
    tokenizer = SASADiscretizingTokenizer()

    values = decode_sasa(torch.tensor([0, 0, 1, 2, 10, 0]), tokenizer)

    # <pad> is None because that is what tokenize_sasa maps back to the mask token; the
    # other specials have no midpoint, so they come back as their vocab string.
    assert values == [None, "<motif>", "<unk>", pytest.approx(46.75)]


def test_sasa_tokenize_rejects_nan():
    tokenizer = SASADiscretizingTokenizer()

    with pytest.raises(ValueError, match="NaN"):
        tokenize_sasa([1.0, float("nan")], tokenizer, add_special_tokens=False)


def test_secondary_structure_decode_encode_round_trip():
    tokenizer = SecondaryStructureTokenizer()
    # BOS, H, <pad> (chainbreak), E, EOS
    tokens = torch.tensor([0, 4, 0, 7, 0])

    decoded = decode_secondary_structure(tokens, tokenizer)
    round_tripped = tokenize_secondary_structure(
        decoded, tokenizer, add_special_tokens=True
    )

    assert torch.equal(round_tripped, tokens)


def test_secondary_structure_decode_is_one_char_per_residue():
    tokenizer = SecondaryStructureTokenizer()
    tokens = torch.tensor([0, 4, 0, 7, 0])

    decoded = decode_secondary_structure(tokens, tokenizer)

    # A literal "<pad>" here would measure 5 residues instead of 1.
    assert decoded == "H_E"
