"""The client-side encoder for an MSA carried compressed inside a request payload."""

import numpy as np

from esm.utils.compression import compress_state_dict, decompress_state_dict
from esm.utils.msa import MSA
from esm.utils.structure.input_builder import (
    ProteinInput,
    StructurePredictionInput,
    deserialize_structure_prediction_input,
    serialize_structure_prediction_input,
)


def test_a_compressed_msa_round_trips():
    """The decoder returns the state dict; rebuilding stays with the type, so this is
    the call a caller actually writes. `deletions` is the a3m insertion counts, and is
    lost silently if the encoder serializes only the sequences."""
    msa = MSA.from_state_dict(
        {"sequences": ["MNM", "MNQ"], "deletions": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
    )

    restored = MSA.from_state_dict(decompress_state_dict(compress_state_dict(msa)))

    assert restored.sequences == msa.sequences
    assert restored.deletions is not None
    np.testing.assert_array_equal(restored.deletions, msa.deletions)


def test_a_chain_msa_is_compressed_on_the_way_out_and_expanded_on_the_way_back():
    """Callers only ever hold the object -- the multimer path compresses per chain on
    serialize, so the wire form is the blob and deserialize has to invert it."""
    msa = MSA.from_sequences(["MNM", "MNQ"])
    built = serialize_structure_prediction_input(
        StructurePredictionInput(
            sequences=[ProteinInput(id="A", sequence="MNM", msa=msa)]
        )
    )

    assert built["sequences"][0]["msa"] == compress_state_dict(msa)

    restored = deserialize_structure_prediction_input(built)
    chain = restored.sequences[0]
    assert isinstance(chain, ProteinInput)
    assert isinstance(chain.msa, MSA)
    assert chain.msa.sequences == ["MNM", "MNQ"]


def test_the_encoding_is_generic_over_the_state_dict_convention():
    class Custom:
        def state_dict(self, json_serializable: bool = False) -> dict:
            return {"values": [1, 2, 3], "serializable": json_serializable}

    assert decompress_state_dict(compress_state_dict(Custom())) == {
        "values": [1, 2, 3],
        "serializable": True,
    }
