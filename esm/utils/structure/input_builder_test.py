"""Round-trip tests for serialize/deserialize StructurePredictionInput."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from esm.utils.msa import MSA
from esm.utils.structure.input_builder import (
    CovalentBond,
    DistogramConditioning,
    DNAInput,
    LigandInput,
    Modification,
    PocketConditioning,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
    deserialize_structure_prediction_input,
    serialize_structure_prediction_input,
)


def _assert_roundtrip(spi: StructurePredictionInput) -> None:
    """``deserialize(serialize(spi))`` is equal to ``spi`` *after* a JSON pass.

    The JSON pass catches subtle representation issues (e.g. tuples
    becoming lists, numpy arrays becoming nested lists)."""
    data = json.loads(json.dumps(serialize_structure_prediction_input(spi)))
    restored = deserialize_structure_prediction_input(data)
    # Re-serialize and compare dict-form so np.ndarray fields compare structurally.
    assert serialize_structure_prediction_input(
        restored
    ) == serialize_structure_prediction_input(spi)


def test_roundtrip_minimal_protein() -> None:
    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="MAKL")])
    _assert_roundtrip(spi)


def test_roundtrip_protein_with_modifications() -> None:
    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(
                id="A",
                sequence="YGPKGPKGPKGKPGPDGDPGDPGDPGPKGPRG",
                modifications=[
                    Modification(position=3, ccd="HYP"),
                    Modification(position=9, ccd="HYP"),
                    Modification(position=15, ccd="HYP"),
                ],
            )
        ]
    )
    _assert_roundtrip(spi)
    restored = deserialize_structure_prediction_input(
        serialize_structure_prediction_input(spi)
    )
    assert isinstance(restored.sequences[0], ProteinInput)
    assert restored.sequences[0].modifications is not None
    assert len(restored.sequences[0].modifications) == 3
    assert restored.sequences[0].modifications[0].position == 3
    assert restored.sequences[0].modifications[0].ccd == "HYP"


def test_roundtrip_protein_with_msa() -> None:
    msa = MSA.from_sequences(["MAKL", "MAKM", "MAKL"])
    spi = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence="MAKL", msa=msa)]
    )
    _assert_roundtrip(spi)
    restored = deserialize_structure_prediction_input(
        serialize_structure_prediction_input(spi)
    )
    assert isinstance(restored.sequences[0], ProteinInput)
    assert isinstance(restored.sequences[0].msa, MSA)
    assert restored.sequences[0].msa.sequences == ["MAKL", "MAKM", "MAKL"]


def test_roundtrip_rna_dna_ligand() -> None:
    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence="MAKL"),
            RNAInput(
                id="B",
                sequence="ACGU",
                modifications=[Modification(position=2, ccd="PSU")],
            ),
            DNAInput(id="C", sequence="ACGT"),
            LigandInput(id="D", smiles="CCO"),
            LigandInput(id="E", ccd=["ATP"]),
        ]
    )
    _assert_roundtrip(spi)


def test_roundtrip_full_with_conditioning() -> None:
    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence="MAKL"),
            ProteinInput(id="B", sequence="MAKM"),
        ],
        pocket=PocketConditioning(binder_chain_id="A", contacts=[("B", 0), ("B", 2)]),
        distogram_conditioning=[
            DistogramConditioning(
                chain_id="A", distogram=np.arange(16).reshape(4, 4).astype(np.float32)
            )
        ],
        covalent_bonds=[
            CovalentBond(
                chain_id1="A",
                res_idx1=0,
                atom_idx1=1,
                chain_id2="B",
                res_idx2=2,
                atom_idx2=3,
            )
        ],
    )
    _assert_roundtrip(spi)
    restored = deserialize_structure_prediction_input(
        serialize_structure_prediction_input(spi)
    )
    assert restored.pocket is not None
    assert restored.pocket.binder_chain_id == "A"
    # Tuples survive the round-trip even though JSON only has lists.
    assert restored.pocket.contacts == [("B", 0), ("B", 2)]
    assert restored.distogram_conditioning is not None
    np.testing.assert_array_equal(
        restored.distogram_conditioning[0].distogram,
        np.arange(16).reshape(4, 4).astype(np.float32),
    )
    assert restored.covalent_bonds is not None
    assert spi.covalent_bonds is not None
    assert asdict(restored.covalent_bonds[0]) == asdict(spi.covalent_bonds[0])


def test_unsupported_sequence_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported sequence type"):
        deserialize_structure_prediction_input(
            {"sequences": [{"type": "lipid", "id": "X", "sequence": "..."}]}
        )
