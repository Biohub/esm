"""Tests for building a MolecularComplex from ESMFold2 features."""

from io import StringIO

import torch
from biotite.structure.io import pdbx

from esm.models.esmfold2.output import build_molecular_complex_from_features
from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
from esm.models.esmfold2.types import (
    LigandInput,
    ProteinInput,
    StructurePredictionInput,
)


def _complex(*sequences):
    features, chain_infos = prepare_esmfold2_input(
        StructurePredictionInput(sequences=list(sequences)), seed=0
    )
    atom_mask = features["atom_attention_mask"]
    return build_molecular_complex_from_features(
        coords=torch.zeros(atom_mask.shape[0], 3),
        plddt=torch.zeros(features["token_attention_mask"].shape[-1]),
        atom_mask=atom_mask,
        ref_element=features["ref_element"],
        ref_atom_name_chars=features["ref_atom_name_chars"],
        chain_infos=chain_infos,
        complex_id="test",
    )


def test_multi_component_ligand_returns_one_residue_per_component(ccd_pickle):
    complex_ = _complex(
        ProteinInput(id="A", sequence="AG"), LigandInput(id="B", ccd=["NAG", "BMA"])
    )
    assert list(complex_.sequence) == ["ALA", "GLY", "NAG", "BMA"]
    assert complex_.chain_id.tolist() == [0, 0, 1, 1]
    start = int(complex_.token_to_atoms[2, 0])
    assert not complex_.atom_hetero[:start].any()
    assert complex_.atom_hetero[start:].all()


def test_single_component_ligands_are_unchanged(ccd_pickle):
    complex_ = _complex(
        ProteinInput(id="A", sequence="AG"),
        LigandInput(id="B", ccd=["HEM"]),
        LigandInput(id="C", smiles="CCO"),
    )
    assert list(complex_.sequence) == ["ALA", "GLY", "HEM", "LIG"]
    sizes = [int(end - start) for start, end in complex_.token_to_atoms]
    assert sizes[0] == 5 and sizes[1] == 4 and sizes[3] == 3


def test_ligands_differing_after_first_component_are_distinct_entities(ccd_pickle):
    complex_ = _complex(
        LigandInput(id="A", ccd=["NAG", "NAG"]), LigandInput(id="B", ccd=["NAG", "BMA"])
    )
    cif = pdbx.CIFFile.read(StringIO(complex_.to_mmcif()))
    entity_ids = cif.block["struct_asym"]["entity_id"].as_array().tolist()
    assert entity_ids == ["1", "2"]
