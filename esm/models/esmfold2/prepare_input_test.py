"""Tests for ESMFold2 input preparation (prepare_input)."""

import pytest
from rdkit import Chem

from esm.models.esmfold2.prepare_input import (
    build_chains_from_input,
    compute_token_bonds,
)
from esm.models.esmfold2.types import LigandInput, StructurePredictionInput


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",  # benzene: 6 atoms, 6 bonds
        # The drug-like ligand from the SMILES-vs-CCD issue.
        "COC1=CC=C(N2C3=C(C(C(N)=O)=N2)CCN(C4=CC=C(N5CCCCC5=O)C=C4)C3=O)C=C1",
    ],
)
def test_smiles_ligand_bonds_match_molecular_graph(smiles: str):
    """SMILES ligand bonds must match the molecular graph, not a clique (#313)."""
    spi = StructurePredictionInput(sequences=[LigandInput(id="B", smiles=smiles)])
    chains, tokens, atoms = build_chains_from_input(spi, seed=0)
    token_bonds = compute_token_bonds(tokens, atoms, spi, chains)

    mol = Chem.MolFromSmiles(smiles)
    assert len(tokens) == mol.GetNumAtoms()
    n_edges = int(token_bonds.sum().item()) // 2  # symmetric matrix
    assert n_edges == mol.GetNumBonds()
    assert n_edges < len(tokens) * (len(tokens) - 1) // 2  # not a clique
