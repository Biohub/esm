"""Tests for ESMFold2 input preparation (prepare_input)."""

import pytest
from rdkit import Chem

from esm.models.esmfold2.prepare_input import (
    _entity_key,
    build_chains_from_input,
    compute_token_bonds,
)
from esm.models.esmfold2.types import (
    LigandInput,
    Modification,
    ProteinInput,
    StructurePredictionInput,
)


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


def test_entity_key_separates_chemically_distinct_chains():
    """Same sequence but different chemistry must not collapse to one entity.

    Entities that merge also become sym_id copies of each other, i.e. claim a
    symmetry between two chains that are not copies.
    """
    plain = ProteinInput(id="A", sequence="AAAGGG")
    modified = ProteinInput(
        id="B", sequence="AAAGGG", modifications=[Modification(position=1, ccd="MSE")]
    )

    assert _entity_key(plain) == _entity_key(ProteinInput(id="D", sequence="AAAGGG"))
    assert _entity_key(plain) != _entity_key(modified)


def test_entity_key_ignores_smiles_when_ccd_is_present():
    """A redundant smiles must not split one ligand entity into two.

    Tokenization takes the ccd branch and warns that the smiles is unused, so a
    key that reads both fields would call two identically tokenized chains
    different entities.
    """
    assert _entity_key(LigandInput(id="A", ccd=["HEM"])) == _entity_key(
        LigandInput(id="B", ccd=["HEM"], smiles="CCO")
    )
    assert _entity_key(LigandInput(id="A", ccd=["HEM"])) != _entity_key(
        LigandInput(id="B", ccd=["NAG"])
    )


def test_build_chains_assigns_entity_and_sym_ids_to_repeated_ligands():
    """Repeated copies are one entity numbered by sym_id, through the real builder.

    sym_id is what the relative position encoding turns into its cross-chain
    feature, so entity dedup is only correct if the sym_id it drives is too.
    """
    spi = StructurePredictionInput(
        sequences=[
            LigandInput(id="A", smiles="CCO"),
            LigandInput(id="B", smiles="c1ccccc1"),
            LigandInput(id="C", smiles="CCO"),
        ]
    )
    chains, _, _ = build_chains_from_input(spi, seed=0)

    assert [c.entity_id for c in chains] == [0, 1, 0]
    assert [c.sym_id for c in chains] == [0, 0, 1]


def test_build_chains_numbers_copies_across_both_id_forms():
    """One entry with several ids and a duplicate entry must keep counting.

    ``id=['A', 'B']`` expands to two chains inside one entry while chain C is a
    separate entry for the same entity, so the three copies are sym_id 0, 1, 2.
    """
    spi = StructurePredictionInput(
        sequences=[
            LigandInput(id=["A", "B"], smiles="CCO"),
            LigandInput(id="C", smiles="CCO"),
        ]
    )
    chains, _, _ = build_chains_from_input(spi, seed=0)

    assert [c.chain_id for c in chains] == ["A", "B", "C"]
    assert [c.entity_id for c in chains] == [0, 0, 0]
    assert [c.sym_id for c in chains] == [0, 1, 2]
