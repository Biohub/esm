"""Tests for MolecularComplex CIF roundtrip: chain separation and entity info.

Verifies that from_mmcif -> to_blob -> from_blob -> to_mmcif preserves:
1. Ligand chain separation (label_asym_id, not auth_asym_id)
2. Entity categories (polymer vs non-polymer)
3. Correct atom counts after roundtrip
"""

import numpy as np
import pytest

from esm.utils import residue_constants
from esm.utils.structure.molecular_complex import MolecularComplex

# Minimal CIF with protein (chain A) + ligand (chain B in label_asym_id, chain A in auth_asym_id)
# This is the standard PDB convention that was previously broken.
# Includes all columns biotite needs: pdbx_PDB_ins_code, B_iso_or_equiv, auth_comp_id, auth_atom_id
PROTEIN_LIGAND_CIF = """\
data_test
#
loop_
_entity.id
_entity.type
_entity.pdbx_description
1 polymer 'test protein'
2 non-polymer 'test ligand'
#
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.B_iso_or_equiv
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
_atom_site.id
ATOM   N  N    . ALA A 1 1  ? 1  ALA A N    50.0 1.0 2.0 3.0  1 1
ATOM   C  CA   . ALA A 1 1  ? 1  ALA A CA   50.0 2.0 3.0 4.0  1 2
ATOM   C  C    . ALA A 1 1  ? 1  ALA A C    50.0 3.0 4.0 5.0  1 3
ATOM   O  O    . ALA A 1 1  ? 1  ALA A O    50.0 4.0 5.0 6.0  1 4
ATOM   C  CB   . ALA A 1 1  ? 1  ALA A CB   50.0 5.0 6.0 7.0  1 5
ATOM   N  N    . GLY A 1 2  ? 2  GLY A N    50.0 6.0 7.0 8.0  1 6
ATOM   C  CA   . GLY A 1 2  ? 2  GLY A CA   50.0 7.0 8.0 9.0  1 7
ATOM   C  C    . GLY A 1 2  ? 2  GLY A C    50.0 8.0 9.0 10.0 1 8
ATOM   O  O    . GLY A 1 2  ? 2  GLY A O    50.0 9.0 10.0 11.0 1 9
HETATM C  C1   . LIG B 2 .  ? 101 LIG A C1  50.0 10.0 11.0 12.0 1 10
HETATM C  C2   . LIG B 2 .  ? 101 LIG A C2  50.0 11.0 12.0 13.0 1 11
HETATM O  O1   . LIG B 2 .  ? 101 LIG A O1  50.0 12.0 13.0 14.0 1 12
#
"""

# CIF with two protein chains (no ligand)
TWO_CHAIN_PROTEIN_CIF = """\
data_test2
#
loop_
_entity.id
_entity.type
1 polymer
2 polymer
#
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.B_iso_or_equiv
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
_atom_site.id
ATOM N  N  . ALA A 1 1 ? 1 ALA A N  50.0 1.0 2.0 3.0  1 1
ATOM C  CA . ALA A 1 1 ? 1 ALA A CA 50.0 2.0 3.0 4.0  1 2
ATOM C  C  . ALA A 1 1 ? 1 ALA A C  50.0 3.0 4.0 5.0  1 3
ATOM O  O  . ALA A 1 1 ? 1 ALA A O  50.0 4.0 5.0 6.0  1 4
ATOM N  N  . GLY B 2 1 ? 1 GLY B N  50.0 5.0 6.0 7.0  1 5
ATOM C  CA . GLY B 2 1 ? 1 GLY B CA 50.0 6.0 7.0 8.0  1 6
ATOM C  C  . GLY B 2 1 ? 1 GLY B C  50.0 7.0 8.0 9.0  1 7
ATOM O  O  . GLY B 2 1 ? 1 GLY B O  50.0 8.0 9.0 10.0 1 8
#
"""


def test_from_mmcif_ligand_on_separate_chain():
    """Ligands should be on a different chain than the protein."""
    mc = MolecularComplex.from_mmcif(PROTEIN_LIGAND_CIF)

    # Should have 3 tokens: ALA, GLY (protein), LIG (ligand)
    assert len(mc.sequence) == 3
    assert mc.sequence[0] == "ALA"
    assert mc.sequence[1] == "GLY"
    assert mc.sequence[2] == "LIG"

    # Protein tokens on chain 0 (A), ligand on chain 1 (B)
    assert int(mc.chain_id[0]) == int(mc.chain_id[1])  # ALA and GLY same chain
    assert int(mc.chain_id[2]) != int(mc.chain_id[0])  # LIG different chain

    # Chain lookup should have both chains
    assert len(mc.metadata.chain_lookup) >= 2
    protein_chain = mc.metadata.chain_lookup[int(mc.chain_id[0])]
    ligand_chain = mc.metadata.chain_lookup[int(mc.chain_id[2])]
    assert protein_chain == "A"
    assert ligand_chain == "B"


def test_to_mmcif_entity_categories():
    """to_mmcif should emit _entity category with polymer/non-polymer types."""
    mc = MolecularComplex.from_mmcif(PROTEIN_LIGAND_CIF)
    cif_out = mc.to_mmcif()

    assert "_entity.id" in cif_out
    assert "_entity.type" in cif_out
    assert "polymer" in cif_out.lower()
    assert "non-polymer" in cif_out.lower()


def test_blob_roundtrip_preserves_chain_separation():
    """from_mmcif -> to_blob -> from_blob -> to_mmcif should preserve ligand chains."""
    mc = MolecularComplex.from_mmcif(PROTEIN_LIGAND_CIF)
    blob = mc.to_blob()
    mc2 = MolecularComplex.from_blob(blob)

    # Same sequence
    assert mc2.sequence == mc.sequence

    # Same chain separation
    assert int(mc2.chain_id[2]) != int(mc2.chain_id[0])

    # Roundtripped CIF should still have entity info
    cif_out = mc2.to_mmcif()
    assert "non-polymer" in cif_out.lower()


def test_blob_roundtrip_preserves_atom_counts():
    """Atom counts should be preserved through blob roundtrip."""
    mc = MolecularComplex.from_mmcif(PROTEIN_LIGAND_CIF)
    blob = mc.to_blob()
    mc2 = MolecularComplex.from_blob(blob)

    assert len(mc2.atom_positions) == len(mc.atom_positions)
    np.testing.assert_allclose(mc2.atom_positions, mc.atom_positions, atol=0.1)


def test_two_chain_protein():
    """Two-chain protein should preserve chain IDs."""
    mc = MolecularComplex.from_mmcif(TWO_CHAIN_PROTEIN_CIF)

    assert len(mc.sequence) == 2
    assert int(mc.chain_id[0]) != int(mc.chain_id[1])
    assert mc.metadata.chain_lookup[int(mc.chain_id[0])] == "A"
    assert mc.metadata.chain_lookup[int(mc.chain_id[1])] == "B"


def test_to_mmcif_entity_types_correct():
    """Entity types should match token types: protein=polymer, ligand=non-polymer."""
    mc = MolecularComplex.from_mmcif(PROTEIN_LIGAND_CIF)
    cif_out = mc.to_mmcif()

    # Parse entity lines
    lines = cif_out.split("\n")
    entity_lines = []
    in_entity = False
    for line in lines:
        if "_entity.id" in line:
            in_entity = True
            continue
        if in_entity and line.startswith("#"):
            break
        if (
            in_entity
            and line.strip()
            and not line.startswith("_")
            and not line.startswith("loop")
        ):
            entity_lines.append(line.strip())

    # Should have at least 2 entities
    assert len(entity_lines) >= 2

    # First entity should be polymer (protein)
    assert "polymer" in entity_lines[0].lower()
    assert "non-polymer" not in entity_lines[0].lower()

    # Second entity should be non-polymer (ligand)
    assert "non-polymer" in entity_lines[1].lower()


@pytest.mark.parametrize(
    "cif_fixture",
    [PROTEIN_LIGAND_CIF, TWO_CHAIN_PROTEIN_CIF],
    ids=["protein_ligand", "two_chain_protein"],
)
def test_from_mmcif_no_token_on_wrong_chain(cif_fixture: str):
    """Non-protein tokens should never share a chain with protein tokens."""
    mc = MolecularComplex.from_mmcif(cif_fixture)

    protein_chains: set[int] = set()
    non_protein_chains: set[int] = set()

    for token, chain in zip(mc.sequence, mc.chain_id):
        if token in residue_constants.restype_3to1:
            protein_chains.add(int(chain))
        else:
            non_protein_chains.add(int(chain))

    # Non-protein tokens should not be on any protein chain
    overlap = protein_chains & non_protein_chains
    assert (
        overlap == set()
    ), f"Non-protein tokens share chain(s) {overlap} with protein tokens"


def _make_four_chain_protein_complex():
    """Build a 4-chain heteromer (A: ALA-LEU, B: GLY-ARG, C: MET, D: PHE-TRP).

    Uses distinct sidechain atoms per residue so that an off-by-one in
    sequence indexing (skipping '|' chain breaks) is detectable:
    with 3 breaks, residues after chain A would be shifted by 1, 2, or 3.
    """
    from esm.utils.structure.protein_chain import ProteinChain
    from esm.utils.structure.protein_complex import ProteinComplex

    rng = np.random.default_rng(42)

    def _chain(seq, cid, eid, masks):
        n = len(seq)
        coords = rng.uniform(-10, 10, (n, 37, 3)).astype(np.float32)
        mask = np.zeros((n, 37), dtype=bool)
        for i, m in enumerate(masks):
            mask[i, m] = True
        return ProteinChain(
            id="test",
            sequence=seq,
            chain_id=cid,
            entity_id=eid,
            residue_index=np.arange(n),
            insertion_code=np.full(n, "", dtype="<U4"),
            atom37_positions=coords,
            atom37_mask=mask,
            confidence=np.ones(n),
        )

    chains = [
        _chain(
            "AL",
            "A",
            1,
            [
                [0, 1, 2, 3, 4],  # ALA
                [0, 1, 2, 3, 4, 5, 12, 13],  # LEU
            ],
        ),
        _chain(
            "GR",
            "B",
            2,
            [
                [0, 1, 2, 3],  # GLY
                [0, 1, 2, 3, 4, 5, 11, 23, 29, 30, 32],  # ARG
            ],
        ),
        _chain(
            "M",
            "C",
            3,
            [
                [0, 1, 2, 3, 4, 5, 18, 19]  # MET
            ],
        ),
        _chain(
            "FW",
            "D",
            4,
            [
                [0, 1, 2, 3, 4, 5, 12, 13, 20, 21, 32],  # PHE
                [0, 1, 2, 3, 4, 5, 12, 13, 21, 22, 24, 28, 33, 34],  # TRP
            ],
        ),
    ]
    return ProteinComplex.from_chains(chains)


def test_protein_complex_roundtrip_preserves_atom37_mask():
    """ProteinComplex -> MolecularComplex -> blob -> MolecularComplex -> ProteinComplex
    must preserve atom37_mask exactly, including across chain breaks.

    Regression tests two bugs:
    1. from_protein_complex used a separate residue_idx counter instead of the
       sequence position to index atom37_positions. After the '|' chain break,
       every residue's atoms were read from the wrong position (off-by-one).
    2. to_protein_complex only placed atoms matching canonical residue_atoms,
       silently dropping atoms at non-canonical atom37 indices.
    """
    pc = _make_four_chain_protein_complex()

    # Roundtrip: PC -> MC -> blob -> MC -> PC
    mc = MolecularComplex.from_protein_complex(pc)
    blob = mc.to_blob()
    mc2 = MolecularComplex.from_blob(blob)
    pc2 = mc2.to_protein_complex()

    assert (
        pc2.sequence == pc.sequence
    ), f"Sequence changed: {pc.sequence!r} -> {pc2.sequence!r}"

    # Check per-chain, per-residue atom mask AND coordinate preservation
    for chain_orig, chain_rt in zip(pc.chain_iter(), pc2.chain_iter()):
        np.testing.assert_array_equal(
            chain_orig.atom37_mask,
            chain_rt.atom37_mask,
            err_msg=f"atom37_mask not preserved for chain {chain_orig.chain_id}",
        )
        # Coordinates must match where mask is True
        for r in range(len(chain_orig.sequence)):
            for a in range(37):
                if chain_orig.atom37_mask[r, a]:
                    np.testing.assert_allclose(
                        chain_orig.atom37_positions[r, a],
                        chain_rt.atom37_positions[r, a],
                        atol=0.01,
                        err_msg=f"Coords differ chain={chain_orig.chain_id} res={r} atom={a}",
                    )
