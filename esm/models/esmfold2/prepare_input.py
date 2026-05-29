"""Prepare ESMFold2 model inputs from sequence-level StructurePredictionInput.

This module converts StructurePredictionInput (protein/DNA/RNA/ligand sequences)
into the tensor dict expected by the ESMFold2 model forward pass.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from esm.models.esmfold2.conformers import (
    get_ccd_leaving_atoms,
    get_idealized_atom_pos,
    get_ligand_ccd_atoms_with_charges,
    get_ligand_ccd_bonds,
    get_ligand_idealized_atom_pos,
)
from esm.models.esmfold2.constants import (
    CHARGED_ATOMS,
    DNA_1TO3,
    DNA_BACKBONE_ATOMS,
    DNA_HEAVY_ATOMS,
    DNA_RESIDUE_TO_RES_TYPE,
    DNA_RNA_LIGAND_INPUT_ID,
    DNA_UNK_RES_TYPE,
    ELEMENT_TO_ATOMIC_NUM,
    ESM_PROTEIN_VOCAB,
    MOL_TYPE_DNA,
    MOL_TYPE_NONPOLYMER,
    MOL_TYPE_PROTEIN,
    MOL_TYPE_RNA,
    MSA_GAP_TOKEN_ID,
    PROTEIN_1TO3,
    PROTEIN_3TO1,
    PROTEIN_HEAVY_ATOMS,
    PROTEIN_RESIDUE_TO_RES_TYPE,
    PROTEIN_UNK_RES_TYPE,
    RNA_1TO3,
    RNA_BACKBONE_ATOMS,
    RNA_HEAVY_ATOMS,
    RNA_RESIDUE_TO_RES_TYPE,
    RNA_UNK_RES_TYPE,
)
from esm.models.esmfold2.types import (
    MSA,
    DNAInput,
    LigandInput,
    Modification,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)

# =============================================================================
# Lightweight data model
# =============================================================================

_ZERO_POS = np.array([0.0, 0.0, 0.0], dtype=np.float32)


@dataclass
class AtomInfo:
    name: str
    element: str
    charge: int
    ref_pos: np.ndarray  # Idealized position from CCD [3]
    pos: np.ndarray  # Experimental position [3] (zeros for inference)
    token_index: int = -1
    atom_index: int = -1
    space_uid: int = -1
    is_valid: bool = True


@dataclass
class TokenInfo:
    token_index: int
    residue_index: int  # Within chain (0-based)
    residue_name: str  # 3-letter code
    mol_type: int  # 0=protein, 1=DNA, 2=RNA, 3=nonpolymer
    res_type: int  # Residue type index (2-32)
    input_id: int  # ESM vocab ID
    asym_id: int
    sym_id: int
    entity_id: int
    atom_start: int  # Index into atoms list
    atom_count: int


@dataclass
class ChainInfo:
    chain_id: str
    asym_id: int
    entity_id: int
    sym_id: int
    mol_type: int
    tokens: list[TokenInfo] = field(default_factory=list)
    # (atom_name1, atom_name2) bonds for SMILES ligands, which have no CCD entry.
    ligand_bonds: list[tuple[str, str]] = field(default_factory=list)


# =============================================================================
# Helper functions
# =============================================================================

# Caches for hot-path functions
_ENCODE_ATOM_NAME_CACHE: dict[str, list[int]] = {}
_ELEMENT_ATOMIC_NUM_CACHE: dict[str, int] = {}


def encode_atom_name(name: str) -> list[int]:
    """Encode atom name as 4 character indices (offset by 32 from ASCII)."""
    if name in _ENCODE_ATOM_NAME_CACHE:
        return _ENCODE_ATOM_NAME_CACHE[name]
    padded = name.ljust(4)[:4]
    result = [ord(c) - 32 if c != " " else 0 for c in padded]
    _ENCODE_ATOM_NAME_CACHE[name] = result
    return result


def get_element_atomic_num(element: str) -> int:
    """Get atomic number for an element symbol."""
    if element in _ELEMENT_ATOMIC_NUM_CACHE:
        return _ELEMENT_ATOMIC_NUM_CACHE[element]
    result = ELEMENT_TO_ATOMIC_NUM.get(element.upper(), 0)
    _ELEMENT_ATOMIC_NUM_CACHE[element] = result
    return result


def _infer_element(atom_name: str) -> str:
    """Infer element from atom name."""
    name = atom_name.strip()
    if not name:
        return "C"
    if name[0].isdigit():
        return name[1] if len(name) > 1 else "H"
    if len(name) == 2 and name in (
        "FE",
        "ZN",
        "MG",
        "MN",
        "CO",
        "NI",
        "CU",
        "SE",
        "BR",
    ):
        return name
    return name[0]


def _compute_res_type(name: str, mol_type: int) -> int:
    """Compute residue type index from residue name and mol_type."""
    if mol_type == MOL_TYPE_PROTEIN:
        return PROTEIN_RESIDUE_TO_RES_TYPE.get(name, PROTEIN_UNK_RES_TYPE)
    elif mol_type == MOL_TYPE_DNA:
        if name in DNA_RESIDUE_TO_RES_TYPE:
            return DNA_RESIDUE_TO_RES_TYPE[name]
        if name in RNA_RESIDUE_TO_RES_TYPE:
            return RNA_RESIDUE_TO_RES_TYPE[name]
        return DNA_UNK_RES_TYPE
    elif mol_type == MOL_TYPE_RNA:
        if name in RNA_RESIDUE_TO_RES_TYPE:
            return RNA_RESIDUE_TO_RES_TYPE[name]
        if name in DNA_RESIDUE_TO_RES_TYPE:
            return DNA_RESIDUE_TO_RES_TYPE[name]
        return RNA_UNK_RES_TYPE
    return PROTEIN_UNK_RES_TYPE


def _compute_esm_input_id(name: str, mol_type: int) -> int:
    """Compute ESM vocabulary input ID."""
    if mol_type == MOL_TYPE_PROTEIN:
        letter = PROTEIN_3TO1.get(name)
        if letter is None:
            return DNA_RNA_LIGAND_INPUT_ID
        return ESM_PROTEIN_VOCAB.get(letter, ESM_PROTEIN_VOCAB["X"])
    return DNA_RNA_LIGAND_INPUT_ID


# =============================================================================
# Tokenization functions — build tokens and atoms from sequences
# =============================================================================


def tokenize_protein(
    sequence: str,
    modifications: list[Modification] | None,
    entity_id: int,
    asym_id: int,
    sym_id: int,
    token_offset: int,
    atom_offset: int,
    space_uid_offset: int,
) -> tuple[list[TokenInfo], list[AtomInfo]]:
    """Tokenize a protein sequence into tokens and atoms.

    Standard residues produce 1 token with all heavy atoms.
    Modified residues (from modifications) are atom-tokenized (1 token per atom).
    """
    tokens: list[TokenInfo] = []
    atoms: list[AtomInfo] = []

    # Build 3-letter sequence, applying modifications
    seq_3letter = [PROTEIN_1TO3.get(c, "UNK") for c in sequence]
    modified_positions: set[int] = set()
    if modifications:
        for mod in modifications:
            seq_3letter[mod.position] = mod.ccd
            modified_positions.add(mod.position)

    token_idx = token_offset
    atom_idx = atom_offset
    space_uid = space_uid_offset

    for res_idx, res_name in enumerate(seq_3letter):
        # MSE → MET for atom lookup
        res_corrected = "MET" if res_name == "MSE" else res_name
        is_modified = res_idx in modified_positions

        # Check if standard residue (has predefined atom list)
        if not is_modified and res_corrected in PROTEIN_HEAVY_ATOMS:
            # Standard residue: 1 token, multiple atoms
            atom_names = PROTEIN_HEAVY_ATOMS[res_corrected]
            res_type = _compute_res_type(res_corrected, MOL_TYPE_PROTEIN)
            input_id = _compute_esm_input_id(res_corrected, MOL_TYPE_PROTEIN)

            atom_start = atom_idx
            for a_name in atom_names:
                ref_pos = get_idealized_atom_pos(res_type, a_name)
                atoms.append(
                    AtomInfo(
                        name=a_name,
                        element=_infer_element(a_name),
                        charge=CHARGED_ATOMS.get((res_corrected, a_name), 0),
                        ref_pos=ref_pos.copy()
                        if ref_pos is not None
                        else _ZERO_POS.copy(),
                        pos=_ZERO_POS.copy(),
                        token_index=token_idx,
                        atom_index=atom_idx,
                        space_uid=space_uid,
                    )
                )
                atom_idx += 1

            tokens.append(
                TokenInfo(
                    token_index=token_idx,
                    residue_index=res_idx,
                    residue_name=res_corrected,
                    mol_type=MOL_TYPE_PROTEIN,
                    res_type=res_type,
                    input_id=input_id,
                    asym_id=asym_id,
                    sym_id=sym_id,
                    entity_id=entity_id,
                    atom_start=atom_start,
                    atom_count=len(atom_names),
                )
            )
            token_idx += 1
            space_uid += 1

        else:
            # Modified or unknown residue: atom-tokenized
            ccd_atoms = get_ligand_ccd_atoms_with_charges(res_name)
            if ccd_atoms is None:
                # Fallback: backbone only
                ccd_atoms = [
                    (_infer_element(n), _infer_element(n), 0)
                    for n in ["N", "CA", "C", "O"]
                ]

            # Filter leaving atoms if not terminal
            is_terminal = res_idx == len(seq_3letter) - 1
            leaving_atoms = set() if is_terminal else get_ccd_leaving_atoms(res_name)
            kept_atoms = [a for a in ccd_atoms if a[0] not in leaving_atoms]
            # Single-atom residues (e.g. NH2 cap): the local frame is
            # ill-defined with one atom; place at origin.
            single_atom_residue = len(kept_atoms) == 1

            for a_name, a_element, a_charge in kept_atoms:
                ref_pos = get_ligand_idealized_atom_pos(res_name, a_name)
                atoms.append(
                    AtomInfo(
                        name=a_name,
                        element=a_element,
                        charge=a_charge,
                        ref_pos=_ZERO_POS.copy()
                        if single_atom_residue
                        else (
                            ref_pos.copy() if ref_pos is not None else _ZERO_POS.copy()
                        ),
                        pos=_ZERO_POS.copy(),
                        token_index=token_idx,
                        atom_index=atom_idx,
                        space_uid=space_uid,
                    )
                )
                tokens.append(
                    TokenInfo(
                        token_index=token_idx,
                        residue_index=res_idx,
                        residue_name=res_name,
                        mol_type=MOL_TYPE_PROTEIN,
                        res_type=PROTEIN_UNK_RES_TYPE,
                        input_id=DNA_RNA_LIGAND_INPUT_ID,
                        asym_id=asym_id,
                        sym_id=sym_id,
                        entity_id=entity_id,
                        atom_start=atom_idx,
                        atom_count=1,
                    )
                )
                token_idx += 1
                atom_idx += 1

            space_uid += 1

    return tokens, atoms


def tokenize_nucleotide(
    sequence: str,
    modifications: list[Modification] | None,
    mol_type: int,
    entity_id: int,
    asym_id: int,
    sym_id: int,
    token_offset: int,
    atom_offset: int,
    space_uid_offset: int,
) -> tuple[list[TokenInfo], list[AtomInfo]]:
    """Tokenize a DNA or RNA sequence into tokens and atoms."""
    tokens: list[TokenInfo] = []
    atoms: list[AtomInfo] = []

    letter_to_3 = DNA_1TO3 if mol_type == MOL_TYPE_DNA else RNA_1TO3
    heavy_atoms = DNA_HEAVY_ATOMS if mol_type == MOL_TYPE_DNA else RNA_HEAVY_ATOMS
    backbone_atoms = (
        DNA_BACKBONE_ATOMS if mol_type == MOL_TYPE_DNA else RNA_BACKBONE_ATOMS
    )
    unk_res_type = DNA_UNK_RES_TYPE if mol_type == MOL_TYPE_DNA else RNA_UNK_RES_TYPE

    seq_3letter = [letter_to_3.get(c, "UNK") for c in sequence]
    modified_positions: set[int] = set()
    if modifications:
        for mod in modifications:
            seq_3letter[mod.position] = mod.ccd
            modified_positions.add(mod.position)

    token_idx = token_offset
    atom_idx = atom_offset
    space_uid = space_uid_offset

    for res_idx, res_name in enumerate(seq_3letter):
        is_modified = res_idx in modified_positions

        if not is_modified and res_name in heavy_atoms:
            # Standard nucleotide
            atom_names = heavy_atoms[res_name]
            res_type = _compute_res_type(res_name, mol_type)
            input_id = DNA_RNA_LIGAND_INPUT_ID

            atom_start = atom_idx
            for a_name in atom_names:
                ref_pos = get_idealized_atom_pos(res_type, a_name)
                atoms.append(
                    AtomInfo(
                        name=a_name,
                        element=_infer_element(a_name),
                        charge=CHARGED_ATOMS.get((res_name, a_name), 0),
                        ref_pos=ref_pos.copy()
                        if ref_pos is not None
                        else _ZERO_POS.copy(),
                        pos=_ZERO_POS.copy(),
                        token_index=token_idx,
                        atom_index=atom_idx,
                        space_uid=space_uid,
                    )
                )
                atom_idx += 1

            tokens.append(
                TokenInfo(
                    token_index=token_idx,
                    residue_index=res_idx,
                    residue_name=res_name,
                    mol_type=mol_type,
                    res_type=res_type,
                    input_id=input_id,
                    asym_id=asym_id,
                    sym_id=sym_id,
                    entity_id=entity_id,
                    atom_start=atom_start,
                    atom_count=len(atom_names),
                )
            )
            token_idx += 1
            space_uid += 1

        elif not is_modified and res_name == "UNK":
            # Unknown nucleotide: backbone only
            atom_names = backbone_atoms
            atom_start = atom_idx
            for a_name in atom_names:
                ref_pos = None  # No idealized positions for UNK
                atoms.append(
                    AtomInfo(
                        name=a_name,
                        element=_infer_element(a_name),
                        charge=0,
                        ref_pos=_ZERO_POS.copy(),
                        pos=_ZERO_POS.copy(),
                        token_index=token_idx,
                        atom_index=atom_idx,
                        space_uid=space_uid,
                    )
                )
                atom_idx += 1

            tokens.append(
                TokenInfo(
                    token_index=token_idx,
                    residue_index=res_idx,
                    residue_name=res_name,
                    mol_type=mol_type,
                    res_type=unk_res_type,
                    input_id=DNA_RNA_LIGAND_INPUT_ID,
                    asym_id=asym_id,
                    sym_id=sym_id,
                    entity_id=entity_id,
                    atom_start=atom_start,
                    atom_count=len(atom_names),
                )
            )
            token_idx += 1
            space_uid += 1

        else:
            # Modified nucleotide: atom-tokenized
            ccd_atoms = get_ligand_ccd_atoms_with_charges(res_name)
            if ccd_atoms is None:
                ccd_atoms = [
                    (_infer_element(n), _infer_element(n), 0) for n in backbone_atoms
                ]

            is_terminal = res_idx == len(seq_3letter) - 1
            leaving_atoms = set() if is_terminal else get_ccd_leaving_atoms(res_name)

            for a_name, a_element, a_charge in ccd_atoms:
                if a_name in leaving_atoms:
                    continue
                ref_pos = get_ligand_idealized_atom_pos(res_name, a_name)
                atoms.append(
                    AtomInfo(
                        name=a_name,
                        element=a_element,
                        charge=a_charge,
                        ref_pos=ref_pos.copy()
                        if ref_pos is not None
                        else _ZERO_POS.copy(),
                        pos=_ZERO_POS.copy(),
                        token_index=token_idx,
                        atom_index=atom_idx,
                        space_uid=space_uid,
                    )
                )
                tokens.append(
                    TokenInfo(
                        token_index=token_idx,
                        residue_index=res_idx,
                        residue_name=res_name,
                        mol_type=mol_type,
                        res_type=PROTEIN_UNK_RES_TYPE,
                        input_id=DNA_RNA_LIGAND_INPUT_ID,
                        asym_id=asym_id,
                        sym_id=sym_id,
                        entity_id=entity_id,
                        atom_start=atom_idx,
                        atom_count=1,
                    )
                )
                token_idx += 1
                atom_idx += 1

            space_uid += 1

    return tokens, atoms


def tokenize_ligand_ccd(
    ccd_codes: list[str],
    entity_id: int,
    asym_id: int,
    sym_id: int,
    token_offset: int,
    atom_offset: int,
    space_uid_offset: int,
    has_covalent_bond: bool,
) -> tuple[list[TokenInfo], list[AtomInfo]]:
    """Tokenize a ligand from CCD codes (1 token per atom)."""
    tokens: list[TokenInfo] = []
    atoms: list[AtomInfo] = []

    token_idx = token_offset
    atom_idx = atom_offset
    space_uid = space_uid_offset

    for res_idx, code in enumerate(ccd_codes):
        ccd_atoms = get_ligand_ccd_atoms_with_charges(code)
        if ccd_atoms is None:
            raise ValueError(f"CCD component {code} not found")

        leaving_atoms = get_ccd_leaving_atoms(code) if has_covalent_bond else set()

        for a_name, a_element, a_charge in ccd_atoms:
            if a_name in leaving_atoms:
                continue
            ref_pos = get_ligand_idealized_atom_pos(code, a_name)
            atoms.append(
                AtomInfo(
                    name=a_name,
                    element=a_element,
                    charge=a_charge,
                    ref_pos=ref_pos.copy() if ref_pos is not None else _ZERO_POS.copy(),
                    pos=_ZERO_POS.copy(),
                    token_index=token_idx,
                    atom_index=atom_idx,
                    space_uid=space_uid,
                )
            )
            tokens.append(
                TokenInfo(
                    token_index=token_idx,
                    residue_index=res_idx,
                    residue_name=code,
                    mol_type=MOL_TYPE_NONPOLYMER,
                    res_type=PROTEIN_UNK_RES_TYPE,
                    input_id=DNA_RNA_LIGAND_INPUT_ID,
                    asym_id=asym_id,
                    sym_id=sym_id,
                    entity_id=entity_id,
                    atom_start=atom_idx,
                    atom_count=1,
                )
            )
            token_idx += 1
            atom_idx += 1

        space_uid += 1

    return tokens, atoms


def tokenize_ligand_smiles(
    smiles: str,
    entity_id: int,
    asym_id: int,
    sym_id: int,
    token_offset: int,
    atom_offset: int,
    space_uid_offset: int,
    seed: int | None = None,
) -> tuple[list[TokenInfo], list[AtomInfo], list[tuple[str, str]]]:
    """Tokenize a ligand from SMILES (1 token per heavy atom).

    Returns tokens, atoms, and heavy-atom bonds as (name1, name2) pairs.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    # Assign atom names using canonical ranking
    canonical_order = AllChem.CanonicalRankAtoms(mol)  # type: ignore[attr-defined]
    for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
        atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
        if len(atom_name) > 4:
            raise ValueError(
                f"SMILES {smiles} has atom name longer than 4 chars: {atom_name}"
            )
        atom.SetProp("name", atom_name)

    # Generate 3D conformer
    options = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    options.clearConfs = False
    if seed is not None:
        options.randomSeed = seed
    conf_id = AllChem.EmbedMolecule(mol, options)  # type: ignore[attr-defined]
    if conf_id == -1:
        options.useRandomCoords = True
        conf_id = AllChem.EmbedMolecule(mol, options)  # type: ignore[attr-defined]
    if conf_id != -1:
        try:
            AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)  # type: ignore[attr-defined]
        except (RuntimeError, ValueError):
            pass

    # Remove hydrogens
    mol_no_h = Chem.RemoveHs(mol)
    if mol_no_h.GetNumConformers() == 0:
        raise ValueError(f"Failed to generate conformer for SMILES: {smiles}")

    conformer = mol_no_h.GetConformer(0)

    tokens: list[TokenInfo] = []
    atoms_list: list[AtomInfo] = []
    token_idx = token_offset
    atom_idx = atom_offset
    space_uid = space_uid_offset

    for atom in mol_no_h.GetAtoms():
        a_name = atom.GetProp("name")
        a_element = atom.GetSymbol()
        a_charge = atom.GetFormalCharge()
        pos_3d = conformer.GetAtomPosition(atom.GetIdx())
        ref_pos = np.array([pos_3d.x, pos_3d.y, pos_3d.z], dtype=np.float32)

        atoms_list.append(
            AtomInfo(
                name=a_name,
                element=a_element,
                charge=a_charge,
                ref_pos=ref_pos,
                pos=_ZERO_POS.copy(),
                token_index=token_idx,
                atom_index=atom_idx,
                space_uid=space_uid,
            )
        )
        tokens.append(
            TokenInfo(
                token_index=token_idx,
                residue_index=0,
                residue_name="LIG",
                mol_type=MOL_TYPE_NONPOLYMER,
                res_type=PROTEIN_UNK_RES_TYPE,
                input_id=DNA_RNA_LIGAND_INPUT_ID,
                asym_id=asym_id,
                sym_id=sym_id,
                entity_id=entity_id,
                atom_start=atom_idx,
                atom_count=1,
            )
        )
        token_idx += 1
        atom_idx += 1

    bonds: list[tuple[str, str]] = []
    for bond in mol_no_h.GetBonds():
        n1 = bond.GetBeginAtom().GetProp("name")
        n2 = bond.GetEndAtom().GetProp("name")
        bonds.append((n1, n2))

    return tokens, atoms_list, bonds


# =============================================================================
# Build chains from StructurePredictionInput
# =============================================================================


def _get_sequence_key(item) -> str:
    """Get a hashable key for entity deduplication."""
    if isinstance(item, ProteinInput):
        return f"PROTEIN:{item.sequence}"
    elif isinstance(item, DNAInput):
        return f"DNA:{item.sequence}"
    elif isinstance(item, RNAInput):
        return f"RNA:{item.sequence}"
    elif isinstance(item, LigandInput):
        if item.ccd:
            return f"LIGAND_CCD:{','.join(item.ccd)}"
        return f"LIGAND_SMILES:{item.smiles}"
    raise ValueError(f"Unknown input type: {type(item)}")


def build_chains_from_input(
    input: StructurePredictionInput, seed: int | None = None
) -> tuple[list[ChainInfo], list[TokenInfo], list[AtomInfo]]:
    """Build chains, tokens, and atoms from StructurePredictionInput.

    Handles entity deduplication (identical sequences get same entity_id),
    sym_id assignment, and delegates to type-specific tokenization functions.
    """
    chains: list[ChainInfo] = []
    all_tokens: list[TokenInfo] = []
    all_atoms: list[AtomInfo] = []

    # Entity deduplication
    sequence_to_entity: dict[str, int] = {}
    entity_sym_count: dict[int, int] = {}
    next_entity_id = 0

    # Gather chain IDs involved in covalent bonds
    covalent_chain_ids: set[str] = set()
    if input.covalent_bonds:
        for cb in input.covalent_bonds:
            covalent_chain_ids.update([cb.chain_id1, cb.chain_id2])

    token_offset = 0
    atom_offset = 0
    space_uid_offset = 0
    asym_id = 0

    for item in input.sequences:
        # Entity deduplication
        seq_key = _get_sequence_key(item)
        if seq_key in sequence_to_entity:
            entity_id = sequence_to_entity[seq_key]
        else:
            entity_id = next_entity_id
            sequence_to_entity[seq_key] = entity_id
            next_entity_id += 1

        # Get all chain IDs for this item
        ids = [item.id] if isinstance(item.id, str) else item.id

        for chain_id_str in ids:
            # sym_id is the per-entity copy index; increment per chain so
            # ProteinInput(id=['A','B']) gives chain A sym_id=0, chain B sym_id=1.
            sym_id = entity_sym_count.get(entity_id, 0)
            entity_sym_count[entity_id] = sym_id + 1
            if isinstance(item, ProteinInput):
                if item.msa is None:
                    warnings.warn(
                        f"No MSA provided for {item.id}, using single sequence mode"
                    )

                new_tokens, new_atoms = tokenize_protein(
                    sequence=item.sequence,
                    modifications=item.modifications,
                    entity_id=entity_id,
                    asym_id=asym_id,
                    sym_id=sym_id,
                    token_offset=token_offset,
                    atom_offset=atom_offset,
                    space_uid_offset=space_uid_offset,
                )

            elif isinstance(item, (DNAInput, RNAInput)):
                mol_type = MOL_TYPE_DNA if isinstance(item, DNAInput) else MOL_TYPE_RNA
                new_tokens, new_atoms = tokenize_nucleotide(
                    sequence=item.sequence,
                    modifications=item.modifications,
                    mol_type=mol_type,
                    entity_id=entity_id,
                    asym_id=asym_id,
                    sym_id=sym_id,
                    token_offset=token_offset,
                    atom_offset=atom_offset,
                    space_uid_offset=space_uid_offset,
                )

            elif isinstance(item, LigandInput):
                has_cov = chain_id_str in covalent_chain_ids
                ligand_bonds: list[tuple[str, str]] = []
                if item.ccd is not None:
                    if item.smiles is not None:
                        warnings.warn("Both ccd and smiles provided, using ccd")
                    new_tokens, new_atoms = tokenize_ligand_ccd(
                        ccd_codes=item.ccd,
                        entity_id=entity_id,
                        asym_id=asym_id,
                        sym_id=sym_id,
                        token_offset=token_offset,
                        atom_offset=atom_offset,
                        space_uid_offset=space_uid_offset,
                        has_covalent_bond=has_cov,
                    )
                elif item.smiles is not None:
                    new_tokens, new_atoms, ligand_bonds = tokenize_ligand_smiles(
                        smiles=item.smiles,
                        entity_id=entity_id,
                        asym_id=asym_id,
                        sym_id=sym_id,
                        token_offset=token_offset,
                        atom_offset=atom_offset,
                        space_uid_offset=space_uid_offset,
                        seed=seed,
                    )
                else:
                    raise ValueError("LigandInput must have either ccd or smiles")
            else:
                raise ValueError(f"Unknown input type: {type(item)}")

            chain = ChainInfo(
                chain_id=chain_id_str,
                asym_id=asym_id,
                entity_id=entity_id,
                sym_id=sym_id,
                mol_type=new_tokens[0].mol_type if new_tokens else MOL_TYPE_PROTEIN,
                tokens=new_tokens,
                ligand_bonds=ligand_bonds if isinstance(item, LigandInput) else [],
            )
            chains.append(chain)
            all_tokens.extend(new_tokens)
            all_atoms.extend(new_atoms)

            token_offset += len(new_tokens)
            atom_offset += len(new_atoms)
            space_uid_offset += len(set(a.space_uid for a in new_atoms))
            asym_id += 1

    return chains, all_tokens, all_atoms


# =============================================================================
# Feature tensor building
# =============================================================================


def compute_frame_indices(
    tokens: list[TokenInfo], atoms: list[AtomInfo]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute backbone frame indices for each token.

    Protein: [N, CA, C]; DNA/RNA: [C1', C3', C4']; Ligand: distance-based.
    """
    # Build atom name -> atom_index lookup per token
    token_atoms: dict[int, dict[str, int]] = defaultdict(dict)
    for atom in atoms:
        if atom.is_valid:
            token_atoms[atom.token_index][atom.name] = atom.atom_index

    # Ligand-token frames come from CCD reference-conformer geometry,
    # grouped per residue. For each token, the frame is the 3 atoms nearest
    # to its own atom in the residue's ref-pos space, ordered
    # (1st-nearest, self, 2nd-nearest).
    ligand_token_to_atom: dict[int, int] = {}
    ligand_tokens_by_res: dict[tuple[int, int], list[int]] = defaultdict(list)
    for t in tokens:
        if t.mol_type == MOL_TYPE_NONPOLYMER:
            ad = token_atoms.get(t.token_index)
            if ad:
                ligand_token_to_atom[t.token_index] = next(iter(ad.values()))
            ligand_tokens_by_res[(t.asym_id, t.residue_index)].append(t.token_index)

    ligand_token_frames: dict[int, tuple[int, int, int]] = {}
    for tok_indices in ligand_tokens_by_res.values():
        atom_indices = [
            ligand_token_to_atom[ti] for ti in tok_indices if ti in ligand_token_to_atom
        ]
        if len(atom_indices) < 3:
            for ti in tok_indices:
                if ti in ligand_token_to_atom:
                    ai = ligand_token_to_atom[ti]
                    ligand_token_frames[ti] = (ai, ai, ai)
            continue

        ref_pos_chain = np.array([atoms[ai].ref_pos for ai in atom_indices])
        dist_mat = np.sqrt(
            ((ref_pos_chain[:, None] - ref_pos_chain[None]) ** 2).sum(-1)
        )
        sort_indices = np.argsort(dist_mat, axis=1)
        local_frames = np.column_stack(
            [sort_indices[:, 1], sort_indices[:, 0], sort_indices[:, 2]]
        )

        for ti in tok_indices:
            if ti not in ligand_token_to_atom:
                continue
            ai = ligand_token_to_atom[ti]
            local_idx = atom_indices.index(ai)
            fl = local_frames[local_idx]
            ligand_token_frames[ti] = (
                atom_indices[fl[0]],
                atom_indices[fl[1]],
                atom_indices[fl[2]],
            )

    # Build frames for all tokens
    frames_list: list[tuple[int, int, int]] = []
    for t in tokens:
        ad = token_atoms.get(t.token_index, {})
        fallback = list(ad.values())[0] if ad else 0

        if t.mol_type == MOL_TYPE_PROTEIN:
            if t.res_type == PROTEIN_UNK_RES_TYPE:
                frames_list.append((fallback, fallback, fallback))
            else:
                frames_list.append((ad.get("N", 0), ad.get("CA", 0), ad.get("C", 0)))
        elif t.mol_type in (MOL_TYPE_DNA, MOL_TYPE_RNA):
            if t.res_type == PROTEIN_UNK_RES_TYPE:
                frames_list.append((fallback, fallback, fallback))
            else:
                frames_list.append(
                    (ad.get("C1'", 0), ad.get("C3'", 0), ad.get("C4'", 0))
                )
        elif t.mol_type == MOL_TYPE_NONPOLYMER:
            if t.token_index in ligand_token_frames:
                frames_list.append(ligand_token_frames[t.token_index])
            else:
                frames_list.append((fallback, fallback, fallback))
        else:
            frames_list.append((fallback, fallback, fallback))

    frames = np.array(frames_list, dtype=np.int64)

    # Compute resolved mask (vectorized)
    n_atoms = len(atoms)
    atom_positions = (
        np.array([a.pos for a in atoms], dtype=np.float32)
        if atoms
        else np.zeros((0, 3), dtype=np.float32)
    )
    atom_is_valid = (
        np.array([a.is_valid for a in atoms], dtype=bool)
        if atoms
        else np.zeros(0, dtype=bool)
    )
    atom_is_resolved = (
        atom_is_valid & np.any(atom_positions != 0, axis=1)
        if n_atoms > 0
        else np.zeros(0, dtype=bool)
    )

    n_tokens = len(tokens)
    if n_tokens == 0:
        return frames, np.zeros(0, dtype=bool)

    pos1 = atom_positions[frames[:, 0]]
    pos2 = atom_positions[frames[:, 1]]
    pos3 = atom_positions[frames[:, 2]]

    all_resolved = (
        atom_is_resolved[frames[:, 0]]
        & atom_is_resolved[frames[:, 1]]
        & atom_is_resolved[frames[:, 2]]
    )
    all_same = (frames[:, 0] == frames[:, 1]) & (frames[:, 1] == frames[:, 2])

    v1 = pos1 - pos2
    v2 = pos3 - pos2
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    valid_norms = (norm1 >= 1e-6) & (norm2 >= 1e-6)

    cos_angle = np.zeros(n_tokens, dtype=np.float32)
    mask = valid_norms
    if np.any(mask):
        cos_angle[mask] = np.sum(v1[mask] * v2[mask], axis=1) / (
            norm1[mask] * norm2[mask]
        )
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_deg = np.degrees(np.arccos(np.abs(cos_angle)))
    not_colinear = angle_deg >= 25

    resolved_mask = all_resolved & ~all_same & valid_norms & not_colinear
    return frames, resolved_mask


def compute_token_bonds(
    tokens: list[TokenInfo],
    atoms: list[AtomInfo],
    input: StructurePredictionInput,
    chains: list[ChainInfo],
) -> torch.Tensor:
    """Compute dense token bond matrix [L, L, 1].

    Includes ligand intra-residue bonds (from CCD) and covalent bonds.
    """
    n_tokens = len(tokens)
    edge_set: set[tuple[int, int]] = set()

    def add_bond(i: int, j: int) -> None:
        if i != j:
            edge_set.add((min(i, j), max(i, j)))

    # Build per-residue atom name -> token_index mapping for ligands and modified residues
    # Key: (asym_id, residue_index, atom_name) -> token_index
    atom_name_to_token: dict[tuple[int, int, str], int] = {}
    for atom in atoms:
        if atom.is_valid:
            t = tokens[atom.token_index] if atom.token_index < len(tokens) else None
            if t and (
                t.mol_type == MOL_TYPE_NONPOLYMER or t.res_type == PROTEIN_UNK_RES_TYPE
            ):
                atom_name_to_token[(t.asym_id, t.residue_index, atom.name)] = (
                    atom.token_index
                )

    # Group atom-tokenized tokens by (asym_id, residue_index)
    residue_tokens: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    for atom in atoms:
        if not atom.is_valid:
            continue
        t = tokens[atom.token_index] if atom.token_index < len(tokens) else None
        if t and (
            t.mol_type == MOL_TYPE_NONPOLYMER or t.res_type == PROTEIN_UNK_RES_TYPE
        ):
            residue_tokens[(t.asym_id, t.residue_index)].append(
                (atom.name, atom.token_index)
            )

    # SMILES ligand bonds keyed by (asym_id, residue_index 0).
    explicit_bonds: dict[tuple[int, int], list[tuple[str, str]]] = {
        (c.asym_id, 0): c.ligand_bonds for c in chains if c.ligand_bonds
    }

    # Add intra-residue bonds from CCD
    for (asym_id_val, res_idx), atom_list in residue_tokens.items():
        if not atom_list:
            continue
        res_name = tokens[atom_list[0][1]].residue_name
        atom_to_tok = {name: ti for name, ti in atom_list}

        bonds = explicit_bonds.get((asym_id_val, res_idx))
        if bonds is None:
            bonds = get_ligand_ccd_bonds(res_name)

        if bonds:
            for a1, a2 in bonds:
                if a1 in atom_to_tok and a2 in atom_to_tok:
                    add_bond(atom_to_tok[a1], atom_to_tok[a2])
        else:
            # Fallback: fully connected within residue
            tok_indices = [ti for _, ti in atom_list]
            for i_idx in tok_indices:
                for j_idx in tok_indices:
                    add_bond(i_idx, j_idx)

    # Add covalent bonds from input
    if input.covalent_bonds:
        # Build chain_id -> chain mapping
        chain_by_id: dict[str, ChainInfo] = {c.chain_id: c for c in chains}
        # Build (asym_id, residue_index) -> list of tokens for atom index lookup
        chain_res_atoms: dict[tuple[int, int], list[AtomInfo]] = defaultdict(list)
        for atom in atoms:
            if atom.is_valid and atom.token_index < len(tokens):
                t = tokens[atom.token_index]
                chain_res_atoms[(t.asym_id, t.residue_index)].append(atom)

        for cb in input.covalent_bonds:
            c1 = chain_by_id.get(cb.chain_id1)
            c2 = chain_by_id.get(cb.chain_id2)
            if c1 is None or c2 is None:
                continue

            atoms_1 = chain_res_atoms.get((c1.asym_id, cb.res_idx1), [])
            atoms_2 = chain_res_atoms.get((c2.asym_id, cb.res_idx2), [])

            if cb.atom_idx1 < len(atoms_1) and cb.atom_idx2 < len(atoms_2):
                add_bond(
                    atoms_1[cb.atom_idx1].token_index, atoms_2[cb.atom_idx2].token_index
                )

    # Add peptide bonds at modified-residue boundaries: an atom-tokenized
    # residue's N atom connects to the prev residue's C atom (and same for
    # the C side to the next residue's N).
    tokens_by_chain_res: dict[tuple[int, int], list[TokenInfo]] = defaultdict(list)
    for t in tokens:
        if t.mol_type == MOL_TYPE_PROTEIN:
            tokens_by_chain_res[(t.asym_id, t.residue_index)].append(t)

    def _backbone_token(res_tokens: list[TokenInfo], atom_name: str) -> int | None:
        # Standard residue (single token wrapping all atoms): return that token.
        if len(res_tokens) == 1 and res_tokens[0].res_type != PROTEIN_UNK_RES_TYPE:
            return res_tokens[0].token_index
        for t in res_tokens:
            for a_idx in range(t.atom_start, t.atom_start + t.atom_count):
                if a_idx < len(atoms) and atoms[a_idx].name == atom_name:
                    return t.token_index
        # Atom-tokenized residue without an atom of that name (e.g. ACE has
        # no N, NH2 has no C). Fall back to the first atom-tokenized token.
        return res_tokens[0].token_index if res_tokens else None

    for (asym_id_val, res_idx), res_tokens in tokens_by_chain_res.items():
        is_atom_tokenized = any(t.res_type == PROTEIN_UNK_RES_TYPE for t in res_tokens)
        if not is_atom_tokenized:
            continue  # Standard residue — no peptide bond added here.
        n_tok = _backbone_token(res_tokens, "N")
        c_tok = _backbone_token(res_tokens, "C")
        prev_tokens = tokens_by_chain_res.get((asym_id_val, res_idx - 1))
        if prev_tokens and n_tok is not None:
            prev_c = _backbone_token(prev_tokens, "C")
            if prev_c is not None:
                add_bond(prev_c, n_tok)
        next_tokens = tokens_by_chain_res.get((asym_id_val, res_idx + 1))
        if next_tokens and c_tok is not None:
            next_n = _backbone_token(next_tokens, "N")
            if next_n is not None:
                add_bond(c_tok, next_n)

    # Expand to dense matrix
    bonds = torch.zeros(n_tokens, n_tokens, 1, dtype=torch.float32)
    for i, j in edge_set:
        bonds[i, j, 0] = 1.0
        bonds[j, i, 0] = 1.0
    return bonds


def compute_representative_atoms(
    tokens: list[TokenInfo], atoms: list[AtomInfo]
) -> torch.Tensor:
    """Compute representative atom index per token (for token_to_rep_atom).

    Returns:
        distogram_atom_idx: [L] — representative atom per token
            Protein: CB (or CA for GLY), DNA/RNA: C4/C2/C1', Ligand: first atom.
    """
    n_tokens = len(tokens)

    # Build atom name -> index lookup per token
    token_atoms: dict[int, dict[str, int]] = defaultdict(dict)
    for atom in atoms:
        if atom.is_valid:
            token_atoms[atom.token_index][atom.name] = atom.atom_index

    distogram_atom_idx = torch.zeros(n_tokens, dtype=torch.int64)

    for t in tokens:
        ad = token_atoms.get(t.token_index, {})
        fallback_idx = list(ad.values())[0] if ad else 0

        if t.mol_type == MOL_TYPE_PROTEIN:
            rep_idx = ad.get("CB", ad.get("CA", fallback_idx))
        elif t.mol_type in (MOL_TYPE_DNA, MOL_TYPE_RNA):
            if t.res_type in (27, 32):  # Unknown nucleotides
                rep_idx = ad.get("C1'", fallback_idx)
            elif t.res_type in (23, 24, 28, 29):  # Purines (A, G)
                rep_idx = ad.get("C4", ad.get("C1'", fallback_idx))
            else:  # Pyrimidines (C, U, T)
                rep_idx = ad.get("C2", ad.get("C1'", fallback_idx))
        else:
            rep_idx = fallback_idx

        distogram_atom_idx[t.token_index] = rep_idx

    return distogram_atom_idx


def compute_msa_features(
    input: StructurePredictionInput,
    chains: list[ChainInfo],
    tokens: list[TokenInfo],
    max_seqs: int = 16384,
) -> dict[str, torch.Tensor]:
    """Compute MSA features from protein MSAs.

    Uses taxonomy-based pairing across chains
    (:func:`paired_msa.construct_paired_msa`): rows whose FASTA header
    contains ``key=N`` get paired across chains sharing the same ``N``.

    Output: msa [M, L], deletion_value [M, L], has_deletion [M, L],
            deletion_mean [L], msa_mask [M, L]
    """
    from esm.models.esmfold2.paired_msa import (
        construct_paired_msa,
        protein_letter_to_res_type,
    )

    n_tokens = len(tokens)

    # A single ProteinInput with id=['A','B','C',...] yields one item but
    # multiple chains (one per id); broadcast the MSA across all of them.
    chain_msas: dict[int, MSA | None] = {}
    item_idx = 0
    for item in input.sequences:
        ids = [item.id] if isinstance(item.id, str) else list(item.id)
        for _ in ids:
            chain = chains[item_idx]
            if isinstance(item, ProteinInput):
                msa = item.msa
                if msa is None:
                    msa = MSA.from_sequences([item.sequence])
                chain_msas[chain.asym_id] = msa
            else:
                chain_msas[chain.asym_id] = None
            item_idx += 1

    letter_to_res_type = protein_letter_to_res_type()

    # Build per-chain query res_types (used for chains without an MSA).
    chain_query_res_types: dict[int, np.ndarray] = {}
    for chain in chains:
        chain_tokens = [t for t in tokens if t.asym_id == chain.asym_id]
        chain_query_res_types[chain.asym_id] = np.array(
            [t.res_type for t in chain_tokens], dtype=np.int64
        )

    token_asym_ids = np.array([t.asym_id for t in tokens], dtype=np.int64)
    token_res_ids = np.array([t.residue_index for t in tokens], dtype=np.int64)

    msa_res, del_counts, paired = construct_paired_msa(
        chain_msas,
        chain_query_res_types,
        token_asym_ids,
        token_res_ids,
        letter_to_res_type=letter_to_res_type,
        max_seqs=max_seqs,
    )

    # Tokens for chains without an MSA get their res_type at row 0 and gap
    # elsewhere; this mirrors the prior non-protein-token branch.
    for t in tokens:
        if chain_msas.get(t.asym_id) is None:
            msa_res[:, t.token_index] = MSA_GAP_TOKEN_ID
            msa_res[0, t.token_index] = t.res_type

    if msa_res.shape[0] == 0:
        msa_res = np.full((1, n_tokens), MSA_GAP_TOKEN_ID, dtype=np.int64)
        del_counts = np.zeros((1, n_tokens), dtype=np.float32)

    msa_data = torch.from_numpy(msa_res)
    del_data = torch.from_numpy(del_counts)

    has_deletion = del_data > 0
    deletion_value = (np.pi / 2) * torch.arctan(del_data / 3)
    deletion_mean = deletion_value.mean(dim=0)

    msa_mask = torch.ones_like(msa_data, dtype=torch.bool)

    return {
        "msa": msa_data,
        "deletion_value": deletion_value,
        "has_deletion": has_deletion,
        "deletion_mean": deletion_mean,
        "msa_attention_mask": msa_mask,
    }


def compute_distogram_conditioning(
    input: StructurePredictionInput,
    chains: list[ChainInfo],
    tokens: list[TokenInfo],
    disto_center: torch.Tensor,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    num_bins: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute distogram conditioning from user-provided distograms.

    Returns:
        disto_cond: [L, L] int64 (bin indices)
        disto_cond_mask: [L, L] bool
    """
    n_tokens = len(tokens)
    disto_cond = torch.zeros(n_tokens, n_tokens, dtype=torch.long)
    disto_cond_mask = torch.zeros(n_tokens, n_tokens, dtype=torch.bool)

    if not input.distogram_conditioning:
        return disto_cond, disto_cond_mask

    # Build chain_id -> asym_id mapping
    chain_id_to_asym: dict[str, int] = {c.chain_id: c.asym_id for c in chains}

    # Build asym_id -> token indices mapping
    asym_to_tokens: dict[int, list[int]] = defaultdict(list)
    for t in tokens:
        asym_to_tokens[t.asym_id].append(t.token_index)

    boundaries = torch.linspace(min_dist, max_dist, num_bins + 1)

    for dc in input.distogram_conditioning:
        asym_id_val = chain_id_to_asym.get(dc.chain_id)
        if asym_id_val is None:
            continue
        tok_indices = asym_to_tokens[asym_id_val]
        n_chain = len(tok_indices)
        distogram = torch.tensor(dc.distogram, dtype=torch.float32)

        if distogram.shape != (n_chain, n_chain):
            raise ValueError(
                f"Distogram shape {distogram.shape} doesn't match chain length {n_chain}"
            )

        # Bin the distogram
        binned = torch.bucketize(distogram, boundaries[:-1]) - 1
        binned = binned.clamp(0, num_bins - 1)

        for i, ti in enumerate(tok_indices):
            for j, tj in enumerate(tok_indices):
                disto_cond[ti, tj] = binned[i, j]
                disto_cond_mask[ti, tj] = True

    return disto_cond, disto_cond_mask


def build_feature_tensors(
    chains: list[ChainInfo],
    tokens: list[TokenInfo],
    atoms: list[AtomInfo],
    input: StructurePredictionInput,
) -> dict[str, torch.Tensor]:
    """Build all model input tensors from tokens and atoms."""
    n_tokens = len(tokens)
    n_real_atoms = len(atoms)

    # Pad atoms to nearest multiple of 32
    target_atoms = math.ceil(n_real_atoms / 32) * 32 if n_real_atoms > 0 else 32
    n_padding = target_atoms - n_real_atoms
    padding_atoms = [
        AtomInfo(
            name="",
            element="",
            charge=0,
            ref_pos=_ZERO_POS.copy(),
            pos=_ZERO_POS.copy(),
            token_index=0,
            atom_index=n_real_atoms + i,
            space_uid=0,
            is_valid=False,
        )
        for i in range(n_padding)
    ]
    all_atoms = atoms + padding_atoms
    n_atoms = len(all_atoms)

    # --- Token-level tensors ---
    token_index_arr = np.empty(n_tokens, dtype=np.int64)
    residue_index_arr = np.empty(n_tokens, dtype=np.int64)
    asym_id_arr = np.empty(n_tokens, dtype=np.int64)
    sym_id_arr = np.empty(n_tokens, dtype=np.int64)
    entity_id_arr = np.empty(n_tokens, dtype=np.int64)
    mol_type_arr = np.empty(n_tokens, dtype=np.int64)
    res_type_arr = np.empty(n_tokens, dtype=np.int64)
    input_ids_arr = np.empty(n_tokens, dtype=np.int64)

    for i, t in enumerate(tokens):
        token_index_arr[i] = t.token_index
        residue_index_arr[i] = t.residue_index
        asym_id_arr[i] = t.asym_id
        sym_id_arr[i] = t.sym_id
        entity_id_arr[i] = t.entity_id
        mol_type_arr[i] = t.mol_type
        res_type_arr[i] = t.res_type
        input_ids_arr[i] = t.input_id

    token_index = torch.from_numpy(token_index_arr)
    residue_index = torch.from_numpy(residue_index_arr)
    asym_id = torch.from_numpy(asym_id_arr)
    sym_id = torch.from_numpy(sym_id_arr)
    entity_id = torch.from_numpy(entity_id_arr)
    mol_type = torch.from_numpy(mol_type_arr)
    res_type = torch.from_numpy(res_type_arr)
    input_ids = torch.from_numpy(input_ids_arr)
    token_pad_mask = torch.ones(n_tokens, dtype=torch.bool)

    # --- Atom-level tensors ---
    ref_pos_arr = np.zeros((n_atoms, 3), dtype=np.float32)
    ref_element_arr = np.zeros(n_atoms, dtype=np.int64)
    ref_charge_arr = np.zeros(n_atoms, dtype=np.int8)
    ref_atom_name_chars_arr = np.zeros((n_atoms, 4), dtype=np.int64)
    ref_space_uid_arr = np.zeros(n_atoms, dtype=np.int64)
    atom_pad_mask_arr = np.zeros(n_atoms, dtype=np.bool_)
    atom_to_token_arr = np.zeros(n_atoms, dtype=np.int64)
    all_positions = np.zeros((n_atoms, 3), dtype=np.float64)
    is_valid_arr = np.zeros(n_atoms, dtype=np.bool_)

    for i, atom in enumerate(all_atoms):
        if atom.ref_pos is not None:
            ref_pos_arr[i] = atom.ref_pos
        ref_charge_arr[i] = atom.charge
        ref_space_uid_arr[i] = (
            atom.space_uid if atom.space_uid >= 0 else atom.token_index
        )
        atom_pad_mask_arr[i] = atom.is_valid
        is_valid_arr[i] = atom.is_valid
        all_positions[i] = atom.pos

        if atom.is_valid:
            ref_element_arr[i] = get_element_atomic_num(atom.element)
            name_indices = encode_atom_name(atom.name)
            ref_atom_name_chars_arr[i] = name_indices
            atom_to_token_arr[i] = atom.token_index

    ref_pos = torch.from_numpy(ref_pos_arr)
    ref_element = torch.from_numpy(ref_element_arr)
    ref_charge = torch.from_numpy(ref_charge_arr)
    ref_atom_name_chars = torch.from_numpy(ref_atom_name_chars_arr)
    ref_space_uid = torch.from_numpy(ref_space_uid_arr)
    atom_pad_mask = torch.from_numpy(atom_pad_mask_arr)
    atom_to_token = torch.from_numpy(atom_to_token_arr)

    # Coordinates — center on resolved atoms
    raw_coords = torch.from_numpy(all_positions)
    is_nonzero = np.any(all_positions != 0, axis=1)
    atom_resolved_arr = is_valid_arr & is_nonzero
    resolved_mask = torch.from_numpy(atom_resolved_arr)
    valid_mask = torch.from_numpy(is_valid_arr)

    if resolved_mask.any():
        centroid = raw_coords[resolved_mask].mean(dim=0, keepdim=True)
        raw_coords = raw_coords - centroid
        raw_coords[~valid_mask] = 0.0

    coords = raw_coords.float().unsqueeze(0)  # [1, A, 3]
    atom_resolved_mask = torch.tensor(atom_resolved_arr, dtype=torch.bool)

    # --- Frames ---
    frames, _ = compute_frame_indices(tokens, atoms)
    frames_idx = torch.from_numpy(frames).to(torch.int64)

    # --- Token bonds ---
    token_bonds = compute_token_bonds(tokens, atoms, input, chains)

    # --- Representative atoms ---
    distogram_atom_idx = compute_representative_atoms(tokens, atoms)

    # --- MSA features ---
    msa_features = compute_msa_features(input, chains, tokens)

    # --- Distogram conditioning ---
    # disto_center is not needed for inference (no experimental coords)
    disto_center = torch.zeros(n_tokens, 3, dtype=torch.float32)
    disto_cond, disto_cond_mask = compute_distogram_conditioning(
        input, chains, tokens, disto_center
    )

    # ref_pos: CCD conformer positions, used as-is for inference.
    # No random rotation or masking — at inference there are no resolved
    # experimental coordinates, so atom_resolved_mask is all False.
    # The model uses ref_pos for atom feature embedding.

    # --- Pocket (dropped) ---
    pocket_feature = torch.zeros(n_tokens, dtype=torch.long)

    return {
        # Token-level
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "res_type": res_type,
        "input_ids": input_ids,
        "token_bonds": token_bonds,
        "token_attention_mask": token_pad_mask,
        "pocket_feature": pocket_feature,
        # Atom-level
        "ref_pos": ref_pos,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "gt_coords": coords,
        "atom_attention_mask": atom_pad_mask,
        "atom_to_token": atom_to_token,
        "is_resolved": atom_resolved_mask,
        "distogram_atom_idx": distogram_atom_idx,
        # Frames
        "frames_idx": frames_idx,
        # Distogram
        "disto_cond": disto_cond,
        "disto_cond_mask": disto_cond_mask,
        # MSA
        **msa_features,
    }


# =============================================================================
# Top-level entry point
# =============================================================================


def prepare_esmfold2_input(
    input: StructurePredictionInput, seed: int | None = None
) -> tuple[dict[str, torch.Tensor], list[ChainInfo]]:
    """Prepare ESMFold2 model inputs from StructurePredictionInput.

    Args:
        input: The structure prediction input (sequences, conditioning, etc.)
        seed: Random seed for SMILES conformer generation and augmentation.

    Returns:
        Tuple of (feature_dict, chain_infos) where feature_dict contains
        all tensors for the model forward pass, and chain_infos contains
        metadata for output processing.
    """
    chains, tokens, atoms = build_chains_from_input(input, seed)
    features = build_feature_tensors(chains, tokens, atoms, input)
    return features, chains
