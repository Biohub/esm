from __future__ import annotations

import io
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from subprocess import check_output
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import biotite.structure as bs
import biotite.structure.io.pdbx as pdbx
import brotli
import msgpack
import numpy as np
import torch
from biotite.structure.io.pdbx import (
    CIFCategory,
    CIFColumn,
    CIFData,
    CIFFile,
    set_structure,
)

from esm.utils import residue_constants
from esm.utils.structure.metrics import compute_lddt, compute_rmsd
from esm.utils.structure.protein_complex import ProteinComplex, ProteinComplexMetadata


@dataclass
class MolecularComplexResult:
    """Result of molecular complex folding"""

    complex: MolecularComplex
    plddt: torch.Tensor | None = None
    ptm: float | None = None
    iptm: float | None = None
    pae: torch.Tensor | None = None
    distogram: torch.Tensor | None = None
    pair_chains_iptm: torch.Tensor | None = None
    output_embedding_sequence: torch.Tensor | None = None
    output_embedding_pair_pooled: torch.Tensor | None = None
    residue_index: torch.Tensor | None = None
    entity_id: torch.Tensor | None = None
    sae_features: np.ndarray | None = None  # [L, n_features]


@dataclass
class MolecularComplexMetadata:
    """Metadata for MolecularComplex objects."""

    entity_lookup: dict[int, str]
    chain_lookup: dict[int, str]
    assembly_composition: dict[str, list[str]] | None = None


@dataclass
class Molecule:
    """Represents a single molecule/token within a MolecularComplex."""

    token: str
    token_idx: int
    atom_positions: np.ndarray  # [N_atoms, 3]
    atom_elements: np.ndarray  # [N_atoms] element strings
    atom_names: np.ndarray | None = None  # [N_atoms] atom names (optional)
    atom_hetero: np.ndarray | None = None  # [N_atoms] hetero flags (optional)
    residue_type: int = 0
    molecule_type: int = 0  # PROTEIN=0, RNA=1, DNA=2, LIGAND=3
    confidence: float = 0.0


@dataclass(frozen=True)
class MolecularComplex:
    """
    Dataclass representing a molecular complex with support for proteins, nucleic acids, and ligands.

    Uses a flat atom representation with token-based sequence indexing, supporting all atom types
    beyond the traditional atom37 protein representation.
    """

    id: str
    sequence: list[str]  # Token sequence like ['MET', 'LYS', 'A', 'G', 'ATP']

    # Flat atom arrays - simplified representation
    atom_positions: np.ndarray  # [N_atoms, 3] 3D coordinates
    atom_elements: np.ndarray  # [N_atoms] element strings

    # Token-to-atom mapping for efficient access
    token_to_atoms: np.ndarray  # [N_tokens, 2] start/end indices into atoms array

    # Chain information
    chain_id: np.ndarray  # [N_tokens] chain identifier for each token

    # Confidence data
    plddt: np.ndarray  # Per-token confidence scores [N_tokens]

    # Metadata
    metadata: MolecularComplexMetadata

    # Optional atom names and hetero flags (preserved from original structures)
    atom_names: np.ndarray | None = None  # [N_atoms] atom names (optional)
    atom_hetero: np.ndarray | None = None  # [N_atoms] hetero flags (optional)

    def __post_init__(self):
        """Validate array dimensions."""
        n_tokens = len(self.sequence)
        n_atoms = len(self.atom_positions)
        assert (
            self.token_to_atoms.shape[0] == n_tokens
        ), f"token_to_atoms shape {self.token_to_atoms.shape} != {n_tokens} tokens"
        assert (
            self.chain_id.shape[0] == n_tokens
        ), f"chain_id shape {self.chain_id.shape} != {n_tokens} tokens"
        assert (
            self.plddt.shape[0] == n_tokens
        ), f"plddt shape {self.plddt.shape} != {n_tokens} tokens"
        if self.atom_names is not None:
            assert (
                self.atom_names.shape[0] == n_atoms
            ), f"atom_names shape {self.atom_names.shape} != {n_atoms} atoms"
        if self.atom_hetero is not None:
            assert (
                self.atom_hetero.shape[0] == n_atoms
            ), f"atom_hetero shape {self.atom_hetero.shape} != {n_atoms} atoms"

    def __len__(self) -> int:
        """Return number of tokens."""
        return len(self.sequence)

    def __getitem__(self, idx: int) -> Molecule:
        """Access individual molecules/tokens by index."""
        if idx >= len(self.sequence) or idx < 0:
            raise IndexError(
                f"Token index {idx} out of range for {len(self.sequence)} tokens"
            )

        token = self.sequence[idx]
        start_atom, end_atom = self.token_to_atoms[idx]

        # Extract atom data for this token
        token_atom_positions = self.atom_positions[start_atom:end_atom]
        token_atom_elements = self.atom_elements[start_atom:end_atom]
        token_atom_names = None
        if self.atom_names is not None:
            token_atom_names = self.atom_names[start_atom:end_atom]
        token_atom_hetero = None
        if self.atom_hetero is not None:
            token_atom_hetero = self.atom_hetero[start_atom:end_atom]

        # Default values for residue/molecule type (would be extended based on actual implementation)
        residue_type = 0  # Default to standard residue
        molecule_type = 0  # Default to protein

        return Molecule(
            token=token,
            token_idx=idx,
            atom_positions=token_atom_positions,
            atom_elements=token_atom_elements,
            atom_names=token_atom_names,
            atom_hetero=token_atom_hetero,
            residue_type=residue_type,
            molecule_type=molecule_type,
            confidence=self.plddt[idx],
        )

    @property
    def atom_coordinates(self) -> np.ndarray:
        """Get flat array of all atom coordinates [N_atoms, 3]."""
        return self.atom_positions

    # Conversion methods
    @classmethod
    def from_protein_complex(cls, pc: ProteinComplex) -> "MolecularComplex":
        """Convert a ProteinComplex to MolecularComplex.

        Args:
            pc: ProteinComplex object with atom37 representation

        Returns:
            MolecularComplex with flat atom arrays and token-based indexing
        """
        from esm.utils import residue_constants

        # Extract sequence without chain breaks
        sequence_no_breaks = pc.sequence.replace("|", "")
        sequence_tokens = [
            residue_constants.restype_1to3.get(aa, "UNK") for aa in sequence_no_breaks
        ]

        # Convert atom37 to flat arrays
        flat_positions = []
        flat_elements = []
        flat_names = []
        flat_hetero = []
        token_to_atoms = []

        atom_idx = 0

        for i, aa in enumerate(pc.sequence):
            if aa == "|":
                # Skip chain break tokens
                continue

            # Get atom37 positions and mask for this residue.
            # ProteinComplex arrays are indexed by sequence position (including |),
            # so use `i` not a separate residue counter.
            res_positions = pc.atom37_positions[i]  # [37, 3]
            res_mask = pc.atom37_mask[i]  # [37]

            # Track start position for this token
            token_start = atom_idx

            # Process each atom type in atom37 representation
            for atom_type_idx, atom_name in enumerate(residue_constants.atom_types):
                if res_mask[atom_type_idx]:  # Atom is present
                    # Add position
                    flat_positions.append(res_positions[atom_type_idx])

                    # Determine element from atom name
                    element = (
                        atom_name[0] if atom_name else "C"
                    )  # First character is element
                    flat_elements.append(element)

                    # Add atom name
                    flat_names.append(atom_name)

                    # Add hetero flag (all proteins are non-hetero)
                    flat_hetero.append(False)

                    atom_idx += 1

            # Record token-to-atom mapping [start_idx, end_idx)
            token_to_atoms.append([token_start, atom_idx])

        # Convert to numpy arrays
        atom_positions = np.array(flat_positions, dtype=np.float32)
        atom_elements = np.array(flat_elements, dtype=object)
        atom_names = np.array(flat_names, dtype=object)
        atom_hetero = np.array(flat_hetero, dtype=bool)
        token_to_atoms_array = np.array(token_to_atoms, dtype=np.int32)

        # Extract confidence scores and chain_ids (skip chain breaks)
        confidence_scores = []
        chain_ids = []
        for seq_idx, aa in enumerate(pc.sequence):
            if aa != "|":
                confidence_scores.append(pc.confidence[seq_idx])
                chain_ids.append(pc.chain_id[seq_idx])

        confidence_array = np.array(confidence_scores, dtype=np.float32)
        chain_id_array = np.array(chain_ids, dtype=np.int64)

        # Create metadata - convert entity IDs to strings for MolecularComplexMetadata
        entity_lookup_str = {k: str(v) for k, v in pc.metadata.entity_lookup.items()}
        metadata = MolecularComplexMetadata(
            entity_lookup=entity_lookup_str,
            chain_lookup=pc.metadata.chain_lookup,
            assembly_composition=pc.metadata.assembly_composition,
        )

        return cls(
            id=pc.id,
            sequence=sequence_tokens,
            atom_positions=atom_positions,
            atom_elements=atom_elements,
            token_to_atoms=token_to_atoms_array,
            chain_id=chain_id_array,
            plddt=confidence_array,
            metadata=metadata,
            atom_names=atom_names,
            atom_hetero=atom_hetero,
        )

    def to_protein_complex(self) -> ProteinComplex:
        """Convert MolecularComplex back to ProteinComplex format.

        Extracts only protein tokens and converts from flat atom representation
        back to atom37 format used by ProteinComplex.

        Returns:
            ProteinComplex with protein residues only, excluding ligands/nucleic acids
        """
        from esm.utils import residue_constants

        # No need for element mapping - already using element characters

        # Filter for protein tokens only (skip ligands, nucleic acids)
        protein_tokens = []
        protein_indices = []

        for i, token in enumerate(self.sequence):
            # Check if token is a standard 3-letter amino acid code
            if token in residue_constants.restype_3to1:
                protein_tokens.append(token)
                protein_indices.append(i)

        if not protein_tokens:
            raise ValueError("No protein tokens found in MolecularComplex")

        n_residues = len(protein_tokens)

        # Initialize atom37 arrays
        atom37_positions = np.full((n_residues, 37, 3), np.nan, dtype=np.float32)
        atom37_mask = np.zeros((n_residues, 37), dtype=bool)

        # Extract confidence scores and chain_ids for protein residues only
        protein_confidence = self.plddt[protein_indices]
        protein_chain_ids = self.chain_id[protein_indices]

        # Convert tokens back to single-letter sequence with chain breaks
        single_letter_residues = []
        prev_chain_id = None

        for i, (token, chain_id_val) in enumerate(
            zip(protein_tokens, protein_chain_ids)
        ):
            # Add chain break if we're switching to a new chain
            if prev_chain_id is not None and chain_id_val != prev_chain_id:
                single_letter_residues.append("|")
            single_letter_residues.append(residue_constants.restype_3to1[token])
            prev_chain_id = chain_id_val

        single_letter_sequence = "".join(single_letter_residues)

        # Calculate final sequence length (includes chain breaks)
        sequence_length = len(single_letter_sequence)

        # Convert flat atoms back to atom37 representation using atom names
        for res_idx, token_idx in enumerate(protein_indices):
            token = self.sequence[token_idx]
            start_atom, end_atom = self.token_to_atoms[token_idx]

            res_atom_positions = self.atom_positions[start_atom:end_atom]
            res_atom_names = (
                np.array(self.atom_names[start_atom:end_atom], dtype=str)
                if self.atom_names is not None
                else np.array([], dtype=str)
            )

            # Build a mapping from normalized atom name -> position for this residue
            # Normalize to uppercase and strip whitespace for robust matching
            name_to_pos: dict[str, np.ndarray] = {}
            for i, nm in enumerate(res_atom_names):
                key = nm.upper().strip()
                # Prefer first occurrence; ignore duplicates/altlocs
                if key not in name_to_pos:
                    name_to_pos[key] = res_atom_positions[i]

            # Place atoms into atom37 by matching stored atom names to atom37 indices.
            # This handles all atoms present in the flat representation, not just
            # the canonical residue_atoms for this residue type. This preserves
            # atoms that were in the original atom37_mask even if they're atypical
            # for the residue (e.g., from alternate conformations or data quirks).
            for atom_name_str, pos in name_to_pos.items():
                idx37 = residue_constants.atom_order.get(atom_name_str)
                if idx37 is not None:
                    atom37_positions[res_idx, idx37] = pos
                    atom37_mask[res_idx, idx37] = True

        # Create arrays that match sequence length (including chain breaks)
        # Initialize arrays with proper size
        chain_id_expanded = np.full(sequence_length, -1, dtype=np.int64)
        entity_id_expanded = np.full(sequence_length, -1, dtype=np.int64)
        sym_id_expanded = np.zeros(sequence_length, dtype=np.int64)
        residue_index_expanded = np.zeros(sequence_length, dtype=np.int64)
        insertion_code_expanded = np.array([""] * sequence_length, dtype=object)
        confidence_expanded = np.zeros(sequence_length, dtype=np.float32)
        atom37_positions_expanded = np.full(
            (sequence_length, 37, 3), np.nan, dtype=np.float32
        )
        atom37_mask_expanded = np.zeros((sequence_length, 37), dtype=bool)

        # Map residue data to sequence positions (skipping chain breaks)
        residue_idx = 0
        residue_counter_per_chain = {}

        for seq_pos, char in enumerate(single_letter_sequence):
            if char != "|":
                # This is a residue position
                chain_id_val = protein_chain_ids[residue_idx]

                chain_id_expanded[seq_pos] = chain_id_val
                entity_id_expanded[seq_pos] = chain_id_val  # Simplified mapping

                # Track residue numbering per chain
                if chain_id_val not in residue_counter_per_chain:
                    residue_counter_per_chain[chain_id_val] = 1
                else:
                    residue_counter_per_chain[chain_id_val] += 1

                residue_index_expanded[seq_pos] = residue_counter_per_chain[
                    chain_id_val
                ]
                confidence_expanded[seq_pos] = protein_confidence[residue_idx]
                atom37_positions_expanded[seq_pos] = atom37_positions[residue_idx]
                atom37_mask_expanded[seq_pos] = atom37_mask[residue_idx]

                residue_idx += 1
            # Chain break positions keep default values (-1, False, etc.)

        # Use the expanded arrays
        chain_id = chain_id_expanded
        entity_id = entity_id_expanded
        sym_id = sym_id_expanded
        residue_index = residue_index_expanded
        insertion_code = insertion_code_expanded
        protein_confidence = confidence_expanded
        atom37_positions = atom37_positions_expanded
        atom37_mask = atom37_mask_expanded

        # Create protein complex metadata preserving chain information
        # Convert MolecularComplex metadata to ProteinComplex format
        unique_chain_ids = np.unique(protein_chain_ids)
        entity_lookup = {int(cid): int(cid) for cid in unique_chain_ids}
        chain_lookup = {
            int(cid): self.metadata.chain_lookup.get(int(cid), chr(65 + int(cid)))
            for cid in unique_chain_ids
        }

        protein_metadata = ProteinComplexMetadata(
            entity_lookup=entity_lookup,
            chain_lookup=chain_lookup,
            assembly_composition=self.metadata.assembly_composition,
        )

        return ProteinComplex(
            id=self.id,
            sequence=single_letter_sequence,
            entity_id=entity_id,
            chain_id=chain_id,
            sym_id=sym_id,
            residue_index=residue_index,
            insertion_code=insertion_code,
            atom37_positions=atom37_positions,
            atom37_mask=atom37_mask,
            confidence=protein_confidence,
            metadata=protein_metadata,
        )

    @classmethod
    def from_mmcif(cls, inp: str, id: str | None = None) -> "MolecularComplex":
        """Read MolecularComplex from mmcif file or string.

        Args:
            inp: Path to mmCIF file or mmCIF content as string
            id: Optional identifier to assign to the complex

        Returns:
            MolecularComplex with all molecules (proteins, ligands, nucleic acids)
        """
        from io import StringIO

        # Check if input is a file path or mmCIF string content
        if os.path.exists(inp):
            # Input is a file path
            mmcif_file = pdbx.CIFFile.read(inp)
        else:
            # Input is mmCIF string content
            mmcif_file = pdbx.CIFFile.read(StringIO(inp))

        # Get structure - handle missing model information gracefully
        try:
            structure = pdbx.get_structure(
                mmcif_file, model=1, extra_fields=["b_factor"]
            )
        except (KeyError, ValueError):
            # Fallback for mmCIF files without model information
            try:
                structure = pdbx.get_structure(mmcif_file)
            except Exception:
                # Last resort: use the first available model or all atoms
                structure = pdbx.get_structure(mmcif_file, model=None)
        # Type hint for pyright - structure is an AtomArray which is iterable
        if TYPE_CHECKING:
            structure: Any = structure

        # Read label_asym_id from the raw CIF atom_site category.
        # Biotite's atom.chain_id uses auth_asym_id, which collapses ligands
        # onto their parent protein chain. label_asym_id gives each entity a
        # distinct chain identifier.
        block = mmcif_file.block
        label_asym_ids: list[str] | None = None
        if "atom_site" in block:
            atom_site = block["atom_site"]
            if "label_asym_id" in atom_site:
                _col = atom_site["label_asym_id"]
                _raw = (
                    _col.as_array(str)
                    if hasattr(_col, "as_array")
                    else np.array(list(_col), dtype=str)  # type: ignore[arg-type]
                )
                # biotite's get_structure(model=1) filters to model 1 AND
                # removes alternate conformations. We must apply the same
                # filters to label_asym_id to keep arrays aligned.
                keep = np.ones(len(_raw), dtype=bool)
                if "pdbx_PDB_model_num" in atom_site:
                    _mc = atom_site["pdbx_PDB_model_num"]
                    _models = (
                        _mc.as_array(str)
                        if hasattr(_mc, "as_array")
                        else np.array(list(_mc), dtype=str)  # type: ignore[arg-type]
                    )
                    keep &= _models == "1"
                if "label_alt_id" in atom_site:
                    _ac = atom_site["label_alt_id"]
                    _alts = (
                        _ac.as_array(str)
                        if hasattr(_ac, "as_array")
                        else np.array(list(_ac), dtype=str)  # type: ignore[arg-type]
                    )
                    keep &= np.isin(_alts, [".", "?", "", "A"])
                filtered = _raw[keep]
                if len(filtered) == len(structure):
                    label_asym_ids = filtered.tolist()
                # If lengths still don't match, fall back to atom.chain_id

        # Get entity information from mmCIF
        entity_info = {}
        try:
            if "entity" in block:
                entity_category = block["entity"]
                if "id" in entity_category and "type" in entity_category:
                    entity_ids = entity_category["id"]
                    entity_types = entity_category["type"]
                    # Convert CIFColumn to list for iteration
                    if hasattr(entity_ids, "__iter__") and hasattr(
                        entity_types, "__iter__"
                    ):
                        # Type annotation to help pyright understand these are iterable
                        entity_ids_list = list(entity_ids)  # type: ignore
                        entity_types_list = list(entity_types)  # type: ignore
                        for eid, etype in zip(entity_ids_list, entity_types_list):
                            entity_info[eid] = etype
        except Exception:
            pass

        # Initialize arrays for flat atom representation
        sequence_tokens = []
        flat_positions = []
        flat_elements = []
        flat_names = []
        flat_hetero = []
        token_to_atoms = []
        confidence_scores = []
        chain_ids = []  # Track chain IDs for each token

        atom_idx = 0

        # Group atoms by chain and residue.
        # Use label_asym_id (distinct per entity) when available, otherwise
        # fall back to biotite's chain_id (auth_asym_id).
        chain_residue_groups: dict[str, dict[tuple[int, str], dict]] = {}
        for atom_i, atom in enumerate(structure):
            chain_id = (
                label_asym_ids[atom_i] if label_asym_ids is not None else atom.chain_id
            )
            res_id = atom.res_id
            res_name = atom.res_name

            if chain_id not in chain_residue_groups:
                chain_residue_groups[chain_id] = {}
            # Key by (res_id, res_name) to distinguish residues that share
            # the same res_id but have different res_name (e.g. a protein
            # residue and a ligand that were on the same auth chain).
            res_key = (res_id, res_name)
            if res_key not in chain_residue_groups[chain_id]:
                chain_residue_groups[chain_id][res_key] = {
                    "atoms": [],
                    "res_name": res_name,
                    "is_hetero": atom.hetero,
                }
            chain_residue_groups[chain_id][res_key]["atoms"].append(atom)

        # Create a mapping from chain_id to numeric indices
        chain_id_to_numeric = {
            chain_id: idx
            for idx, chain_id in enumerate(sorted(chain_residue_groups.keys()))
        }

        # Process each chain and residue
        for chain_id in sorted(chain_residue_groups.keys()):
            residues = chain_residue_groups[chain_id]
            numeric_chain_id = chain_id_to_numeric[chain_id]

            for res_key in sorted(residues.keys()):
                residue_data = residues[res_key]
                res_name = residue_data["res_name"]
                atoms = residue_data["atoms"]
                is_hetero = residue_data["is_hetero"]

                # Skip water molecules
                if res_name == "HOH":
                    continue

                # Determine token name
                if not is_hetero and res_name in residue_constants.restype_3to1:
                    # Standard amino acid
                    token_name = res_name
                elif res_name in ["A", "T", "G", "C", "U", "DA", "DT", "DG", "DC"]:
                    # Nucleotide
                    token_name = res_name
                else:
                    # Ligand or other molecule
                    token_name = res_name

                sequence_tokens.append(token_name)
                chain_ids.append(
                    numeric_chain_id
                )  # Store the numeric chain ID for this token
                token_start = atom_idx

                # Add all atoms from this residue
                for atom in atoms:
                    flat_positions.append(atom.coord)

                    # Get element character
                    element = atom.element
                    flat_elements.append(element)

                    # Get atom name
                    atom_name = atom.atom_name
                    flat_names.append(atom_name)

                    # Get hetero flag
                    hetero_flag = atom.hetero
                    flat_hetero.append(hetero_flag)

                    atom_idx += 1

                # Record token-to-atom mapping
                token_to_atoms.append([token_start, atom_idx])

                # Add confidence score (B-factor if available, otherwise 1.0)
                bfactor = getattr(atoms[0], "b_factor", 50.0) if atoms else 50.0
                confidence_scores.append(min(bfactor / 100.0, 1.0))

        # Convert to numpy arrays
        if not flat_positions:
            # Create minimal arrays if no atoms found
            atom_positions = np.zeros((0, 3), dtype=np.float32)
            atom_elements = np.zeros(0, dtype=object)
            atom_names = np.zeros(0, dtype=object)
            atom_hetero = np.zeros(0, dtype=bool)
            token_to_atoms_array = np.zeros((len(sequence_tokens), 2), dtype=np.int32)
            chain_id_array = (
                np.array(chain_ids, dtype=np.int64)
                if chain_ids
                else np.zeros(len(sequence_tokens), dtype=np.int64)
            )
        else:
            atom_positions = np.array(flat_positions, dtype=np.float32)
            atom_elements = np.array(flat_elements, dtype=object)
            atom_names = np.array(flat_names, dtype=object)
            atom_hetero = np.array(flat_hetero, dtype=bool)
            token_to_atoms_array = np.array(token_to_atoms, dtype=np.int32)
            chain_id_array = np.array(chain_ids, dtype=np.int64)

        confidence_array = np.array(confidence_scores, dtype=np.float32)

        # Create metadata using the chain_id_to_numeric mapping
        if chain_residue_groups:
            chain_lookup = {
                numeric_id: chain_id
                for chain_id, numeric_id in chain_id_to_numeric.items()
            }
        else:
            chain_lookup = {}

        metadata = MolecularComplexMetadata(
            entity_lookup=entity_info,
            chain_lookup=chain_lookup,
            assembly_composition=None,
        )

        # Set complex ID - if input was a path, use the stem; otherwise use default
        if os.path.exists(inp):
            complex_id = id or Path(inp).stem
        else:
            complex_id = id or "complex_from_string"

        return cls(
            id=complex_id,
            sequence=sequence_tokens,
            atom_positions=atom_positions,
            atom_elements=atom_elements,
            token_to_atoms=token_to_atoms_array,
            chain_id=chain_id_array,
            plddt=confidence_array,
            metadata=metadata,
            atom_names=atom_names,
            atom_hetero=atom_hetero,
        )

    def _get_entity_mapping(
        self,
    ) -> tuple[dict[str, list[str]], dict[str, int], dict[int, tuple[str, ...]]]:
        """Compute chain→sequence, chain→entity_id, and entity_id→sequence mappings.

        Returns:
            (chain_sequences, chain_to_entity, entity_sequences)
        """
        chain_sequences: dict[str, list[str]] = {}
        for token_idx in range(len(self.token_to_atoms)):
            chain_id_numeric = self.chain_id[token_idx]
            chain_id_str = self.metadata.chain_lookup.get(
                int(chain_id_numeric), chr(65 + int(chain_id_numeric))
            )
            if chain_id_str not in chain_sequences:
                chain_sequences[chain_id_str] = []
            chain_sequences[chain_id_str].append(self.sequence[token_idx])

        sequence_to_entity: dict[tuple[str, ...], int] = {}
        chain_to_entity: dict[str, int] = {}
        entity_sequences: dict[int, tuple[str, ...]] = {}
        entity_id_counter = 1
        for chain_id_str, sequence in chain_sequences.items():
            seq_tuple = tuple(sequence)
            if seq_tuple not in sequence_to_entity:
                sequence_to_entity[seq_tuple] = entity_id_counter
                entity_sequences[entity_id_counter] = seq_tuple
                entity_id_counter += 1
            chain_to_entity[chain_id_str] = sequence_to_entity[seq_tuple]

        return chain_sequences, chain_to_entity, entity_sequences

    def _add_entity_information(
        self, cif_file: CIFFile, entity_sequences: dict[int, tuple[str, ...]]
    ) -> None:
        """Add _entity category to CIF file so OST can identify ligands vs polymers."""

        entity_ids: list[str] = []
        entity_types: list[str] = []
        entity_descriptions: list[str] = []
        for eid in sorted(entity_sequences.keys()):
            seq = entity_sequences[eid]
            entity_ids.append(str(eid))
            has_protein = any(t in residue_constants.restype_3to1 for t in seq)
            has_na = any(
                t in ("A", "T", "G", "C", "U", "DA", "DT", "DG", "DC") for t in seq
            )
            if has_protein or has_na:
                entity_types.append("polymer")
                if has_protein:
                    entity_descriptions.append(f"Polymer entity {eid} (protein)")
                else:
                    entity_descriptions.append(f"Polymer entity {eid} (nucleic acid)")
            else:
                entity_types.append("non-polymer")
                entity_descriptions.append(f"Non-polymer entity {eid}")

        if entity_ids:
            cif_file.block["entity"] = CIFCategory(
                name="entity",
                columns={
                    "id": CIFColumn(
                        data=CIFData(array=np.array(entity_ids), dtype=np.str_)
                    ),
                    "type": CIFColumn(
                        data=CIFData(array=np.array(entity_types), dtype=np.str_)
                    ),
                    "pdbx_description": CIFColumn(
                        data=CIFData(array=np.array(entity_descriptions), dtype=np.str_)
                    ),
                },
            )

        # Add _struct_asym to map chain IDs to entity IDs
        _, chain_to_entity, _ = self._get_entity_mapping()
        if chain_to_entity:
            asym_ids = sorted(chain_to_entity.keys())
            asym_entity_ids = [str(chain_to_entity[c]) for c in asym_ids]
            cif_file.block["struct_asym"] = CIFCategory(
                name="struct_asym",
                columns={
                    "id": CIFColumn(
                        data=CIFData(array=np.array(asym_ids), dtype=np.str_)
                    ),
                    "entity_id": CIFColumn(
                        data=CIFData(array=np.array(asym_entity_ids), dtype=np.str_)
                    ),
                },
            )

    def to_mmcif(self) -> str:
        """Write MolecularComplex to mmcif string using biotite.

        Returns:
            String representation of the complex in mmCIF format
        """
        # Pre-allocate AtomArray
        n_atoms = len(self.atom_positions)
        atom_array = bs.AtomArray(length=n_atoms)

        # Set coordinates directly (already vectorized)
        atom_array.coord = self.atom_positions

        # Pre-allocate per-atom arrays
        atom_res_ids = np.zeros(n_atoms, dtype=np.int32)
        atom_chain_ids = np.empty(n_atoms, dtype=object)
        atom_res_names = np.empty(n_atoms, dtype=object)
        atom_hetero = np.zeros(n_atoms, dtype=bool)
        atom_bfactors = np.zeros(n_atoms, dtype=np.float32)
        atom_names = np.empty(n_atoms, dtype=object)

        # Build entity mappings: chains with identical sequences share entity ID
        _, chain_to_entity, entity_sequences = self._get_entity_mapping()

        atom_entity_ids = np.zeros(n_atoms, dtype=np.int32)

        # Track residue IDs per chain
        chain_res_counters: dict[int, int] = {}

        # Vectorized expansion of token-level to atom-level annotations
        for token_idx, (start, end) in enumerate(self.token_to_atoms):
            token = self.sequence[token_idx]
            chain_id_numeric = self.chain_id[token_idx]
            chain_id_str = self.metadata.chain_lookup.get(
                int(chain_id_numeric), chr(65 + int(chain_id_numeric))
            )

            # Track residue numbering per chain
            if chain_id_numeric not in chain_res_counters:
                chain_res_counters[chain_id_numeric] = 1
            res_id = chain_res_counters[chain_id_numeric]
            chain_res_counters[chain_id_numeric] += 1

            # Determine if protein
            is_protein = token in residue_constants.restype_3to1

            # Get atom names for this residue
            if self.atom_names is not None:
                # Use stored atom names (preserves original names from mmCIF)
                names = list(self.atom_names[start:end])
            elif is_protein:
                # Fallback: use standard protein atom names
                standard_names = residue_constants.residue_atoms.get(
                    token, ["N", "CA", "C", "O"]
                )
                names = standard_names[: end - start]
                # Pad if needed
                while len(names) < (end - start):
                    names.append(f"X{len(names)+1}")
            else:
                # Fallback: generate names for ligands/nucleic acids
                names = [f"C{i+1}" for i in range(end - start)]

            # Vectorized assignment for this token's atoms
            atom_res_ids[start:end] = res_id
            atom_chain_ids[start:end] = chain_id_str
            atom_res_names[start:end] = token
            # Use stored hetero flags if available, otherwise guess based on protein status
            if self.atom_hetero is not None:
                atom_hetero[start:end] = self.atom_hetero[start:end]
            else:
                atom_hetero[start:end] = not is_protein
            atom_bfactors[start:end] = self.plddt[token_idx] * 100.0
            atom_names[start:end] = names
            atom_entity_ids[start:end] = chain_to_entity.get(chain_id_str, 1)

        # Set all AtomArray attributes at once (convert object arrays to proper string arrays)
        # res_name uses U8 to accommodate CCD codes up to 5 characters (e.g., A1AZ2);
        # chain_id uses U16 because chain names like ``ligand_1`` / ``ligand_2`` /
        # auth-asym ids of arbitrary length are possible.
        atom_array.res_id = atom_res_ids
        atom_array.chain_id = np.array(atom_chain_ids, dtype="U16")
        atom_array.res_name = np.array(atom_res_names, dtype="U8")
        atom_array.hetero = atom_hetero
        atom_array.atom_name = np.array(atom_names, dtype="U4")
        atom_array.add_annotation("b_factor", dtype=float)
        atom_array.b_factor = atom_bfactors
        atom_array.add_annotation("entity_id", dtype=int)
        atom_array.entity_id = atom_entity_ids

        # Use existing elements or infer them from atom names
        if self.atom_elements is not None and len(self.atom_elements) == n_atoms:
            # Convert object array to proper string array for biotite
            atom_array.element = np.array(self.atom_elements, dtype="U4")
        else:
            # Use biotite's built-in element inference
            atom_array.element = bs.infer_elements(atom_array)

        # Create CIF file and set structure
        cif_file = CIFFile()
        set_structure(cif_file, atom_array, data_block=self.id)

        # Manually fix label_entity_id (biotite doesn't use entity_id annotation correctly)
        if "atom_site" in cif_file.block:
            atom_site = cif_file.block["atom_site"]
            if "label_asym_id" in atom_site and "label_entity_id" in atom_site:
                label_asym_ids = atom_site["label_asym_id"]
                if hasattr(label_asym_ids, "as_array"):
                    chain_ids_list = label_asym_ids.as_array(str).tolist()
                elif hasattr(label_asym_ids, "__iter__"):
                    chain_ids_list = list(label_asym_ids)  # type: ignore[arg-type]
                else:
                    chain_ids_list = []
                updated_entity_ids = [
                    str(chain_to_entity.get(cid, 1)) for cid in chain_ids_list
                ]
                if updated_entity_ids:
                    atom_site["label_entity_id"] = CIFColumn(
                        data=CIFData(array=np.array(updated_entity_ids), dtype=np.str_)
                    )

        # Add _entity category for OST compatibility
        self._add_entity_information(cif_file, entity_sequences)

        # Convert to string
        output = io.StringIO()
        cif_file.write(output)
        return output.getvalue()

    def dockq(self, native: "MolecularComplex") -> Any:
        """Compute DockQ score against native structure.

        Args:
            native: Native MolecularComplex to compute DockQ against

        Returns:
            DockQ result containing score and alignment information
        """
        # Imports moved to top of file

        # Convert both complexes to ProteinComplex format for DockQ computation
        # This extracts only the protein portion and converts to PDB format
        try:
            self_pc = self.to_protein_complex()
            native_pc = native.to_protein_complex()
        except ValueError as e:
            raise ValueError(
                f"Cannot convert MolecularComplex to ProteinComplex for DockQ: {e}"
            )

        # Normalize chain IDs for PDB compatibility
        self_pc = self_pc.normalize_chain_ids_for_pdb()
        native_pc = native_pc.normalize_chain_ids_for_pdb()

        # Use the existing ProteinComplex.dockq() method
        try:
            dockq_result = self_pc.dockq(native_pc)
            return dockq_result
        except Exception:
            # Fallback to manual DockQ computation if ProteinComplex.dockq() fails
            return self._compute_dockq_manual(native)

    def _compute_dockq_manual(self, native: "MolecularComplex") -> Any:
        """Manual DockQ computation fallback."""
        # Imports moved to top of file

        # Convert both complexes to ProteinComplex format
        try:
            self_pc = self.to_protein_complex()
            native_pc = native.to_protein_complex()
        except ValueError as e:
            raise ValueError(
                f"Cannot convert MolecularComplex to ProteinComplex for DockQ: {e}"
            )

        # Normalize chain IDs for PDB compatibility
        self_pc = self_pc.normalize_chain_ids_for_pdb()
        native_pc = native_pc.normalize_chain_ids_for_pdb()

        # Write temporary PDB files and run DockQ
        with TemporaryDirectory() as tdir:
            dir_path = Path(tdir)
            self_pdb = dir_path / "self.pdb"
            native_pdb = dir_path / "native.pdb"

            # Write PDB files
            self_pc.to_pdb(self_pdb)
            native_pc.to_pdb(native_pdb)

            # Run DockQ
            try:
                output = check_output(["DockQ", str(self_pdb), str(native_pdb)])
                output_text = output.decode()

                # Parse DockQ output
                lines = output_text.split("\n")

                # Find the total DockQ score
                dockq_score = None
                for line in lines:
                    if "Total DockQ" in line:
                        match = re.search(r"Total DockQ.*: ([\d.]+)", line)
                        if match:
                            dockq_score = float(match.group(1))
                            break

                if dockq_score is None:
                    # Try to find individual DockQ scores
                    for line in lines:
                        if line.startswith("DockQ") and ":" in line:
                            try:
                                dockq_score = float(line.split(":")[1].strip())
                                break
                            except (ValueError, IndexError):
                                continue

                if dockq_score is None:
                    raise ValueError("Could not parse DockQ score from output")

                # Return a simple result structure
                return {
                    "total_dockq": dockq_score,
                    "raw_output": output_text,
                    "aligned": self,  # Return self as aligned structure
                }

            except FileNotFoundError:
                raise RuntimeError(
                    "DockQ is not installed. Please install DockQ to use this method."
                )
            except Exception as e:
                raise RuntimeError(f"DockQ computation failed: {e}")

    def rmsd(self, target: "MolecularComplex", **kwargs) -> float:
        """Compute RMSD against target structure.

        Args:
            target: Target MolecularComplex to compute RMSD against
            **kwargs: Additional arguments passed to compute_rmsd

        Returns:
            float: RMSD value between the two structures
        """
        # Imports moved to top of file

        # Ensure both complexes have the same number of tokens
        if len(self) != len(target):
            raise ValueError(
                f"Complexes must have the same number of tokens: {len(self)} vs {len(target)}"
            )

        # Extract center positions for each token (using centroid of atoms)
        mobile_coords = []
        target_coords = []
        atom_mask = []

        for i in range(len(self)):
            # Get atom positions for this token
            mobile_start, mobile_end = self.token_to_atoms[i]
            target_start, target_end = target.token_to_atoms[i]

            # Extract atom positions
            mobile_atoms = self.atom_positions[mobile_start:mobile_end]
            target_atoms = target.atom_positions[target_start:target_end]

            # Check if both tokens have atoms
            if len(mobile_atoms) == 0 or len(target_atoms) == 0:
                # Skip tokens with no atoms
                continue

            # For simplicity, use the centroid of atoms as the representative position
            mobile_center = mobile_atoms.mean(axis=0)
            target_center = target_atoms.mean(axis=0)

            mobile_coords.append(mobile_center)
            target_coords.append(target_center)
            atom_mask.append(True)

        if len(mobile_coords) == 0:
            raise ValueError("No valid atoms found for RMSD computation")

        # Convert to tensors
        mobile_tensor = torch.from_numpy(np.stack(mobile_coords, axis=0)).unsqueeze(
            0
        )  # [1, N, 3]
        target_tensor = torch.from_numpy(np.stack(target_coords, axis=0)).unsqueeze(
            0
        )  # [1, N, 3]
        mask_tensor = torch.tensor(atom_mask, dtype=torch.bool).unsqueeze(0)  # [1, N]

        # Compute RMSD using existing infrastructure
        rmsd_value = compute_rmsd(
            mobile=mobile_tensor,
            target=target_tensor,
            atom_exists_mask=mask_tensor,
            reduction="batch",
            **kwargs,
        )

        return float(rmsd_value)

    def lddt_ca(self, target: "MolecularComplex", **kwargs) -> float:
        """Compute LDDT score against target structure.

        Args:
            target: Target MolecularComplex to compute LDDT against
            **kwargs: Additional arguments passed to compute_lddt

        Returns:
            float: LDDT value between the two structures
        """
        # Imports moved to top of file

        # Ensure both complexes have the same number of tokens
        if len(self) != len(target):
            raise ValueError(
                f"Complexes must have the same number of tokens: {len(self)} vs {len(target)}"
            )

        # Extract center positions for each token (using centroid of atoms)
        mobile_coords = []
        target_coords = []
        atom_mask = []

        for i in range(len(self)):
            # Get atom positions for this token
            mobile_start, mobile_end = self.token_to_atoms[i]
            target_start, target_end = target.token_to_atoms[i]

            # Extract atom positions
            mobile_atoms = self.atom_positions[mobile_start:mobile_end]
            target_atoms = target.atom_positions[target_start:target_end]

            # Check if both tokens have atoms
            if len(mobile_atoms) == 0 or len(target_atoms) == 0:
                # Skip tokens with no atoms
                mobile_coords.append(np.full(3, np.nan))
                target_coords.append(np.full(3, np.nan))
                atom_mask.append(False)
                continue

            # For simplicity, use the centroid of atoms as the representative position
            mobile_center = mobile_atoms.mean(axis=0)
            target_center = target_atoms.mean(axis=0)

            mobile_coords.append(mobile_center)
            target_coords.append(target_center)
            atom_mask.append(True)

        if not any(atom_mask):
            raise ValueError("No valid atoms found for LDDT computation")

        # Convert to tensors
        mobile_tensor = torch.from_numpy(np.stack(mobile_coords, axis=0)).unsqueeze(
            0
        )  # [1, N, 3]
        target_tensor = torch.from_numpy(np.stack(target_coords, axis=0)).unsqueeze(
            0
        )  # [1, N, 3]
        mask_tensor = torch.tensor(atom_mask, dtype=torch.bool).unsqueeze(0)  # [1, N]

        # Compute LDDT using existing infrastructure
        lddt_value = compute_lddt(
            all_atom_pred_pos=mobile_tensor,
            all_atom_positions=target_tensor,
            all_atom_mask=mask_tensor,
            per_residue=False,  # Return overall LDDT score
            **kwargs,
        )

        return float(lddt_value)

    def state_dict(self):
        """This state dict is optimized for storage, so it turns things to fp16 whenever
        possible and converts numpy arrays to lists for JSON serialization.
        """
        dct = {k: v for k, v in vars(self).items()}
        for k, v in dct.items():
            if isinstance(v, np.ndarray):
                match v.dtype:
                    case np.int64:
                        dct[k] = v.astype(np.int32).tolist()
                    case np.float64 | np.float32:
                        dct[k] = v.astype(np.float16).tolist()
                    case _:
                        dct[k] = v.tolist()
            elif isinstance(v, MolecularComplexMetadata):
                dct[k] = asdict(v)

        return dct

    def to_blob(self) -> bytes:
        return brotli.compress(msgpack.dumps(self.state_dict()), quality=5)

    @classmethod
    def from_state_dict(cls, dct):
        for k, v in dct.items():
            if isinstance(v, list) and k in [
                "atom_positions",
                "atom_elements",
                "atom_names",
                "atom_hetero",
                "token_to_atoms",
                "chain_id",
                "plddt",
            ]:
                dct[k] = np.array(v)

        for k, v in dct.items():
            if isinstance(v, np.ndarray):
                if k in ["atom_positions", "plddt"]:
                    dct[k] = v.astype(np.float32)
                elif k in ["token_to_atoms", "chain_id"]:
                    dct[k] = (
                        v.astype(np.int32)
                        if k == "token_to_atoms"
                        else v.astype(np.int64)
                    )

        dct["metadata"] = MolecularComplexMetadata(**dct["metadata"])

        # Backward compatibility: if chain_id is missing, create default array
        if "chain_id" not in dct:
            # Default all tokens to chain 0
            dct["chain_id"] = np.zeros(len(dct["sequence"]), dtype=np.int64)

        return cls(**dct)

    @classmethod
    def from_blob(cls, input: Path | str | io.BytesIO | bytes):
        match input:
            case Path() | str():
                bytes = Path(input).read_bytes()
            case io.BytesIO():
                bytes = input.getvalue()
            case _:
                bytes = input
        return cls.from_state_dict(
            msgpack.loads(brotli.decompress(bytes), strict_map_key=False)
        )
