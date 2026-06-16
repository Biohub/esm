from __future__ import annotations

import functools
import io
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Union

import biotite.structure as bs
import biotite.structure.io.pdbx as pdbx
import numpy as np
from biotite.structure.io.pdbx import CIFColumn, CIFData, CIFFile

from esm.utils import residue_constants

# Define PathOrBuffer for the opensource version
PathOrBuffer = Union[str, os.PathLike, io.StringIO]

# pLDDT/confidence is stored internally on a 0-1 scale, but written to the mmCIF
# B-factor column on the conventional 0-100 scale
PLDDT_B_FACTOR_SCALE = 100.0

_MMCIF_COLUMN_DECIMALS = {"Cartn_x": 3, "Cartn_y": 3, "Cartn_z": 3, "B_iso_or_equiv": 2}


def round_mmcif_columns(cif_file: CIFFile) -> None:
    """Round coordinate and B-factor columns in ``cif_file`` in-place."""
    if "atom_site" not in cif_file.block:
        return
    atom_site = cif_file.block["atom_site"]
    for col_name, decimals in _MMCIF_COLUMN_DECIMALS.items():
        if col_name not in atom_site:
            continue
        column = atom_site[col_name]
        values = column.as_array(np.float64)
        atom_site[col_name] = CIFColumn(
            data=CIFData(
                array=np.array([f"{v:.{decimals}f}" for v in values], dtype=np.str_),
                dtype=np.str_,
            ),
            mask=column.mask,
        )


class NoProteinError(Exception):
    pass


@dataclass
class Residue:
    residue_number: int | None = None
    insertion_code: str = ""
    hetflag: bool = False


@dataclass
class MmcifHeader:
    release_date: datetime | None = None
    resolution: float | None = None
    structure_method: str = "UNKNOWN"


class MmcifWrapper:
    def __init__(self, id: str | None = None):
        self.id: str = id or ""
        self.raw: pdbx.CIFFile | None = None
        self.structure: bs.AtomArray
        self.header: MmcifHeader = MmcifHeader()
        self.entities: dict[int, list[str]] = {}
        self.chain_to_seqres: dict[str, str] = {}
        self.seqres_to_structure: dict[str, dict[int, Residue]] = {}

    @classmethod
    def read(cls, path: PathOrBuffer, id: str | None = None) -> MmcifWrapper:
        obj = cls(id=id)
        obj._load(path)
        return obj

    def _load(self, path: PathOrBuffer, fileid: str | None = None):
        """Load mmCIF data from file."""
        self.raw = pdbx.CIFFile.read(path)

        self._parse_structure()
        self._parse_header()
        self._parse_entities()
        self._parse_sequences()

    def _parse_structure(self):
        """Parse the atomic structure from mmCIF."""
        try:
            structure = pdbx.get_structure(self.raw, model=1)
            if structure is None or not isinstance(structure, bs.AtomArray):
                raise NoProteinError("No structure found in mmCIF file")
            if len(structure) == 0:
                raise NoProteinError("Empty structure in mmCIF file")
            self.structure = structure
        except Exception as e:
            raise ValueError(f"Failed to parse structure: {e}")

    def _parse_header(self):
        """Parse header information from mmCIF."""
        if not self.raw:
            return

        try:
            # Get the first (and usually only) block
            block = self.raw.block

            # Parse release date
            if "pdbx_database_status" in block:
                status_cat = block["pdbx_database_status"]
                if "recvd_initial_deposition_date" in status_cat:
                    date_str = status_cat["recvd_initial_deposition_date"].as_item()
                    if date_str and date_str != "?":
                        try:
                            self.header.release_date = datetime.strptime(
                                date_str, "%Y-%m-%d"
                            )
                        except ValueError:
                            pass

            # Parse resolution
            if "refine" in block:
                refine_cat = block["refine"]
                if "ls_d_res_high" in refine_cat:
                    res_str = refine_cat["ls_d_res_high"].as_item()
                    if res_str and res_str != "?":
                        try:
                            self.header.resolution = float(res_str)
                        except ValueError:
                            pass

            # Parse structure method
            if "exptl" in block:
                exptl_cat = block["exptl"]
                if "method" in exptl_cat:
                    method = exptl_cat["method"].as_item()
                    if method and method != "?":
                        self.header.structure_method = method.upper()

        except Exception:
            # If parsing fails, keep default values
            pass

    def _parse_entities(self):
        """Parse entity information and map to chains."""
        if not self.raw:
            return

        try:
            block = self.raw.block

            # Parse entity information
            if "entity" in block:
                entity_cat = block["entity"]
                entity_ids = entity_cat["id"].as_array(str)
                entity_types = entity_cat["type"].as_array(str)

                # Initialize entities dict with all entities (not just polymers)
                for i, (entity_id, entity_type) in enumerate(
                    zip(entity_ids, entity_types)
                ):
                    self.entities[int(entity_id)] = []

            # Map polymer chains to entities using entity_poly
            if "entity_poly" in block:
                poly_cat = block["entity_poly"]
                entity_ids = poly_cat["entity_id"].as_array(str)
                chain_lists = poly_cat["pdbx_strand_id"].as_array(str)

                for entity_id, chain_list in zip(entity_ids, chain_lists):
                    entity_id = int(entity_id)
                    # Chain list is comma-separated
                    chains = [c.strip() for c in chain_list.split(",") if c.strip()]
                    if entity_id in self.entities:
                        self.entities[entity_id] = chains

            # Map non-polymer chains using struct_asym for entities not covered by entity_poly
            if "struct_asym" in block:
                asym_cat = block["struct_asym"]
                asym_ids = asym_cat["id"].as_array(str)
                entity_ids = asym_cat["entity_id"].as_array(str)

                for asym_id, entity_id in zip(asym_ids, entity_ids):
                    entity_id = int(entity_id)
                    # Only add if entity exists but has no chains yet (non-polymer entities)
                    if entity_id in self.entities and not self.entities[entity_id]:
                        self.entities[entity_id].append(asym_id)

        except Exception:
            # If parsing fails, try to infer from structure
            if (
                self.structure
                and hasattr(self.structure, "chain_id")
                and self.structure.chain_id is not None
                and hasattr(self.structure.chain_id, "__iter__")
            ):
                chain_ids = list(set(self.structure.chain_id))
                self.entities = {1: chain_ids}

    def _parse_sequences(self):
        """Parse sequence information from mmCIF."""
        if not self.raw:
            return

        block = self.raw.block

        # Parse polymer sequences
        if "entity_poly" in block:
            poly_cat = block["entity_poly"]
            entity_ids = poly_cat["entity_id"].as_array(str)
            sequences = poly_cat["pdbx_seq_one_letter_code_can"].as_array(str)
            chain_lists = poly_cat["pdbx_strand_id"].as_array(str)

            for entity_id, sequence, chain_list in zip(
                entity_ids, sequences, chain_lists
            ):
                # Clean up sequence (remove whitespace and newlines)
                clean_seq = "".join(sequence.split())
                chains = [c.strip() for c in chain_list.split(",") if c.strip()]

                for chain_id in chains:
                    self.chain_to_seqres[chain_id] = clean_seq

        # Parse sequence to structure mapping
        if "pdbx_poly_seq_scheme" in block:
            seq_cat = block["pdbx_poly_seq_scheme"]
            asym_ids = seq_cat["asym_id"].as_array(str)  # Internal chain IDs
            seq_positions = seq_cat["seq_id"].as_array(str)
            auth_seq_nums = seq_cat["auth_seq_num"].as_array(str)
            ins_codes = (
                seq_cat["pdb_ins_code"].as_array(str)
                if "pdb_ins_code" in seq_cat
                else [""] * len(asym_ids)
            )
            hetflags = (
                seq_cat["hetflag"].as_array(str)
                if "hetflag" in seq_cat
                else ["N"] * len(asym_ids)
            )

            # Get author chain IDs if available
            auth_chain_ids = (
                seq_cat["pdb_strand_id"].as_array(str)
                if "pdb_strand_id" in seq_cat
                else asym_ids  # Fallback to internal IDs
            )

            # Build mapping from internal chain ID to author chain ID
            asym_to_auth_mapping = {}
            for asym_id, auth_id in zip(asym_ids, auth_chain_ids):
                asym_to_auth_mapping[asym_id] = auth_id

            # Group by internal chain ID first, then map to author chain ID
            chain_data = {}
            for asym_id, seq_pos, auth_seq, ins_code, hetflag in zip(
                asym_ids, seq_positions, auth_seq_nums, ins_codes, hetflags
            ):
                if asym_id not in chain_data:
                    chain_data[asym_id] = {}

                try:
                    seq_index = int(seq_pos) - 1  # Convert to 0-based indexing
                    res_num = int(auth_seq) if auth_seq != "?" else None
                except ValueError:
                    continue

                if res_num is not None:
                    # Convert mmCIF "." and "?" to empty string
                    clean_ins_code = "" if ins_code in [".", "?"] else ins_code
                else:
                    clean_ins_code = ""
                    res_num = None

                is_het = hetflag.upper() == "Y"
                chain_data[asym_id][seq_index] = Residue(
                    residue_number=res_num,
                    insertion_code=clean_ins_code,
                    hetflag=is_het,
                )

            # Handle cases where multiple residues have the same auth_seq_num
            # by adjusting residue numbers to be unique within each chain
            for asym_id, residue_data in chain_data.items():
                # Check if there are duplicate residue numbers in this chain
                positions_with_same_num = {}
                for seq_idx, res_at_pos in residue_data.items():
                    if res_at_pos.residue_number is not None:
                        res_num = res_at_pos.residue_number
                        if res_num not in positions_with_same_num:
                            positions_with_same_num[res_num] = []
                        positions_with_same_num[res_num].append(seq_idx)

                # Fix duplicate residue numbers by making them sequential
                for res_num, seq_indices in positions_with_same_num.items():
                    if len(seq_indices) > 1:
                        # Multiple residues have the same residue number
                        # Make them sequential starting from the original number
                        seq_indices.sort()  # Ensure consistent ordering
                        for i, seq_idx in enumerate(seq_indices):
                            original_pos = residue_data[seq_idx]
                            new_pos = Residue(
                                residue_number=res_num + i,
                                insertion_code=original_pos.insertion_code,
                                hetflag=original_pos.hetflag,
                            )
                            residue_data[seq_idx] = new_pos

            # Create ordered mappings using author chain IDs
            for asym_id in chain_data:
                auth_chain_id = asym_to_auth_mapping.get(asym_id, asym_id)
                if auth_chain_id in self.chain_to_seqres:
                    seq_len = len(self.chain_to_seqres[auth_chain_id])
                    ordered_mapping = {}

                    for i in range(seq_len):
                        if i in chain_data[asym_id]:
                            ordered_mapping[i] = chain_data[asym_id][i]
                        else:
                            # Missing residue - no structure coordinates
                            ordered_mapping[i] = Residue(
                                residue_number=None, insertion_code="", hetflag=False
                            )

                    self.seqres_to_structure[auth_chain_id] = ordered_mapping
                else:
                    # Handle case where auth_chain_id is not in chain_to_seqres
                    # This can happen if the chain is not a polymer or if there's a parsing issue
                    # Create a basic mapping based on the chain_data
                    if chain_data[asym_id]:
                        # Sort by sequence index to create ordered mapping
                        sorted_indices = sorted(chain_data[asym_id].keys())
                        ordered_mapping = {}
                        for i, seq_idx in enumerate(sorted_indices):
                            ordered_mapping[i] = chain_data[asym_id][seq_idx]
                        self.seqres_to_structure[auth_chain_id] = ordered_mapping

        # Ensure all chains have complete mappings
        for chain_id in self.chain_to_seqres:
            if chain_id not in self.seqres_to_structure:
                seq_len = len(self.chain_to_seqres[chain_id])
                self.seqres_to_structure[chain_id] = {
                    i: Residue(residue_number=None, insertion_code="", hetflag=False)
                    for i in range(seq_len)
                }
            else:
                # Fill in any missing indices
                seq_len = len(self.chain_to_seqres[chain_id])
                mapping = self.seqres_to_structure[chain_id]
                for i in range(seq_len):
                    if i not in mapping:
                        mapping[i] = Residue(
                            residue_number=None, insertion_code="", hetflag=False
                        )

        # Fallback: create basic mappings from structure for missing chains
        if (
            self.structure
            and hasattr(self.structure, "chain_id")
            and self.structure.chain_id is not None
            and hasattr(self.structure.chain_id, "__iter__")
        ):
            for chain_id in set(self.structure.chain_id):
                if chain_id not in self.seqres_to_structure:
                    chain_structure = self.structure[
                        self.structure.chain_id == chain_id
                    ]
                    if (
                        hasattr(chain_structure, "res_id")
                        and chain_structure.res_id is not None
                        and hasattr(chain_structure.res_id, "__iter__")
                    ):
                        residue_ids = list(set(chain_structure.res_id))
                        residue_ids.sort()

                        self.seqres_to_structure[chain_id] = {
                            i: Residue(
                                residue_number=res_id, insertion_code="", hetflag=False
                            )
                            for i, res_id in enumerate(residue_ids)
                        }

    def _parse_nonpoly_from_mmcif(self) -> dict[tuple, bs.AtomArray]:
        """Parse non-polymer coordinates from mmCIF block data."""
        nonpoly_coords = {}

        # Get non-polymer entities from the mmCIF block
        assert self.raw is not None
        block = self.raw.block
        nonpoly_entities = set()

        # Find non-polymer entities
        if "entity" in block:
            entity_cat = block["entity"]
            entity_ids = entity_cat["id"].as_array(str)
            entity_types = entity_cat["type"].as_array(str)

            for entity_id, entity_type in zip(entity_ids, entity_types):
                if entity_type.upper() in ["NON-POLYMER", "WATER", "BRANCHED"]:
                    nonpoly_entities.add(entity_id)

        # Map entities to chains for non-polymers
        entity_to_chains = {}
        if "pdbx_entity_nonpoly" in block:
            nonpoly_cat = block["pdbx_entity_nonpoly"]
            entity_ids = nonpoly_cat["entity_id"].as_array(str)
            comp_ids = nonpoly_cat["comp_id"].as_array(str)

            for entity_id, comp_id in zip(entity_ids, comp_ids):
                if entity_id in nonpoly_entities:
                    entity_to_chains[entity_id] = comp_id

        # Get atom site information for non-polymers
        if "atom_site" in block:
            atom_cat = block["atom_site"]
            atom_chain_ids = atom_cat["label_asym_id"].as_array(str)
            atom_entity_ids = atom_cat["label_entity_id"].as_array(str)
            atom_comp_ids = atom_cat["label_comp_id"].as_array(str)

            # Group non-polymer atoms by entity and chain
            nonpoly_atom_groups = {}
            for i, (chain_id, entity_id, comp_id) in enumerate(
                zip(atom_chain_ids, atom_entity_ids, atom_comp_ids)
            ):
                if entity_id in nonpoly_entities:
                    key = (comp_id, chain_id)
                    if key not in nonpoly_atom_groups:
                        nonpoly_atom_groups[key] = []
                    nonpoly_atom_groups[key].append(i)

            # Extract coordinates for each non-polymer group
            for (comp_id, chain_id), atom_indices in nonpoly_atom_groups.items():
                # Match atoms by comparing chain_id and residue name
                structure_mask = (self.structure.chain_id == chain_id) & (
                    self.structure.res_name == comp_id
                )

                if structure_mask.any():
                    nonpoly_array = self.structure[structure_mask]
                    if (
                        isinstance(nonpoly_array, (bs.AtomArray, bs.AtomArrayStack))
                        and len(nonpoly_array) > 0
                    ):
                        nonpoly_coords[(comp_id, chain_id)] = nonpoly_array

        return nonpoly_coords

    def _parse_nonpoly_fallback(self) -> dict[tuple, bs.AtomArray]:
        """Fallback method to extract heteroatoms directly from structure."""
        nonpoly_coords = {}

        if not (self.structure and hasattr(self.structure, "chain_id")):
            return nonpoly_coords

        # Create set of standard residues from residue_constants
        standard_residues = set(residue_constants.resnames[:-1])  # Exclude 'UNK'
        standard_residues.update({"A", "C", "G", "T", "U"})  # Add nucleic acids

        if hasattr(self.structure, "chain_id") and self.structure.chain_id is not None:
            for chain_id in set(self.structure.chain_id):
                chain_structure = self.structure[self.structure.chain_id == chain_id]

                # Find non-standard residues
                if (
                    hasattr(chain_structure, "res_name")
                    and chain_structure.res_name is not None
                    and hasattr(chain_structure.res_name, "__iter__")
                ):
                    for res_name in set(chain_structure.res_name):
                        if res_name not in standard_residues:
                            res_mask = (chain_structure.chain_id == chain_id) & (
                                chain_structure.res_name == res_name
                            )
                            if res_mask.any() and isinstance(
                                chain_structure, (bs.AtomArray, bs.AtomArrayStack)
                            ):
                                nonpoly_array = chain_structure[res_mask]
                                nonpoly_coords[(res_name, chain_id)] = nonpoly_array

        return nonpoly_coords

    @functools.cached_property
    def non_polymer_coords(self) -> dict[tuple, bs.AtomArray]:
        """
        Extract non-polymer coordinates (ligands, cofactors, etc.) from mmCIF structure.

        Returns a dictionary mapping (nonpolymer_info, chain_id) tuples to AtomArrays.
        """
        if not self.structure or not self.raw:
            return {}

        try:
            return self._parse_nonpoly_from_mmcif()
        except Exception:
            return self._parse_nonpoly_fallback()
