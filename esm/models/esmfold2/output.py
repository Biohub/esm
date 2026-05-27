from itertools import groupby
from typing import Any

import numpy as np
import torch

from esm.models.esmfold2.constants import ELEMENT_NUMBER_TO_SYMBOL, MOL_TYPE_NONPOLYMER
from esm.utils.structure.molecular_complex import (
    MolecularComplex,
    MolecularComplexMetadata,
)


def get_element_symbol(atomic_num: int) -> str:
    return ELEMENT_NUMBER_TO_SYMBOL.get(atomic_num, "X")


def build_molecular_complex_from_features(
    coords: torch.Tensor,
    plddt: torch.Tensor,
    atom_mask: torch.Tensor,
    ref_element: torch.Tensor,
    ref_atom_name_chars: torch.Tensor,
    chain_infos: list,
    complex_id: str,
) -> MolecularComplex:
    """Construct a MolecularComplex from feature-dict tensors and chain metadata.

    Non-polymer chains (ligands) collapse all per-atom tokens into a single
    residue token whose pLDDT is the per-token average and whose hetero flag
    is True.
    """
    mask_np = atom_mask.bool().cpu().numpy()
    coords_np = coords.float().cpu().numpy()
    name_chars_np = ref_atom_name_chars.cpu().numpy()
    elements_np = ref_element.cpu().numpy()
    plddt_np = plddt.float().cpu().numpy()

    sequence_tokens: list[str] = []
    chain_ids_per_token: list[int] = []
    token_to_atoms: list[list[int]] = []
    confidence: list[float] = []
    flat_positions: list[list[float]] = []
    flat_elements: list[str] = []
    flat_names: list[str] = []
    flat_hetero: list[bool] = []

    chain_lookup: dict[int, str] = {}
    entity_info: dict[int, str] = {}
    out_atom_cursor = 0

    for ci in chain_infos:
        chain_lookup[ci.asym_id] = ci.chain_id
        is_nonpolymer = ci.mol_type == MOL_TYPE_NONPOLYMER
        entity_info[ci.entity_id] = "non-polymer" if is_nonpolymer else "polymer"

        if is_nonpolymer:
            residue_name = ci.tokens[0].residue_name if ci.tokens else "LIG"
            sequence_tokens.append(residue_name)
            chain_ids_per_token.append(ci.asym_id)
            avg_plddt = (
                float(np.mean([plddt_np[ti.token_index] for ti in ci.tokens]))
                if ci.tokens
                else 0.0
            )
            confidence.append(avg_plddt)
            token_atom_start = out_atom_cursor
            for ti in ci.tokens:
                for atom_idx in range(ti.atom_start, ti.atom_start + ti.atom_count):
                    if not mask_np[atom_idx]:
                        continue
                    flat_positions.append(coords_np[atom_idx].tolist())
                    flat_elements.append(get_element_symbol(int(elements_np[atom_idx])))
                    chars = name_chars_np[atom_idx]
                    name = "".join(
                        chr(int(c) + 32) for c in chars if int(c) != 0
                    ).strip()
                    flat_names.append(name)
                    flat_hetero.append(True)
                    out_atom_cursor += 1
            token_to_atoms.append([token_atom_start, out_atom_cursor])
            continue

        # Atom-tokenized modified residues (HYP, MSE, ...) span multiple
        # tokens per residue; collapse them back to one mmCIF residue.
        for _residue_index, ti_iter in groupby(
            ci.tokens, key=lambda t: t.residue_index
        ):
            ti_group = list(ti_iter)
            sequence_tokens.append(ti_group[0].residue_name)
            chain_ids_per_token.append(ci.asym_id)
            confidence.append(
                float(np.mean([plddt_np[ti.token_index] for ti in ti_group]))
            )
            token_atom_start = out_atom_cursor
            for ti in ti_group:
                for atom_idx in range(ti.atom_start, ti.atom_start + ti.atom_count):
                    if not mask_np[atom_idx]:
                        continue
                    flat_positions.append(coords_np[atom_idx].tolist())
                    flat_elements.append(get_element_symbol(int(elements_np[atom_idx])))
                    chars = name_chars_np[atom_idx]
                    name = "".join(
                        chr(int(c) + 32) for c in chars if int(c) != 0
                    ).strip()
                    flat_names.append(name)
                    flat_hetero.append(False)
                    out_atom_cursor += 1
            token_to_atoms.append([token_atom_start, out_atom_cursor])

    return MolecularComplex(
        id=complex_id,
        sequence=sequence_tokens,
        atom_positions=np.array(flat_positions, dtype=np.float32).reshape(-1, 3),
        atom_elements=np.array(flat_elements, dtype=object),
        token_to_atoms=np.array(token_to_atoms, dtype=np.int32).reshape(-1, 2),
        chain_id=np.array(chain_ids_per_token, dtype=np.int64),
        plddt=np.array(confidence, dtype=np.float32),
        atom_names=np.array(flat_names, dtype=object),
        atom_hetero=np.array(flat_hetero, dtype=bool),
        metadata=MolecularComplexMetadata(
            entity_lookup=entity_info,
            chain_lookup=chain_lookup,
            assembly_composition=None,
        ),
    )


def build_molecular_complex(
    structure: Any, coords: torch.Tensor, plddt: torch.Tensor, complex_id: str
) -> MolecularComplex:
    """Directly constructs a MolecularComplex from model outputs without intermediate files.

    Args:
        structure: Object with .chains, .residues, .atoms numpy structured arrays.
        coords: [N_atoms, 3] predicted atom coordinates.
        plddt: [N_residues] per-residue confidence scores.
        complex_id: Identifier string for the resulting complex.
    """
    flat_positions = []
    flat_elements = []
    flat_names = []
    flat_hetero = []

    sequence_tokens = []
    token_to_atoms = []
    chain_ids_per_token = []
    confidence_scores = []

    chain_lookup = {}
    entity_info = {}

    global_atom_cursor = 0
    global_res_cursor = 0
    atom_array_idx = 0

    for chain in structure.chains:
        chain_idx_numeric = chain["asym_id"]
        chain_name_str = str(chain["name"])
        mol_type = chain["mol_type"]

        chain_lookup[chain_idx_numeric] = chain_name_str
        entity_info[chain["entity_id"]] = (
            "polymer" if mol_type != MOL_TYPE_NONPOLYMER else "non-polymer"
        )

        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        residues = structure.residues[res_start:res_end]

        for residue in residues:
            res_name = str(residue["name"])

            sequence_tokens.append(res_name)
            chain_ids_per_token.append(chain_idx_numeric)

            score = plddt[global_res_cursor].item()
            confidence_scores.append(score)
            token_start_idx = atom_array_idx

            atom_start = residue["atom_idx"]
            atom_end = residue["atom_idx"] + residue["atom_num"]
            atoms = structure.atoms[atom_start:atom_end]

            for atom in atoms:
                if not atom["is_present"]:
                    continue

                pos = coords[global_atom_cursor].tolist()
                flat_positions.append(pos)

                elem = get_element_symbol(atom["element"].item())
                flat_elements.append(elem)

                raw_name = atom["name"]
                if hasattr(raw_name, "tolist"):
                    raw_name = raw_name.tolist()
                name_str = "".join([chr(c + 32) for c in raw_name if c != 0])
                flat_names.append(name_str)

                flat_hetero.append(mol_type == MOL_TYPE_NONPOLYMER)

                global_atom_cursor += 1
                atom_array_idx += 1

            token_to_atoms.append([token_start_idx, atom_array_idx])
            global_res_cursor += 1

    return MolecularComplex(
        id=complex_id,
        sequence=sequence_tokens,
        atom_positions=np.array(flat_positions, dtype=np.float32),
        atom_elements=np.array(flat_elements, dtype=object),
        token_to_atoms=np.array(token_to_atoms, dtype=np.int32),
        chain_id=np.array(chain_ids_per_token, dtype=np.int64),
        plddt=np.array(confidence_scores, dtype=np.float32),
        atom_names=np.array(flat_names, dtype=object),
        atom_hetero=np.array(flat_hetero, dtype=bool),
        metadata=MolecularComplexMetadata(
            entity_lookup=entity_info,
            chain_lookup=chain_lookup,
            assembly_composition=None,
        ),
    )
