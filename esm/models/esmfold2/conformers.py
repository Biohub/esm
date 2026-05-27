"""CCD conformer loading utilities.

Loads idealized conformer coordinates from a CCD pickle file containing RDKit molecules.
Conformer priority follows AF3 Section 2.8: Computed > Ideal > first available.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from esm.models.esmfold2.constants import RES_TYPE_TO_CCD

if os.environ.get("ESMCFOLD_CCD_PATH"):
    CCD_PICKLE_PATH = Path(os.environ["ESMCFOLD_CCD_PATH"])
else:
    CCD_PICKLE_PATH = None


# Lazily loaded CCD dictionary
_CCD_MOLECULES: dict | None = None

# Caches
_CCD_CONFORMERS: dict[str, dict[str, np.ndarray]] = {}
_CCD_ATOM_CACHE: dict[str, list[tuple[str, str, int]]] = {}
_CCD_BONDS_CACHE: dict[str, list[tuple[str, str]]] = {}
_CCD_LEAVING_ATOMS_CACHE: dict[str, set[str]] = {}
_IDEALIZED_POS_CACHE: dict[tuple[int, str], np.ndarray | None] = {}
_LIGAND_IDEALIZED_POS_CACHE: dict[tuple[str, str], np.ndarray | None] = {}


def load_ccd(cache_dir: Path | str | None = None) -> dict:
    """Load CCD molecules from pickle file, downloading if needed.

    Args:
        cache_dir: Directory to cache the downloaded CCD pickle.
            If None, uses CCD_PICKLE_PATH env var or downloads to ~/.cache/esmcfold/.
    """
    global _CCD_MOLECULES
    if _CCD_MOLECULES is not None:
        return _CCD_MOLECULES

    # Determine pickle path
    if CCD_PICKLE_PATH is not None and CCD_PICKLE_PATH.exists():
        pkl_path = CCD_PICKLE_PATH
    elif cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = cache_dir / "ccd.pkl"
    else:
        try:
            pkl_path = Path(
                hf_hub_download(repo_id="biohub/ESMFold2", filename="ccd.pkl")
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download CCD pickle file from Hugging Face repository: {e}"
            )

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"CCD pickle file not found: {pkl_path}. Please set the ESMCFOLD_CCD_PATH environment variable to the path of a valid CCD pickle file or download the file from the Hugging Face repository."
        )

    print(f"Loading CCD dictionary from {pkl_path}")
    with open(pkl_path, "rb") as f:
        _CCD_MOLECULES = pickle.load(f)

    if _CCD_MOLECULES is None:
        _CCD_MOLECULES = {}

    return _CCD_MOLECULES


def _get_ccd_molecules() -> dict:
    """Get CCD molecules, loading lazily on first call."""
    global _CCD_MOLECULES
    if _CCD_MOLECULES is None:
        return load_ccd()
    return _CCD_MOLECULES


def _get_ccd_mol_with_significant_h(comp_id: str):
    """Get CCD molecule with only chemically significant hydrogens.

    Returns (mol, conformer) tuple or (None, None) if not available.
    """
    ccd = _get_ccd_molecules()
    if comp_id not in ccd:
        return None, None

    mol = ccd[comp_id]
    if mol.GetNumConformers() == 0:
        return None, None

    # Find the "Computed" conformer (RDKit ETKDGv3), fall back to "Ideal"
    conf_idx = 0
    for i, c in enumerate(mol.GetConformers()):
        props = c.GetPropsAsDict()
        if props.get("name") == "Computed":
            conf_idx = i
            break
    else:
        for i, c in enumerate(mol.GetConformers()):
            props = c.GetPropsAsDict()
            if props.get("name") == "Ideal":
                conf_idx = i
                break

    from rdkit import Chem

    mol_no_h = Chem.RemoveHs(mol, sanitize=False)

    if mol_no_h.GetNumConformers() == 0:
        return None, None

    return mol_no_h, mol_no_h.GetConformer(
        min(conf_idx, mol_no_h.GetNumConformers() - 1)
    )


def get_ccd_conformer(comp_id: str) -> dict[str, np.ndarray] | None:
    """Get idealized conformer as dict of atom_name -> position [3].

    Conformer priority: Computed > Ideal > first available.
    """
    if comp_id in _CCD_CONFORMERS:
        cached = _CCD_CONFORMERS[comp_id]
        return cached if cached else None

    mol, conf = _get_ccd_mol_with_significant_h(comp_id)
    if mol is None or conf is None:
        _CCD_CONFORMERS[comp_id] = {}
        return None

    conformer: dict[str, np.ndarray] = {}
    for atom in mol.GetAtoms():
        props = atom.GetPropsAsDict()
        atom_name = props.get("name")
        if not isinstance(atom_name, str) or not atom_name:
            continue
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        conformer[atom_name] = np.array([pos.x, pos.y, pos.z], dtype=np.float32)

    _CCD_CONFORMERS[comp_id] = conformer
    return conformer if conformer else None


def get_idealized_atom_pos(res_type: int, atom_name: str) -> np.ndarray | None:
    """Get idealized position for a standard residue atom.

    Uses res_type index to look up CCD component, then returns position.
    Returns None if not found.
    """
    cache_key = (res_type, atom_name)
    if cache_key in _IDEALIZED_POS_CACHE:
        return _IDEALIZED_POS_CACHE[cache_key]

    comp_id = RES_TYPE_TO_CCD.get(res_type)
    if comp_id:
        ccd_conformer = get_ccd_conformer(comp_id)
        if ccd_conformer and atom_name in ccd_conformer:
            pos = ccd_conformer[atom_name]
            _IDEALIZED_POS_CACHE[cache_key] = pos
            return pos

    _IDEALIZED_POS_CACHE[cache_key] = None
    return None


def get_ligand_idealized_atom_pos(res_name: str, atom_name: str) -> np.ndarray | None:
    """Get idealized position for a ligand/modified residue atom.

    Returns None if not found.
    """
    cache_key = (res_name, atom_name)
    if cache_key in _LIGAND_IDEALIZED_POS_CACHE:
        return _LIGAND_IDEALIZED_POS_CACHE[cache_key]

    ccd_conformer = get_ccd_conformer(res_name)
    if ccd_conformer and atom_name in ccd_conformer:
        pos = ccd_conformer[atom_name]
        _LIGAND_IDEALIZED_POS_CACHE[cache_key] = pos
        return pos

    _LIGAND_IDEALIZED_POS_CACHE[cache_key] = None
    return None


def get_ligand_ccd_atoms_with_charges(
    comp_id: str,
) -> list[tuple[str, str, int]] | None:
    """Get list of (atom_name, element, charge) for a CCD component.

    Uses RDKit RemoveHs(sanitize=False) to keep chemically significant hydrogens.
    Returns None if CCD data not available.
    """
    if comp_id in _CCD_ATOM_CACHE:
        cached = _CCD_ATOM_CACHE[comp_id]
        return cached if cached else None

    mol, _ = _get_ccd_mol_with_significant_h(comp_id)
    if mol is None:
        _CCD_ATOM_CACHE[comp_id] = []
        return None

    atoms: list[tuple[str, str, int]] = []
    for atom in mol.GetAtoms():
        props = atom.GetPropsAsDict()
        atom_name = props.get("name")
        if not isinstance(atom_name, str) or not atom_name:
            continue
        element = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        atoms.append((atom_name, element, charge))

    _CCD_ATOM_CACHE[comp_id] = atoms
    return atoms if atoms else None


def get_ligand_ccd_bonds(comp_id: str) -> list[tuple[str, str]] | None:
    """Get list of (atom1_name, atom2_name) bonds for a CCD component.

    Returns None if CCD data not available.
    """
    if comp_id in _CCD_BONDS_CACHE:
        cached = _CCD_BONDS_CACHE[comp_id]
        return cached if cached else None

    mol, _ = _get_ccd_mol_with_significant_h(comp_id)
    if mol is None:
        _CCD_BONDS_CACHE[comp_id] = []
        return None

    # Get included atom names
    included_atoms = set()
    for atom in mol.GetAtoms():
        props = atom.GetPropsAsDict()
        atom_name = props.get("name")
        if isinstance(atom_name, str) and atom_name:
            included_atoms.add(atom_name)

    bonds: list[tuple[str, str]] = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        n1 = a1.GetPropsAsDict().get("name")
        n2 = a2.GetPropsAsDict().get("name")
        if (
            isinstance(n1, str)
            and isinstance(n2, str)
            and n1
            and n2
            and n1 in included_atoms
            and n2 in included_atoms
        ):
            bonds.append((n1, n2))

    _CCD_BONDS_CACHE[comp_id] = bonds
    return bonds if bonds else None


def get_ccd_leaving_atoms(comp_id: str) -> set[str]:
    """Get set of atom names marked as leaving atoms in CCD.

    Leaving atoms are removed during polymerization (e.g., OP3 in nucleotides).
    """
    if comp_id in _CCD_LEAVING_ATOMS_CACHE:
        return _CCD_LEAVING_ATOMS_CACHE[comp_id]

    ccd = _get_ccd_molecules()
    if comp_id not in ccd:
        _CCD_LEAVING_ATOMS_CACHE[comp_id] = set()
        return set()

    mol = ccd[comp_id]
    leaving_atoms = set()
    for atom in mol.GetAtoms():
        if atom.HasProp("leaving_atom"):
            if atom.GetProp("leaving_atom") == "1":
                name = atom.GetProp("name") if atom.HasProp("name") else ""
                if name:
                    leaving_atoms.add(name)

    _CCD_LEAVING_ATOMS_CACHE[comp_id] = leaving_atoms
    return leaving_atoms
