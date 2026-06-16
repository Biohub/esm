from __future__ import annotations

from dataclasses import Field, replace
from typing import Any, ClassVar, Protocol, TypeVar

import numpy as np
import torch

from esm.utils.structure.protein_structure import compute_affine_and_rmsd


class Alignable(Protocol):
    # Trick to detect whether an object is a dataclass
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @property
    def atom37_positions(self) -> np.ndarray:
        pass

    @property
    def atom37_mask(self) -> np.ndarray:
        pass

    def __len__(self) -> int: ...


T = TypeVar("T", bound=Alignable)


class Aligner:
    def __init__(
        self,
        mobile: Alignable,
        target: Alignable,
        only_use_backbone: bool = False,
        use_reflection: bool = False,
    ):
        """
        Aligns a mobile protein chain against a target protein chain.

        Args:
            mobile (ProteinChain): Protein chain to be aligned.
            target (ProteinChain): Protein chain target.
            only_use_backbone (bool): Whether to only use backbone atoms.
            use_reflection (bool): Whether to align to target reflection.
        """
        # Check proteins must have same number of residues
        assert len(mobile) == len(target)

        # Determine overlapping atoms
        joint_atom37_mask = mobile.atom37_mask.astype(bool) & target.atom37_mask.astype(
            bool
        )

        # Backbone atoms are first sites in atom37 representation
        if only_use_backbone:
            joint_atom37_mask[:, 3:] = False

        # Extract matching atom positions and convert to batched tensors
        mobile_atom_tensor = (
            torch.from_numpy(mobile.atom37_positions).type(torch.double).unsqueeze(0)
        )
        target_atom_tensor = (
            torch.from_numpy(target.atom37_positions).type(torch.double).unsqueeze(0)
        )
        joint_atom37_mask = (
            torch.from_numpy(joint_atom37_mask).type(torch.bool).unsqueeze(0)
        )

        # If using reflection flip target
        if use_reflection:
            target_atom_tensor = -target_atom_tensor

        # Compute alignment and rmsd
        affine3D, rmsd = compute_affine_and_rmsd(
            mobile_atom_tensor, target_atom_tensor, atom_exists_mask=joint_atom37_mask
        )
        self._affine3D = affine3D
        self._rmsd = rmsd.item()

    @property
    def rmsd(self):
        return self._rmsd

    def apply(self, mobile: T) -> T:
        """Apply alignment to a protein chain"""
        # Extract atom positions and convert to batched tensors
        mobile_atom_tensor = (
            torch.from_numpy(mobile.atom37_positions[mobile.atom37_mask])
            .type(torch.float32)
            .unsqueeze(0)
        )

        # Transform atom arrays
        aligned_atom_tensor = self._affine3D.apply(mobile_atom_tensor).squeeze(0)

        # Rebuild atom37 positions
        aligned_atom37_positions = np.full_like(mobile.atom37_positions, np.nan)
        aligned_atom37_positions[mobile.atom37_mask] = aligned_atom_tensor

        return replace(mobile, atom37_positions=aligned_atom37_positions)
