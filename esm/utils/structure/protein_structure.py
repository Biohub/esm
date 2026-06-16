from __future__ import annotations

from typing import Tuple, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from esm.utils import residue_constants
from esm.utils.misc import unbinpack
from esm.utils.structure.affine3d import Affine3D

ArrayOrTensor = TypeVar("ArrayOrTensor", np.ndarray, Tensor)


def index_by_atom_name(
    atom37: ArrayOrTensor, atom_names: str | list[str], dim: int = -2
) -> ArrayOrTensor:
    squeeze = False
    if isinstance(atom_names, str):
        atom_names = [atom_names]
        squeeze = True
    indices = [residue_constants.atom_order[atom_name] for atom_name in atom_names]
    dim = dim % atom37.ndim
    index = tuple(slice(None) if dim != i else indices for i in range(atom37.ndim))
    result = atom37[index]  # type: ignore
    if squeeze:
        result = result.squeeze(dim)
    return result  # ty:ignore[invalid-return-type]


def infer_cbeta_from_atom37(
    atom37: ArrayOrTensor, L: float = 1.522, A: float = 1.927, D: float = -2.143
):
    """
    Inspired by a util in trDesign:
    https://github.com/gjoni/trDesign/blob/f2d5930b472e77bfacc2f437b3966e7a708a8d37/02-GD/utils.py#L92

    input:  atom37, (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """
    N = index_by_atom_name(atom37, "N", dim=-2)
    CA = index_by_atom_name(atom37, "CA", dim=-2)
    C = index_by_atom_name(atom37, "C", dim=-2)

    if isinstance(atom37, np.ndarray):

        def normalize(x: ArrayOrTensor):
            return x / np.linalg.norm(x, axis=-1, keepdims=True)

        cross = np.cross
    else:
        normalize = F.normalize
        cross = torch.cross

    with np.errstate(invalid="ignore"):  # inf - inf = nan is ok here
        vec_nca = N - CA
        vec_nc = N - C
    nca = normalize(vec_nca)
    n = normalize(cross(vec_nc, nca))  # type: ignore
    m = [nca, cross(n, nca), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return CA + sum([m * d for m, d in zip(m, d)])


@torch.no_grad()
@autocast("cuda", enabled=False)
def compute_alignment_tensors(
    mobile: torch.Tensor,
    target: torch.Tensor,
    atom_exists_mask: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
):
    """
    Align two batches of structures with support for masking invalid atoms using PyTorch.

    Args:
    - mobile (torch.Tensor): Batch of coordinates of structure to be superimposed in shape (B, N, 3)
    - target (torch.Tensor): Batch of coordinates of structure that is fixed in shape (B, N, 3)
    - atom_exists_mask (torch.Tensor, optional): Mask for Whether an atom exists of shape (B, N)
    - sequence_id (torch.Tensor, optional): Sequence id tensor for binpacking.

    Returns:
    - centered_mobile (torch.Tensor): Batch of coordinates of structure centered mobile (B, N, 3)
    - centroid_mobile (torch.Tensor): Batch of coordinates of mobile centeroid (B, 3)
    - centered_target (torch.Tensor): Batch of coordinates of structure centered target (B, N, 3)
    - centroid_target (torch.Tensor): Batch of coordinates of target centeroid (B, 3)
    - rotation_matrix (torch.Tensor): Batch of coordinates of rotation matrix (B, 3, 3)
    - num_valid_atoms (torch.Tensor): Batch of number of valid atoms for alignment (B,)
    """

    # Ensure both batches have the same number of structures, atoms, and dimensions
    if sequence_id is not None:
        mobile = unbinpack(mobile, sequence_id, pad_value=torch.nan)
        target = unbinpack(target, sequence_id, pad_value=torch.nan)
        if atom_exists_mask is not None:
            atom_exists_mask = unbinpack(atom_exists_mask, sequence_id, pad_value=0)
        else:
            atom_exists_mask = torch.isfinite(target).all(-1)

    assert mobile.shape == target.shape, "Batch structure shapes do not match!"

    # Number of structures in the batch
    batch_size = mobile.shape[0]

    # if [B, Nres, Natom, 3], resize
    if mobile.dim() == 4:
        mobile = mobile.view(batch_size, -1, 3)
    if target.dim() == 4:
        target = target.view(batch_size, -1, 3)
    if atom_exists_mask is not None and atom_exists_mask.dim() == 3:
        atom_exists_mask = atom_exists_mask.view(batch_size, -1)

    # Number of atoms
    num_atoms = mobile.shape[1]

    # Apply masks if provided
    if atom_exists_mask is not None:
        mobile = mobile.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)
        target = target.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)
    else:
        atom_exists_mask = torch.ones(
            batch_size, num_atoms, dtype=torch.bool, device=mobile.device
        )

    num_valid_atoms = atom_exists_mask.sum(dim=-1, keepdim=True)
    # Compute centroids for each batch
    centroid_mobile = mobile.sum(dim=-2, keepdim=True) / num_valid_atoms.unsqueeze(-1)
    centroid_target = target.sum(dim=-2, keepdim=True) / num_valid_atoms.unsqueeze(-1)

    # Handle potential division by zero if all atoms are invalid in a structure
    centroid_mobile[num_valid_atoms == 0] = 0
    centroid_target[num_valid_atoms == 0] = 0

    # Center structures by subtracting centroids
    centered_mobile = mobile - centroid_mobile
    centered_target = target - centroid_target

    centered_mobile = centered_mobile.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)
    centered_target = centered_target.masked_fill(~atom_exists_mask.unsqueeze(-1), 0)

    # Compute covariance matrix for each batch
    covariance_matrix = torch.matmul(centered_mobile.transpose(1, 2), centered_target)

    # Singular Value Decomposition for each batch
    u, _, v = torch.svd(covariance_matrix)

    # Calculate rotation matrices for each batch
    rotation_matrix = torch.matmul(u, v.transpose(1, 2))

    return (
        centered_mobile,
        centroid_mobile,
        centered_target,
        centroid_target,
        rotation_matrix,
        num_valid_atoms,
    )


@torch.no_grad()
@autocast("cuda", enabled=False)
def compute_rmsd_no_alignment(
    aligned: torch.Tensor,
    target: torch.Tensor,
    num_valid_atoms: torch.Tensor,
    reduction: str = "batch",
) -> torch.Tensor:
    """
    Compute RMSD between two batches of structures without alignment.

    Args:
    - mobile (torch.Tensor): Batch of coordinates of structure to be superimposed in shape (B, N, 3)
    - target (torch.Tensor): Batch of coordinates of structure that is fixed in shape (B, N, 3)
    - num_valid_atoms (torch.Tensor): Batch of number of valid atoms for alignment (B,)
    - reduction (str): One of "batch", "per_sample", "per_residue".

    Returns:

    If reduction == "batch":
        (torch.Tensor): 0-dim, Average Root Mean Square Deviation between the structures for each batch
    If reduction == "per_sample":
        (torch.Tensor): (B,)-dim, Root Mean Square Deviation between the structures for each batch
    If reduction == "per_residue":
        (torch.Tensor): (B, N)-dim, Root Mean Square Deviation between the structures for residue in the batch
    """
    if reduction not in ("per_residue", "per_sample", "batch"):
        raise ValueError("Unrecognized reduction: '{reduction}'")
    # Compute RMSD for each batch
    diff = aligned - target
    if reduction == "per_residue":
        mean_squared_error = diff.square().view(diff.size(0), -1, 9).mean(dim=-1)
    else:
        mean_squared_error = diff.square().sum(dim=(1, 2)) / (
            num_valid_atoms.squeeze(-1)
        )

    rmsd = torch.sqrt(mean_squared_error)
    if reduction in ("per_sample", "per_residue"):
        return rmsd
    elif reduction == "batch":
        avg_rmsd = rmsd.masked_fill(num_valid_atoms.squeeze(-1) == 0, 0).sum() / (
            (num_valid_atoms > 0).sum() + 1e-8
        )
        return avg_rmsd
    else:
        raise ValueError(reduction)


@torch.no_grad()
@autocast("cuda", enabled=False)
def compute_affine_and_rmsd(
    mobile: torch.Tensor,
    target: torch.Tensor,
    atom_exists_mask: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
) -> Tuple[Affine3D, torch.Tensor]:
    """
    Compute RMSD between two batches of structures with support for masking invalid atoms using PyTorch.

    Args:
    - mobile (torch.Tensor): Batch of coordinates of structure to be superimposed in shape (B, N, 3)
    - target (torch.Tensor): Batch of coordinates of structure that is fixed in shape (B, N, 3)
    - atom_exists_mask (torch.Tensor, optional): Mask for Whether an atom exists of shape (B, N)
    - sequence_id (torch.Tensor, optional): Sequence id tensor for binpacking.

    Returns:
    - affine (Affine3D): Transformation between mobile and target structure
    - avg_rmsd (torch.Tensor): Average Root Mean Square Deviation between the structures for each batch
    """

    (
        centered_mobile,
        centroid_mobile,
        centered_target,
        centroid_target,
        rotation_matrix,
        num_valid_atoms,
    ) = compute_alignment_tensors(
        mobile=mobile,
        target=target,
        atom_exists_mask=atom_exists_mask,
        sequence_id=sequence_id,
    )

    # Apply rotation to mobile centroid
    translation = torch.matmul(-centroid_mobile, rotation_matrix) + centroid_target
    affine = Affine3D.from_tensor_pair(
        translation, rotation_matrix.unsqueeze(dim=-3).transpose(-2, -1)
    )

    # Apply transformation to centered structure to compute rmsd
    rotated_mobile = torch.matmul(centered_mobile, rotation_matrix)
    avg_rmsd = compute_rmsd_no_alignment(
        rotated_mobile, centered_target, num_valid_atoms, reduction="batch"
    )

    return affine, avg_rmsd


def compute_gdt_ts_no_alignment(
    aligned: torch.Tensor,
    target: torch.Tensor,
    atom_exists_mask: torch.Tensor,
    reduction: str = "batch",
) -> torch.Tensor:
    """
    Compute GDT_TS between two batches of structures without alignment.

    Args:
    - mobile (torch.Tensor): Batch of coordinates of structure to be superimposed in shape (B, N, 3)
    - target (torch.Tensor): Batch of coordinates of structure that is fixed in shape (B, N, 3)
    - atom_exists_mask (torch.Tensor): Mask for Whether an atom exists of shape (B, N). noo
    - reduction (str): One of "batch", "per_sample".

    Returns:
    If reduction == "batch":
        (torch.Tensor): 0-dim, GDT_TS between the structures for each batch
    If reduction == "per_sample":
        (torch.Tensor): (B,)-dim, GDT_TS between the structures for each sample in the batch
    """
    if reduction not in ("per_sample", "batch"):
        raise ValueError("Unrecognized reduction: '{reduction}'")

    if atom_exists_mask is None:
        atom_exists_mask = torch.isfinite(target).all(dim=-1)

    deviation = torch.linalg.vector_norm(aligned - target, dim=-1)
    num_valid_atoms = atom_exists_mask.sum(dim=-1)

    # Compute GDT_TS
    score = (
        ((deviation < 1) * atom_exists_mask).sum(dim=-1) / num_valid_atoms
        + ((deviation < 2) * atom_exists_mask).sum(dim=-1) / num_valid_atoms
        + ((deviation < 4) * atom_exists_mask).sum(dim=-1) / num_valid_atoms
        + ((deviation < 8) * atom_exists_mask).sum(dim=-1) / num_valid_atoms
    ) * 0.25

    if reduction == "batch":
        return score.mean()
    elif reduction == "per_sample":
        return score
    else:
        raise ValueError("Unrecognized reduction: '{reduction}'")
