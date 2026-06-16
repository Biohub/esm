from typing import TypeVar

import numpy as np
import torch
from torch import Tensor

from esm.utils import residue_constants as RC
from esm.utils.structure.affine3d import Affine3D

ArrayOrTensor = TypeVar("ArrayOrTensor", np.ndarray, Tensor)


def atom3_to_backbone_frames(bb_positions: torch.Tensor) -> Affine3D:
    N, CA, C = bb_positions.unbind(dim=-2)
    return Affine3D.from_graham_schmidt(C, CA, N)


def index_by_atom_name(
    atom37: ArrayOrTensor, atom_names: str | list[str], dim: int = -2
) -> ArrayOrTensor:
    squeeze = False
    if isinstance(atom_names, str):
        atom_names = [atom_names]
        squeeze = True
    indices = [RC.atom_order[atom_name] for atom_name in atom_names]
    dim = dim % atom37.ndim
    index = tuple(slice(None) if dim != i else indices for i in range(atom37.ndim))
    result = atom37[index]  # type: ignore
    if squeeze:
        result = result.squeeze(dim)
    return result  # ty:ignore[invalid-return-type]


def get_protein_normalization_frame(coords: Tensor) -> Affine3D:
    """Given a set of coordinates for a protein, compute a single frame that can be used to normalize the coordinates.
    Specifically, we compute the average position of the N, CA, and C atoms use those 3 points to construct a frame
    using the Gram-Schmidt algorithm. The average CA position is used as the origin of the frame.

    Args:
        coords (torch.FloatTensor): [L, 37, 3] tensor of coordinates

    Returns:
        Affine3D: tensor of Affine3D frame
    """
    bb_coords = index_by_atom_name(coords, ["N", "CA", "C"], dim=-2)
    coord_mask = torch.all(torch.all(torch.isfinite(bb_coords), dim=-1), dim=-1)

    average_position_per_n_ca_c = bb_coords.masked_fill(
        ~coord_mask[..., None, None], 0
    ).sum(-3) / (coord_mask.sum(-1)[..., None, None] + 1e-8)
    frame = atom3_to_backbone_frames(average_position_per_n_ca_c.float())

    return frame


def apply_frame_to_coords(coords: Tensor, frame: Affine3D) -> Tensor:
    """Given a set of coordinates and a single frame, apply the frame to the coordinates.

    Args:
        coords (torch.FloatTensor): [L, 37, 3] tensor of coordinates
        frame (Affine3D): Affine3D frame

    Returns:
        torch.FloatTensor: [L, 37, 3] tensor of transformed coordinates
    """
    coords_trans_rot = frame[..., None, None].invert().apply(coords)

    # only transform coordinates with frame that have a valid rotation
    valid_frame = frame.trans.norm(dim=-1) > 0

    is_inf = torch.isinf(coords)
    coords = coords_trans_rot.where(valid_frame[..., None, None, None], coords)
    coords.masked_fill_(is_inf, torch.inf)

    return coords


def normalize_coordinates(coords: Tensor) -> Tensor:
    return apply_frame_to_coords(coords, get_protein_normalization_frame(coords))
