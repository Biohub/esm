import torch
import torch.nn as nn

from esm.utils.constants.physics import BB_COORDINATES
from esm.utils.structure.affine3d import (
    Affine3D,
    RotationMatrix,
)


class Dim6RotStructureHead(nn.Module):
    # Normally, AF2 uses quaternions to specify rotations. There's some evidence that
    # other representations are more well behaved - the best one according to
    # https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf
    # is using graham schmidt on 2 vectors, which is implemented here.
    def __init__(
        self,
        input_dim: int,
        trans_scale_factor: float = 10,
        norm_type: str = "layernorm",
        activation_fn: str = "esm_gelu",
        predict_torsion_angles: bool = True,
    ):
        super().__init__()
        self.ffn1 = nn.Linear(input_dim, input_dim)
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, 9 + 7 * 2)
        self.trans_scale_factor = trans_scale_factor
        self.predict_torsion_angles = predict_torsion_angles
        self.bb_local_coords = torch.tensor(BB_COORDINATES).float()

    def forward(self, x, affine, affine_mask, **kwargs):
        if affine is None:
            rigids = Affine3D.identity(
                x.shape[:-1],
                dtype=x.dtype,
                device=x.device,
                requires_grad=self.training,
                rotation_type=RotationMatrix,
            )
        else:
            rigids = affine

        # [*, N]
        x = self.ffn1(x)
        x = self.activation_fn(x)
        x = self.norm(x)
        trans, x, y, angles = self.proj(x).split([3, 3, 3, 7 * 2], dim=-1)
        trans = trans * self.trans_scale_factor
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-5)
        y = y / (y.norm(dim=-1, keepdim=True) + 1e-5)
        update = Affine3D.from_graham_schmidt(x + trans, trans, y + trans)
        rigids = rigids.compose(update.mask(affine_mask))
        affine = rigids.tensor

        # We approximate the positions of the backbone atoms in the global frame by applying the rigid
        # transformation to the mean of the backbone atoms in the local frame.
        all_bb_coords_local = (
            self.bb_local_coords[None, None, :, :]
            .expand(*x.shape[:-1], 3, 3)
            .to(x.device)
        )
        pred_xyz = rigids[..., None].apply(all_bb_coords_local)

        return affine, pred_xyz
