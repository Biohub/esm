"""Geometric attention in reduced precision."""

import torch

from esm.layers.geom_attention import GeometricReasoningOriginalImpl
from esm.utils.structure.affine3d import build_affine3d_from_coordinates

C_S = 64
V_HEADS = 4
B, L = 2, 10


def test_geometric_attention_runs_in_bf16_with_fp32_frames():
    """Issue #178: a bf16 ESM3 still builds its frames from fp32 coordinates.

    ``ESM3.forward`` passes ``structure_coords`` straight to
    ``build_affine3d_from_coordinates``, so geometric attention has to mix fp32
    rotations with its own bf16 projections and hand ``out_proj`` bf16 input.
    """
    torch.manual_seed(0)
    attention = GeometricReasoningOriginalImpl(c_s=C_S, v_heads=V_HEADS)
    attention = attention.to(torch.bfloat16).eval()

    affine, affine_mask = build_affine3d_from_coordinates(torch.randn(B, L, 3, 3))
    s = torch.randn(B, L, C_S, dtype=torch.bfloat16)
    sequence_id = torch.zeros(B, L, dtype=torch.long)
    chain_id = torch.ones(B, L, dtype=torch.long)

    with torch.no_grad():
        out = attention(s, affine, affine_mask, sequence_id, chain_id)

    assert out.dtype == torch.bfloat16
    assert out.shape == (B, L, C_S)
    assert torch.isfinite(out.float()).all()
