"""Training through ``set_kernel_backend("fused")`` must reach every trunk weight.

The fused kernels are bit-exact in the forward, so the backend lets autograd
through. These tests pin that every parameter the reference path trains also
trains under "fused", with gradients that match the reference.
"""

import pytest
import torch

from esm.models.esmfold2 import layers as L
from esm.models.esmfold2.layers import FoldingTrunk

D_PAIR = 256


def _trunk() -> FoldingTrunk:
    torch.manual_seed(0)
    trunk = FoldingTrunk(n_layers=2, d_pair=D_PAIR).cuda().train()
    with torch.no_grad():
        for p in trunk.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    trunk.set_chunk_size(None)
    return trunk


def _grads(trunk: FoldingTrunk, pair: torch.Tensor, mask: torch.Tensor) -> dict:
    x = pair.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = trunk(x, pair_attention_mask=mask)
    out.float().pow(2).mean().backward()
    grads = {"input": x.grad}
    grads.update({n: p.grad for n, p in trunk.named_parameters()})
    return grads


@pytest.mark.gpu
@pytest.mark.parametrize("n_tokens", [96, 128])
def test_fused_trunk_trains_every_weight_like_the_reference(monkeypatch, n_tokens):
    """128 takes the Triton einsum backward; 96 takes torch.einsum."""
    served = {"trimul": 0, "transition": 0}
    trimul, transition = L._fused_trimul_with_residual, L._FusedLNLinearSwiGLUFunction

    def counted_trimul(*args, **kwargs):
        out = trimul(*args, **kwargs)
        served["trimul"] += 1
        return out

    class CountedTransition:
        @staticmethod
        def apply(*args):
            out = transition.apply(*args)
            served["transition"] += 1
            return out

    monkeypatch.setattr(L, "_fused_trimul_with_residual", counted_trimul)
    monkeypatch.setattr(L, "_FusedLNLinearSwiGLUFunction", CountedTransition)

    torch.manual_seed(1)
    pair = torch.randn(1, n_tokens, n_tokens, D_PAIR, device="cuda")
    mask = torch.ones(1, n_tokens, n_tokens, device="cuda")
    mask[:, -8:, :] = 0
    mask[:, :, -8:] = 0

    ref = _grads(_trunk(), pair, mask)
    assert served == {"trimul": 0, "transition": 0}
    trunk = _trunk()
    trunk.set_kernel_backend("fused")
    got = _grads(trunk, pair, mask)
    # >=: activation checkpointing replays each block's forward in backward.
    assert served["trimul"] >= 4 and served["transition"] >= 2, served

    for name, g_ref in ref.items():
        g_got = got[name]
        assert g_ref is not None and g_ref.abs().max() > 0, f"{name}: reference is zero"
        assert g_got is not None, f"{name} got no gradient under 'fused'"
        rel = (g_got.float() - g_ref.float()).norm() / g_ref.float().norm()
        assert rel < 5e-2, f"{name}: relative gradient error {rel.item():.3e}"
