"""The forward-only pair ``Transition`` must be bitwise and must skip the backward buffer.

``FusedLNLinearSwiGLU`` writes an ``[M, 2*d_inner]`` linear-output buffer that
only backward reads. A forward-only call now skips that store. These tests pin
the two things that could go wrong: the forward result changing, and the store
being skipped on a call that does have a backward.
"""

from __future__ import annotations

import pytest
import torch

from esm.models.esmfold2.layers import Transition

D_MODEL = 256
D_INNER = Transition(D_MODEL).ffn.hidden_features


def _transition(device: str = "cuda") -> Transition:
    torch.manual_seed(0)
    return Transition(D_MODEL).to(device=device, dtype=torch.bfloat16).eval()


def _with_buffer(t: Transition, x: torch.Tensor) -> torch.Tensor:
    """The pre-change path: force the [M, 2N] store, then the same addmm residual."""
    # Local: the kernels package needs CUDA at import, and collection has to
    # work on the CPU boxes (see esmfold2_cpu_only_test).
    from esm.models.esmfold2.kernels.fused_lnlin_swiglu import (
        FusedLNLinearSwiGLUFunction,
    )

    fused = t._fused_swiglu
    assert fused is not None
    hidden = FusedLNLinearSwiGLUFunction.apply(
        x, fused.W12, fused.LN_W, fused.LN_B, True
    )
    return t._addmm_residual(x, hidden)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(1, 64, 64), (1, 137, 137), (1, 200, 200)])
def test_forward_only_transition_is_bitwise(shape):
    """Skipping the store must not move a single bit of the output."""
    t = _transition()
    t.set_chunk_size(None)
    t.set_kernel_backend("fused")
    x = torch.randn(*shape, D_MODEL, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        ref = _with_buffer(t, x)
        got = t(x)

    assert torch.equal(got, ref), (
        f"max delta {(got.float() - ref.float()).abs().max().item():.3e}"
    )


@pytest.mark.gpu
def test_frozen_weights_with_input_grad_keeps_the_buffer():
    """The binder-design shape: params frozen, input requires grad.

    ``_fused_pair_stack_active`` deliberately permits grad for this path, so the
    no-backward detection must not key off the weights. Getting it wrong here
    silently breaks the design gradient.
    """
    t = _transition()
    t.set_chunk_size(None)
    t.set_kernel_backend("fused")
    t.requires_grad_(False)  # frozen weights, as binder design runs it
    rows = 128 * 128
    buffer_bytes = rows * 2 * D_INNER * 2
    x = torch.randn(1, 128, 128, D_MODEL, device="cuda", dtype=torch.bfloat16)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    xr = x.clone().requires_grad_(True)
    out = t(xr)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base

    assert out.requires_grad, "frozen weights must still yield a differentiable output"
    assert peak >= buffer_bytes, f"peak {peak} < buffer {buffer_bytes}"

    out.float().square().sum().backward()
    assert xr.grad is not None

    # And the gradient matches the path that always stored the buffer.
    xr2 = x.clone().requires_grad_(True)
    _with_buffer(t, xr2).float().square().sum().backward()
    assert xr2.grad is not None
    assert torch.equal(xr.grad, xr2.grad)


@pytest.mark.gpu
def test_no_grad_with_input_grad_still_skips_the_buffer():
    """Grad globally off outranks a grad-requiring input: no backward can run."""
    t = _transition()
    t.set_chunk_size(None)
    t.set_kernel_backend("fused")
    rows = 128 * 128
    buffer_bytes = rows * 2 * D_INNER * 2
    x = torch.randn(1, 128, 128, D_MODEL, device="cuda", dtype=torch.bfloat16)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        out = t(x.clone().requires_grad_(True))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out

    assert peak < buffer_bytes, f"peak {peak} still fits the {buffer_bytes}-byte buffer"


@pytest.mark.gpu
def test_chunked_forward_is_bitwise_too():
    """Chunking changes the row count per call; every chunk must still match."""
    t = _transition()
    t.set_kernel_backend("fused")
    t.set_chunk_size(32)
    x = torch.randn(1, 70, 70, D_MODEL, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        got = t(x)
        ref = torch.cat(
            [_with_buffer(t, x[:, s : s + 32]) for s in range(0, 70, 32)], dim=1
        )
    assert torch.equal(got, ref)
