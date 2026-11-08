"""``set_kernel_backend("fast")`` reaches the MSA encoder's triangle updates.

The MSA encoder ran the eager fp32 contraction while the trunk ran the bf16
Triton kernel on the identical operation. The kernel is NOT bitwise against the
eager engine, so it sits behind "fast", never behind "fused".

These tests pin what can go wrong: the kernel not firing under "fast", the
kernel firing under "fused" (which must stay bit-exact), the trunk's own gate
moving, and the CPU fallback losing its output.
"""

import contextlib

import pytest
import torch

from esm.models.esmfold2.layers import PairUpdateBlock, TriangleMultiplicativeUpdate
from esm.models.esmfold2.model import MSAEncoderBlock

D_PAIR = 256


def _msa_block(device: str) -> MSAEncoderBlock:
    torch.manual_seed(0)
    return (
        MSAEncoderBlock(
            d_msa=64,
            d_pair=D_PAIR,
            d_hidden=32,
            n_heads_msa=8,
            msa_head_width=16,
            is_final_block=True,
        )
        .to(device)
        .eval()
    )


def _pair_and_mask(L: int, device: str, dtype: torch.dtype):
    torch.manual_seed(1)
    pair = torch.randn(1, L, L, D_PAIR, device=device, dtype=dtype)
    mask = torch.ones(1, L, L, device=device, dtype=torch.bool)
    mask[:, :, L - 5 :] = False  # padded tail, so the mask is load-bearing
    mask[:, L - 5 :, :] = False
    return pair, mask


def _amp(device: str):
    """The recycle loop runs the MSA encoder under bf16 autocast with fp32 weights.

    Without it the eager engine's LayerNorm dtype-errors on a bf16 pair.
    """
    if device != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)


def _eager_updates(blk: MSAEncoderBlock, pair, mask):
    """The pre-change arithmetic: two eager updates, each with a plain residual."""
    out = pair + blk.tri_mul_out(pair, mask=mask)
    return out + blk.tri_mul_in(out, mask=mask)


def test_invalid_backend_raises():
    tri = TriangleMultiplicativeUpdate(dim=32)
    with pytest.raises(ValueError, match="backend must be one of"):
        tri.set_kernel_backend("bogus")


def test_set_kernel_backend_is_recorded_on_the_wrapper():
    """It used to be a no-op, so the MSA encoder could never see the request."""
    tri = TriangleMultiplicativeUpdate(dim=32)
    assert not tri.fast_backend_requested()
    tri.set_kernel_backend("fast")
    assert tri.fast_backend_requested()
    tri.set_kernel_backend(None)
    assert not tri.fast_backend_requested()


def test_fused_is_not_fast_on_the_wrapper():
    """ "fused" must not arm the approximate MSA kernel -- that is the whole split."""
    tri = TriangleMultiplicativeUpdate(dim=32)
    tri.set_kernel_backend("fused")
    assert not tri.fast_backend_requested()


def test_cpu_msa_block_warns_and_keeps_the_eager_result():
    """No CUDA means no kernel: the fallback must warn and stay bit-exact."""
    import esm.models.esmfold2.layers as layers

    blk = _msa_block("cpu")
    blk.set_chunk_size(None)
    pair, mask = _pair_and_mask(24, "cpu", torch.float32)

    with torch.inference_mode():
        blk.set_kernel_backend(None)
        ref = blk._triangle_updates(pair, mask)
        blk.set_kernel_backend("fast")
        layers._fast_trimul_refusal_warned = False
        with pytest.warns(UserWarning, match="eager triangle multiplication"):
            got = blk._triangle_updates(pair, mask)

    assert torch.equal(got, ref)


def test_cpu_fused_backend_does_not_warn():
    """ "fused" never asks for this kernel, so a CPU fold must stay silent."""
    import warnings as _warnings

    import esm.models.esmfold2.layers as layers

    blk = _msa_block("cpu")
    blk.set_chunk_size(None)
    pair, mask = _pair_and_mask(24, "cpu", torch.float32)

    layers._fast_trimul_refusal_warned = False
    blk.set_kernel_backend("fused")
    with torch.inference_mode(), _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any warning here is a regression
        blk._triangle_updates(pair, mask)


@pytest.mark.gpu
def test_fused_msa_updates_fire_and_track_the_eager_path():
    """The kernel must run twice per block and land within bf16 tolerance."""
    from esm.models.esmfold2 import layers

    blk = _msa_block("cuda")
    blk.set_chunk_size(None)
    blk.set_kernel_backend("fast")
    pair, mask = _pair_and_mask(128, "cuda", torch.bfloat16)

    calls = {"n": 0}
    orig = layers._fused_trimul_with_residual

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    layers._fused_trimul_with_residual = counting
    try:
        with torch.inference_mode(), _amp("cuda"):
            eager = _eager_updates(blk, pair, mask)
            fused = blk._triangle_updates(pair, mask)
    finally:
        layers._fused_trimul_with_residual = orig

    assert calls["n"] == 2, "one call per triangle update, or the gate refused"
    assert fused.dtype == pair.dtype
    assert fused.shape == eager.shape

    # Not bitwise, on purpose: fp32 eager contraction vs bf16 kernel.
    assert not torch.equal(fused, eager)
    scale = eager.double().abs().max().clamp_min(1e-6)
    rel = (fused.double() - eager.double()).abs().max() / scale
    assert rel < 0.05, f"relative max delta {rel.item():.3e}"


@pytest.mark.gpu
def test_fused_msa_updates_ignore_chunk_size():
    """Chunking is an eager-only knob; the fused path must not change with it."""
    blk = _msa_block("cuda")
    blk.set_kernel_backend("fast")
    pair, mask = _pair_and_mask(128, "cuda", torch.bfloat16)

    with torch.inference_mode(), _amp("cuda"):
        blk.set_chunk_size(None)
        unchunked = blk._triangle_updates(pair, mask)
        blk.set_chunk_size(64)
        chunked = blk._triangle_updates(pair, mask)

    assert torch.equal(chunked, unchunked)


@pytest.mark.gpu
def test_fp32_pair_is_cast_rather_than_refused():
    """The trunk casts at the stack level; the MSA block casts around the pair."""
    blk = _msa_block("cuda")
    blk.set_chunk_size(None)
    blk.set_kernel_backend("fast")
    pair, mask = _pair_and_mask(64, "cuda", torch.float32)

    with torch.inference_mode(), _amp("cuda"):
        out = blk._triangle_updates(pair, mask)

    assert out.dtype == torch.float32


@pytest.mark.gpu
def test_trunk_gate_is_unchanged_by_the_delegation():
    """PairUpdateBlock still demands bf16 on CUDA and nothing else."""
    blk = PairUpdateBlock(d_pair=D_PAIR).cuda().eval()
    blk.set_kernel_backend("fast")
    bf16 = torch.zeros(1, 8, 8, D_PAIR, device="cuda", dtype=torch.bfloat16)

    assert blk._can_use_fused_trimul_with_residual(bf16)
    assert not blk._can_use_fused_trimul_with_residual(bf16.float())
    assert not blk._can_use_fused_trimul_with_residual(bf16.cpu())

    blk.set_kernel_backend(None)
    assert not blk._can_use_fused_trimul_with_residual(bf16)


@pytest.mark.gpu
def test_fast_does_not_touch_gradients():
    """ "fast" is an inference accelerator: it must not move training gradients.

    The kernel is approximate, so arming it under autograd would silently change
    what a training run optimises. The eager engine must serve instead, and the
    backward must still work.
    """
    from esm.models.esmfold2 import layers

    blk = _msa_block("cuda")
    blk.set_chunk_size(None)
    blk.set_kernel_backend("fast")
    pair, mask = _pair_and_mask(64, "cuda", torch.bfloat16)
    pair = pair.clone().requires_grad_(True)

    assert not blk.tri_mul_out.fast_residual_available(pair), (
        "the approximate kernel armed itself with gradients enabled"
    )

    calls = {"n": 0}
    orig = layers._fused_trimul_with_residual

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    layers._fused_trimul_with_residual = counting
    try:
        with _amp("cuda"):
            out = blk._triangle_updates(pair, mask)
        out.float().pow(2).mean().backward()
    finally:
        layers._fused_trimul_with_residual = orig

    assert calls["n"] == 0
    assert pair.grad is not None
    assert torch.isfinite(pair.grad).all()
    assert pair.grad.abs().sum() > 0


@pytest.mark.gpu
def test_fast_arms_itself_under_inference_mode():
    """The mirror of the gradient test: no grad means the kernel does serve."""
    blk = _msa_block("cuda")
    blk.set_kernel_backend("fast")
    pair, _ = _pair_and_mask(64, "cuda", torch.bfloat16)

    with torch.inference_mode():
        assert blk.tri_mul_out.fast_residual_available(pair)
    with torch.no_grad():
        assert blk.tri_mul_out.fast_residual_available(pair)


@pytest.mark.gpu
def test_fused_backend_leaves_the_msa_encoder_bit_exact():
    """The contract: "fused" is bit-exact, so it must not touch this path.

    If "fused" ever arms the MSA kernel, the bit-exactness guarantee of the
    default accelerated backend silently breaks for every MSA fold.
    """
    from esm.models.esmfold2 import layers

    blk = _msa_block("cuda")
    blk.set_chunk_size(None)
    pair, mask = _pair_and_mask(128, "cuda", torch.bfloat16)

    calls = {"n": 0}
    orig = layers._fused_trimul_with_residual

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    layers._fused_trimul_with_residual = counting
    try:
        with torch.inference_mode(), _amp("cuda"):
            blk.set_kernel_backend(None)
            ref = blk._triangle_updates(pair, mask)
            blk.set_kernel_backend("fused")
            got = blk._triangle_updates(pair, mask)
    finally:
        layers._fused_trimul_with_residual = orig

    assert calls["n"] == 0, "the approximate MSA kernel fired under 'fused'"
    assert torch.equal(got, ref)


@pytest.mark.gpu
def test_fast_keeps_the_trunk_identical_to_fused():
    """ "fast" is a superset: the trunk's bit-exact kernel must still run, unchanged."""
    blk = PairUpdateBlock(d_pair=D_PAIR).cuda().eval()
    pair, mask = _pair_and_mask(64, "cuda", torch.bfloat16)
    mask_f32 = mask.to(torch.float32)

    with torch.inference_mode():
        blk.set_kernel_backend("fused")
        assert blk._can_use_fused_trimul_with_residual(pair)
        fused = blk(pair, pair_attention_mask=mask_f32)
        blk.set_kernel_backend("fast")
        assert blk._can_use_fused_trimul_with_residual(pair)
        fast = blk(pair, pair_attention_mask=mask_f32)

    assert torch.equal(fast, fused)
