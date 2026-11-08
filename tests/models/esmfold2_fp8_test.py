"""fp8 ESMC backbone for ESMFold2, on native ``torch._scaled_mm``."""

import pytest
import torch

from esm.models.esmfold2.fp8 import (
    Fp8LayerNormLinear,
    Fp8LayerNormMLP,
    Fp8Linear,
    _has_fp8_gemm,
    _quantize_rows_torch,
    convert_esmc_to_fp8_,
)


def _dequantized(module: Fp8Linear) -> torch.Tensor:
    weight = module.weight_fp8.view(torch.float8_e4m3fn).float()
    return weight * module.weight_scale.view(torch.float32)


def test_fp8_weight_survives_dtype_casts():
    """``model.to(bf16)`` runs after conversion in several load paths; it must not
    upcast the fp8 weight or round its fp32 scale."""
    torch.manual_seed(0)
    weight = torch.randn(64, 32) * 0.02
    module = Fp8Linear.from_weight(weight).to(torch.bfloat16)

    assert module.weight_fp8.dtype == torch.uint8
    assert module.weight_scale.dtype == torch.int32
    # e4m3 keeps 3 mantissa bits, so one rounding step is at most 2**-4 relative.
    torch.testing.assert_close(_dequantized(module), weight, atol=0, rtol=2**-4)


def test_the_upcast_fallback_warns_once_per_device(caplog):
    _has_fp8_gemm.cache_clear()
    module = Fp8Linear.from_weight(torch.randn(16, 32))
    with caplog.at_level("WARNING"), torch.no_grad():
        module(torch.randn(3, 32))
        module(torch.randn(3, 32))
    warnings = [r for r in caplog.records if "no fp8 GEMM" in r.getMessage()]
    assert len(warnings) == 1


def test_cpu_conversion_runs_the_upcast_path(tiny_esmc):
    """Conversion happens on the CPU before the move to GPU, and a CPU forward
    upcasts the fp8 weight instead of failing."""
    ids = torch.randint(4, 24, (2, 13))
    with torch.no_grad():
        expected = tiny_esmc(input_ids=ids).last_hidden_state
        assert convert_esmc_to_fp8_(tiny_esmc) == 3 * len(tiny_esmc.transformer.blocks)
        actual = tiny_esmc(input_ids=ids).last_hidden_state
    rel = (actual - expected).norm() / expected.norm()
    assert rel < 0.05, rel


@pytest.mark.gpu
def test_esmfold2_fp8_backbone_is_fully_fp8(tiny_esmfold2, tiny_esmc):
    """Every ESMC projection must be fp8. The old path converted only
    ``out_proj`` and left the LayerNorm-fused QKV and MLP weights in bf16."""
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("fp8 needs compute capability >= 8.9")  # ty:ignore[too-many-positional-arguments]
    tiny_esmfold2.esmc = tiny_esmc
    model = tiny_esmfold2.cuda()
    model.set_esmc_precision("fp8")

    for block in model.esmc.transformer.blocks:
        assert isinstance(block.attn.layernorm_qkv, Fp8LayerNormLinear)
        assert isinstance(block.attn.out_proj, Fp8Linear)
        assert isinstance(block.ffn, Fp8LayerNormMLP)
    assert not [
        name
        for name, p in model.esmc.named_parameters()
        if p.dim() == 2 and name != "embed.weight"
    ], "a 2D weight was left in bf16"

    ids = torch.randint(4, 24, (2, 37), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.esmc(input_ids=ids).last_hidden_state
    assert out.shape == (2, 37, tiny_esmc.config.hidden_size)
    assert torch.isfinite(out.float()).all()


@pytest.mark.gpu
@pytest.mark.parametrize("fused", ["plain", "swiglu", "norm"])
def test_triton_quantization_matches_the_torch_reference(fused):
    # Triton rejects fp8e4nv below compute capability 8.9 (e.g. A10G, A100).
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("fp8 needs compute capability >= 8.9")  # ty:ignore[too-many-positional-arguments]
    from esm.models.esmfold2.kernels.fp8_quantize import quantize_rows_fp8

    torch.manual_seed(0)
    # 6912 is ESMC-6B's MLP width: not a power of two, so the tail is masked.
    x = torch.randn(37, 2 * 6912 if fused == "swiglu" else 6912, device="cuda")
    x = x.to(torch.bfloat16)
    norm = None
    if fused == "norm":
        norm = (
            torch.randn(6912, device="cuda"),
            torch.randn(6912, device="cuda"),
            1e-5,
        )
    kwargs = {"swiglu": fused == "swiglu", "norm": norm}

    q, scale = quantize_rows_fp8(x, **kwargs)
    q_ref, scale_ref = _quantize_rows_torch(x, **kwargs)
    torch.testing.assert_close(scale, scale_ref, rtol=1e-5, atol=0)
    # Different reduction and sigmoid orders may flip a rounding: allow one e4m3
    # step (2**-3 relative) on a handful of elements.
    deq, deq_ref = q.float() * scale, q_ref.float() * scale_ref
    torch.testing.assert_close(deq, deq_ref, rtol=2**-3, atol=scale_ref.max().item())
    assert (q.view(torch.uint8) != q_ref.view(torch.uint8)).float().mean() < 1e-3
