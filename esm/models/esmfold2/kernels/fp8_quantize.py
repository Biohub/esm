"""Row-wise fp8 quantization, optionally fused with the op that feeds the GEMM.

One program per row: an optional LayerNorm or SwiGLU, then the row amax, the
scale and the fp8 cast. Replaces five to seven separate PyTorch kernels per GEMM.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


@triton.jit
def _quantize_rows_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    q_ptr,
    s_ptr,
    K,
    stride_x,
    eps,
    fp8_max,
    NORM: tl.constexpr,
    SWIGLU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    base = x_ptr + row * stride_x
    x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    if SWIGLU:
        up = tl.load(base + K + offs, mask=mask, other=0.0).to(tl.float32)
        x = x * tl.sigmoid(x) * up
    if NORM:
        mean = tl.sum(x, axis=0) / K
        centred = tl.where(mask, x - mean, 0.0)
        rstd = tl.rsqrt(tl.sum(centred * centred, axis=0) / K + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = centred * rstd * w + b
    # IEEE division, as torch does: Triton's default `/` is approximate and flips fp8 roundings.
    scale = tl.math.div_rn(tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12), fp8_max)
    q = tl.math.div_rn(x, scale)
    # Snap to the e4m3 grid here: sm89 casts via fp16, and that double rounding
    # sends ~0.3% of values one fp8 step the wrong way. The cast below is then exact.
    exponent = tl.maximum(((q.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127, -6)
    step = ((exponent - 3 + 127) << 23).to(tl.float32, bitcast=True)
    q = libdevice.rint(q / step) * step
    tl.store(q_ptr + row * K + offs, q.to(tl.float8e4nv), mask=mask)
    tl.store(s_ptr + row, scale)


def quantize_rows_fp8(
    x: Tensor, *, swiglu: bool = False, norm: tuple[Tensor, Tensor, float] | None = None
) -> tuple[Tensor, Tensor]:
    """Quantize a 2D CUDA tensor to fp8 e4m3 with one fp32 scale per row.

    ``swiglu``: ``x`` is ``[gate, up]`` along the last dim; quantize
    ``silu(gate) * up``. ``norm``: ``(weight, bias, eps)`` of a LayerNorm applied
    first.
    """
    assert x.dim() == 2 and x.stride(-1) == 1
    rows, width = x.shape
    K = width // 2 if swiglu else width
    q = torch.empty((rows, K), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    w, b, eps = norm if norm is not None else (x, x, 0.0)
    block = triton.next_power_of_2(K)
    _quantize_rows_kernel[(rows,)](
        x,
        w,
        b,
        q,
        scale,
        K,
        x.stride(0),
        eps,
        _FP8_MAX,
        NORM=norm is not None,  # ty:ignore[invalid-argument-type]
        SWIGLU=swiglu,  # ty:ignore[invalid-argument-type]
        BLOCK=block,
        num_warps=8 if block >= 4096 else 4,  # ty:ignore[unknown-argument]
    )
    return q, scale
