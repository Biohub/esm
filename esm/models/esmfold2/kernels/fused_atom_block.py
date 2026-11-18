"""Fused fp32 Triton kernels for the ESMFold2 SWA atom-transformer block.

The atom transformer runs 3 blocks per call and 2 calls per diffusion step
(encoder + decoder), i.e. 6 atom blocks on a [B, A, 128] tensor with
A up to ~7.3k. Every diffusion step pays for that. Outside the attention
itself the block is pure bandwidth: four elementwise passes
(``rms_adaln`` x2, ``gated_residual`` x2) plus a tiny two-GEMM FFN at
d_model=128 / hidden=256, where each eager op is its own kernel launch and
its own HBM round-trip of the [B, A, 128] activation.

What is fused here, and why:

  * :func:`rms_adaln` -- ``F.rms_norm(x) * (1 + scale) + shift``. Eager is
    3 kernels / ~8 tensor-passes; this is 1 kernel / 2 passes.
  * :func:`gated_residual_rms_adaln` -- ``x_new = x + gate * y`` followed by
    ``rms_adaln(x_new, ...)``, returning BOTH. The block needs ``x_new`` for
    the next residual and ``y_norm`` for the next sublayer, so returning both
    avoids re-reading ``x_new`` from HBM. Eager is 4 kernels / ~9 passes;
    this is 1 kernel / 4 passes.
  * :func:`swiglu_ffn` -- ``chunk`` + ``silu`` + ``mul`` collapsed into one
    kernel between two cuBLAS SGEMMs. Eager is 2 kernels / 5 [M, 256]
    passes; this is 1 kernel / 3 passes. The GEMMs themselves are
    deliberately NOT fused -- see the function docstring.
  * :func:`segment_mean_atom_to_token` -- the atom-to-token mean, replacing
    ``scatter_reduce_(reduce="mean", include_self=False)``, which materialises
    an [B, A, d] expanded index tensor and runs a generic gather/scatter.

By default everything is fp32 in, fp32 out, fp32 accumulate. With
``ESMFOLD2_BF16_DIT_GEMM=1`` set before import, the projection and FFN GEMM
operands are bf16, while the residual stream, the statistics and the kernel
arithmetic stay fp32. The GEMMs stay in cuBLAS either way: fp32 ``tl.dot``
has no tensor-core path.

INFERENCE ONLY. None of these entry points is a ``torch.autograd.Function``
and none saves anything for backward. Calling them under ``requires_grad``
silently detaches the graph, so the wrappers refuse inputs that require
grad. Wire them in behind the same "fast backend" switch the other
esmfold2 kernels use.

Tile constants are static, chosen by a one-off offline sweep on L40 and
H100. There is no runtime autotune: a cold autotune pass on the first
diffusion step is unshippable.

Measured per call (median of 25 reps of 25 back-to-back calls, CUDA-event
timed), B=5 A=7328 d=128, fp32:

    op                          L40 eager  L40 fused   H100 eager  H100 fused
    rms_adaln                      50.8       16.6        65.5        16.8
    gated_residual_rms_adaln       76.8       22.5        99.4        29.2
    swiglu_ffn                    562.6      472.5       276.9       236.5
    segment_mean (L=916)           84.5       73.3        91.5        37.4

i.e. 3.1x / 3.4x on the two elementwise fusions (L40), 3.9x / 3.4x (H100).
At that shape ``gated_residual_rms_adaln`` reaches 85% of the card's
same-footprint device-copy ceiling on L40 and 74% on H100 -- the activation
is ~19 MB and fits in L2 on both, so HBM is not the relevant roofline.

Two caveats worth knowing before wiring this in:

  * Below ~B=5 the ops are CPU-dispatch-bound, not GPU-bound: the kernels
    themselves run in 11-29 us but a Python call costs ~17 us regardless of
    shape. Under CUDA-graph capture that overhead disappears and the small
    shapes should track the large ones.
  * :func:`swiglu_ffn` is only a partial fusion, on purpose. See its
    docstring: fully fusing the GEMMs is 18-27x SLOWER at true fp32.
"""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - Triton is a hard dep of the fast path

    class _TritonStub:
        """Keeps ``@triton.jit`` / ``tl.constexpr`` resolvable without Triton.

        The decorated kernels are never launched: every public entry point
        raises :class:`AtomBlockKernelUnavailable` before it gets that far.
        """

        constexpr = None

        def jit(self, fn):
            return fn

        def __getattr__(self, name):
            return None

    triton = _TritonStub()  # ty:ignore[invalid-assignment]
    tl = _TritonStub()  # ty:ignore[invalid-assignment]
    HAS_TRITON = False


class AtomBlockKernelUnavailable(RuntimeError):
    """These inputs cannot run through the fused atom-block kernels.

    Callers that want a silent fallback should catch this and call the eager
    path; it is raised for a missing Triton, a non-CUDA / non-fp32 /
    non-contiguous tensor, a grad-requiring tensor, or an unsupported shape.
    """


# ---------------------------------------------------------------------------
# Tile constants (static; no runtime autotune)
# ---------------------------------------------------------------------------

# Bandwidth-bound elementwise kernels at D=128 fp32 (512 B per row). Swept
# offline over BLOCK_M in {16,32,64,128} x warps in {4,8} x stages in {1,2};
# BLOCK_M=16 / 8 warps won on both L40 and H100 and the stage count was a wash.
_ELEM_BLOCK_M = 16
_ELEM_NUM_WARPS = 8
_ELEM_NUM_STAGES = 2

# SwiGLU activation. Each program owns BLOCK_M rows of the full 2H width.
# L40 is sensitive here (162 us at BLOCK_M=8 vs 135 us at 64); H100 is flat
# within 5% across the whole grid, so 64 is the joint pick.
_FFN_BLOCK_M = 64
_FFN_NUM_WARPS = 8
_FFN_NUM_STAGES = 2

# Segmented atom->token mean. BLOCK_A=32 / 8 warps won on both cards.
_SEG_BLOCK_A = 32
_SEG_NUM_WARPS = 8
_SEG_DIV_BLOCK_R = 32
_SEG_DIV_NUM_WARPS = 4

_SWA_BLOCK_M = 64
_SWA_BLOCK_K = 64
_SWA_NUM_WARPS = 4

# F.rms_norm(x, (D,)) with eps=None uses torch.finfo(x.dtype).eps.
RMS_NORM_EPS_FP32 = 1.1920928955078125e-07


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _rms_adaln_kernel(
    x_ptr,  # [B, A, D] fp32 contiguous
    scale_ptr,  # [B, {1,A}, D] fp32, last dim stride 1
    shift_ptr,
    y_ptr,  # [B, A, D] fp32 contiguous
    M,  # B * A
    A,
    s_sb,
    s_sa,  # scale strides (batch, atom); s_sa == 0 when broadcast over A
    h_sb,
    h_sa,  # shift strides
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    mask_m = offs_m < M
    mask_d = offs_d < D
    mask = mask_m[:, None] & mask_d[None, :]

    x = tl.load(x_ptr + offs_m[:, None] * D + offs_d[None, :], mask=mask, other=0.0)

    b = offs_m // A
    a = offs_m % A
    scale = tl.load(
        scale_ptr + b[:, None] * s_sb + a[:, None] * s_sa + offs_d[None, :],
        mask=mask,
        other=0.0,
    )
    shift = tl.load(
        shift_ptr + b[:, None] * h_sb + a[:, None] * h_sa + offs_d[None, :],
        mask=mask,
        other=0.0,
    )

    ms = tl.sum(x * x, axis=1) / D
    rstd = 1.0 / tl.sqrt(ms + EPS)
    y = (x * rstd[:, None]) * (1.0 + scale) + shift
    tl.store(
        y_ptr + offs_m[:, None] * D + offs_d[None, :],
        y.to(y_ptr.type.element_ty),
        mask=mask,
    )


@triton.jit
def _gated_residual_rms_adaln_kernel(
    x_ptr,  # [B, A, D] residual stream in
    y_ptr,  # [B, A, D] sublayer output
    gate_ptr,
    scale_ptr,
    shift_ptr,
    xnew_ptr,  # [B, A, D] out: x + gate * y
    ynorm_ptr,  # [B, A, D] out: rms_adaln(x_new, scale, shift)
    M,
    A,
    g_sb,
    g_sa,
    s_sb,
    s_sa,
    h_sb,
    h_sa,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    mask_m = offs_m < M
    mask_d = offs_d < D
    mask = mask_m[:, None] & mask_d[None, :]

    lin = offs_m[:, None] * D + offs_d[None, :]
    x = tl.load(x_ptr + lin, mask=mask, other=0.0)
    y = tl.load(y_ptr + lin, mask=mask, other=0.0)

    b = offs_m // A
    a = offs_m % A
    gate = tl.load(
        gate_ptr + b[:, None] * g_sb + a[:, None] * g_sa + offs_d[None, :],
        mask=mask,
        other=0.0,
    )
    scale = tl.load(
        scale_ptr + b[:, None] * s_sb + a[:, None] * s_sa + offs_d[None, :],
        mask=mask,
        other=0.0,
    )
    shift = tl.load(
        shift_ptr + b[:, None] * h_sb + a[:, None] * h_sa + offs_d[None, :],
        mask=mask,
        other=0.0,
    )

    x_new = x + gate * y
    tl.store(xnew_ptr + lin, x_new, mask=mask)

    ms = tl.sum(x_new * x_new, axis=1) / D
    rstd = 1.0 / tl.sqrt(ms + EPS)
    y_norm = (x_new * rstd[:, None]) * (1.0 + scale) + shift
    tl.store(ynorm_ptr + lin, y_norm.to(ynorm_ptr.type.element_ty), mask=mask)


@triton.jit
def _swiglu_act_kernel(
    lin_ptr,  # [M, 2H] fp32 = x @ w_up.T
    out_ptr,  # [M, H] fp32 = silu(lin[:, :H]) * lin[:, H:]
    M,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """``silu(x1) * x2`` over the two halves of the up-projection, one pass.

    Eager runs this as ``chunk`` + ``silu`` + ``mul`` -- two kernels and five
    [M, H] tensor passes against this kernel's three.
    """
    pid = tl.program_id(0).to(tl.int64)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_h = tl.arange(0, BLOCK_H).to(tl.int64)
    mask = (offs_m < M)[:, None] & (offs_h < H)[None, :]

    base = lin_ptr + offs_m[:, None] * (2 * H)
    a = tl.load(base + offs_h[None, :], mask=mask, other=0.0).to(tl.float32)
    b = tl.load(base + H + offs_h[None, :], mask=mask, other=0.0).to(tl.float32)
    y = (a * tl.sigmoid(a)) * b
    tl.store(
        out_ptr + offs_m[:, None] * H + offs_h[None, :],
        y.to(out_ptr.type.element_ty),
        mask=mask,
    )


@triton.jit
def _segment_accum_kernel(
    feat_ptr,  # [B, A, D] fp32
    tok_ptr,  # [B, A] int
    mask_ptr,  # [B, A] bool (ignored when HAS_MASK == 0)
    sum_ptr,  # [B, L, D] fp32, zeroed
    cnt_ptr,  # [B, L] fp32, zeroed
    A,
    L,
    HAS_MASK: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_A: tl.constexpr,
):
    b = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    offs_a = pid * BLOCK_A + tl.arange(0, BLOCK_A).to(tl.int64)
    in_range = offs_a < A

    tok = tl.load(tok_ptr + b * A + offs_a, mask=in_range, other=0).to(tl.int64)
    # Guard rather than trust: an out-of-range token index would otherwise be
    # an unmasked write into neighbouring memory.
    valid = in_range & (tok >= 0) & (tok < L)
    if HAS_MASK:
        m = tl.load(mask_ptr + b * A + offs_a, mask=in_range, other=0)
        valid = valid & (m != 0)
    tok = tl.where(valid, tok, 0)

    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    mask_d = offs_d < D
    mask_2d = valid[:, None] & mask_d[None, :]

    f = tl.load(
        feat_ptr + (b * A + offs_a)[:, None] * D + offs_d[None, :],
        mask=mask_2d,
        other=0.0,
    )
    tl.atomic_add(
        sum_ptr + (b * L + tok)[:, None] * D + offs_d[None, :], f, mask=mask_2d
    )
    tl.atomic_add(
        cnt_ptr + b * L + tok, tl.full((BLOCK_A,), 1.0, tl.float32), mask=valid
    )


@triton.jit
def _segment_divide_kernel(
    sum_ptr,  # [R, D] fp32, divided in place
    cnt_ptr,  # [R] fp32
    R,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs_r = pid * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    mask = (offs_r < R)[:, None] & (offs_d < D)[None, :]

    cnt = tl.load(cnt_ptr + offs_r, mask=offs_r < R, other=0.0)
    # count == 0 means the token got no atoms; the sum row is already 0, and
    # dividing by 1 keeps it 0 -- matching scatter_reduce_(include_self=False).
    denom = tl.where(cnt > 0.0, cnt, 1.0)

    ptrs = sum_ptr + offs_r[:, None] * D + offs_d[None, :]
    val = tl.load(ptrs, mask=mask, other=0.0)
    tl.store(ptrs, val / denom[:, None], mask=mask)


# ---------------------------------------------------------------------------
# Input checking
#
# These run on every call, 12+ times per diffusion step. Keep the happy path
# to one boolean chain and build the (f-string) diagnostics only on failure:
# formatting them eagerly cost ~20 us of CPU per call, which is more than any
# of these kernels spends on the GPU.
# ---------------------------------------------------------------------------


def _grad_tracked(t: torch.Tensor) -> bool:
    """Whether this tensor would actually record a backward here.

    A model parameter has ``requires_grad=True`` even in ``.eval()``, so the
    flag alone is not a reason to refuse: under ``no_grad`` / ``inference_mode``
    nothing is recorded and the kernels are safe. Checking the flag by itself
    made every call refuse on ``w_up`` and cost a fold 4-8%.
    """
    return t.requires_grad and torch.is_grad_enabled()


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AtomBlockKernelUnavailable(msg)


def _raise_activation(name: str, t: torch.Tensor) -> None:
    _require(HAS_TRITON, "Triton is not installed")
    _require(t.is_cuda, f"{name}: expected a CUDA tensor, got device {t.device}")
    _require(t.dtype is torch.float32, f"{name}: expected float32, got {t.dtype}")
    _require(t.is_contiguous(), f"{name}: expected a contiguous tensor")
    _require(
        not _grad_tracked(t), f"{name}: kernels are inference-only (grad is recording)"
    )
    raise AtomBlockKernelUnavailable(f"{name}: unsupported input")


def _check_activation(name: str, t: torch.Tensor) -> None:
    if not (
        HAS_TRITON
        and t.is_cuda
        and t.dtype is torch.float32
        and t.is_contiguous()
        and not _grad_tracked(t)
    ):
        _raise_activation(name, t)


def _check_activation_bf16(name: str, t: torch.Tensor) -> None:
    """Like :func:`_check_activation` but for the bf16 GEMM-operand path."""
    if not (
        HAS_TRITON
        and t.is_cuda
        and t.dtype is torch.bfloat16
        and t.is_contiguous()
        and not _grad_tracked(t)
    ):
        _require(HAS_TRITON, "Triton is not installed")
        _require(t.is_cuda, f"{name}: expected a CUDA tensor, got {t.device}")
        _require(t.is_contiguous(), f"{name}: expected a contiguous tensor")
        _require(
            not _grad_tracked(t),
            f"{name}: kernels are inference-only (grad is recording)",
        )
        raise AtomBlockKernelUnavailable(f"{name}: unsupported input")


def _raise_mod(name: str, t: torch.Tensor, B: int, A: int, D: int) -> None:
    _require(HAS_TRITON, "Triton is not installed")
    _require(t.is_cuda, f"{name}: expected a CUDA tensor, got device {t.device}")
    _require(t.dtype is torch.float32, f"{name}: expected float32, got {t.dtype}")
    _require(
        not _grad_tracked(t), f"{name}: kernels are inference-only (grad is recording)"
    )
    _require(t.dim() == 3, f"{name}: expected a 2D or 3D tensor, got {t.dim()}D")
    _require(
        t.shape[0] == B and t.shape[2] == D,
        f"{name}: expected [{B}, 1 or {A}, {D}], got {tuple(t.shape)}",
    )
    _require(t.shape[1] in (1, A), f"{name}: dim 1 must be 1 or {A}, got {t.shape[1]}")
    _require(t.stride(2) == 1, f"{name}: last dim must be contiguous")
    raise AtomBlockKernelUnavailable(f"{name}: unsupported input")


def _mod_strides(
    name: str, t: torch.Tensor, B: int, A: int, D: int
) -> tuple[torch.Tensor, int, int]:
    """Normalise an adaLN modulation tensor to 3D and return (t, s_b, s_a).

    Accepts [B, D], [B, 1, D] or [B, A, D]. Only the last dim must be
    stride-1: the production caller slices these out of a [B, 1, 6D] tensor
    with ``.chunk(6, dim=-1)``, which is not contiguous.
    """
    if t.dim() == 2:
        t = t.unsqueeze(1)
    shape = t.shape
    if not (
        HAS_TRITON
        and t.is_cuda
        and t.dtype is torch.float32
        and not _grad_tracked(t)
        and len(shape) == 3
        and shape[0] == B
        and shape[2] == D
        and (shape[1] == 1 or shape[1] == A)
        and t.stride(2) == 1
    ):
        _raise_mod(name, t, B, A, D)
    return t, t.stride(0), 0 if shape[1] == 1 else t.stride(1)


# ---------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------


def rms_adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``F.rms_norm(x, (D,)) * (1 + scale) + shift`` in one kernel.

    Inference only; fp32 only.

    Parameters
    ----------
    x : torch.Tensor
        [B, A, D] fp32, contiguous, CUDA.
    scale, shift : torch.Tensor
        [B, D], [B, 1, D] or [B, A, D] fp32; last dim must be stride-1.

    Returns
    -------
    torch.Tensor
        [B, A, D] fp32.

    Raises
    ------
    AtomBlockKernelUnavailable
        If Triton is missing or an input is unsupported.
    """
    _check_activation("x", x)
    if x.dim() != 3:
        raise AtomBlockKernelUnavailable(f"x: expected [B, A, D], got {tuple(x.shape)}")
    B, A, D = x.shape
    scale, s_sb, s_sa = _mod_strides("scale", scale, B, A, D)
    shift, h_sb, h_sa = _mod_strides("shift", shift, B, A, D)

    y = torch.empty_like(x, dtype=out_dtype or x.dtype)
    M = B * A
    grid = (triton.cdiv(M, _ELEM_BLOCK_M),)
    _rms_adaln_kernel[grid](
        x,
        scale,
        shift,
        y,
        M,
        A,
        s_sb,
        s_sa,
        h_sb,
        h_sa,
        D=D,  # ty:ignore[invalid-argument-type]
        BLOCK_D=_next_pow2(D),  # ty:ignore[invalid-argument-type]
        BLOCK_M=_ELEM_BLOCK_M,  # ty:ignore[invalid-argument-type]
        EPS=RMS_NORM_EPS_FP32,  # ty:ignore[invalid-argument-type]
        num_warps=_ELEM_NUM_WARPS,  # ty:ignore[unknown-argument]
        num_stages=_ELEM_NUM_STAGES,  # ty:ignore[unknown-argument]
    )
    return y


def gated_residual_rms_adaln(
    x: torch.Tensor,
    gate: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x_new = x + gate * y`` and ``y_norm = rms_adaln(x_new, scale, shift)``.

    Both outputs come from one pass, so ``x_new`` is never re-read from HBM.

    Inference only; fp32 only.

    Parameters
    ----------
    x, y : torch.Tensor
        [B, A, D] fp32, contiguous, CUDA.
    gate, scale, shift : torch.Tensor
        [B, D], [B, 1, D] or [B, A, D] fp32; last dim must be stride-1.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(x_new, y_norm)``, both [B, A, D] fp32.

    Raises
    ------
    AtomBlockKernelUnavailable
        If Triton is missing or an input is unsupported.
    """
    _check_activation("x", x)
    _check_activation("y", y)
    if x.dim() != 3 or x.shape != y.shape:
        raise AtomBlockKernelUnavailable(
            f"expected matching [B, A, D] x/y, got {tuple(x.shape)} and "
            f"{tuple(y.shape)}"
        )
    B, A, D = x.shape
    gate, g_sb, g_sa = _mod_strides("gate", gate, B, A, D)
    scale, s_sb, s_sa = _mod_strides("scale", scale, B, A, D)
    shift, h_sb, h_sa = _mod_strides("shift", shift, B, A, D)

    x_new = torch.empty_like(x)
    y_norm = torch.empty_like(x, dtype=out_dtype or x.dtype)
    M = B * A
    grid = (triton.cdiv(M, _ELEM_BLOCK_M),)
    _gated_residual_rms_adaln_kernel[grid](
        x,
        y,
        gate,
        scale,
        shift,
        x_new,
        y_norm,
        M,
        A,
        g_sb,
        g_sa,
        s_sb,
        s_sa,
        h_sb,
        h_sa,
        D=D,  # ty:ignore[invalid-argument-type]
        BLOCK_D=_next_pow2(D),  # ty:ignore[invalid-argument-type]
        BLOCK_M=_ELEM_BLOCK_M,  # ty:ignore[invalid-argument-type]
        EPS=RMS_NORM_EPS_FP32,  # ty:ignore[invalid-argument-type]
        num_warps=_ELEM_NUM_WARPS,  # ty:ignore[unknown-argument]
        num_stages=_ELEM_NUM_STAGES,  # ty:ignore[unknown-argument]
    )
    return x_new, y_norm


def swiglu_ffn(
    x: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    """``w_down(silu(x1) * x2)`` where ``(x1, x2) = chunk(w_up(x), 2, -1)``.

    PARTIAL fusion on purpose: cuBLAS SGEMM, one Triton SwiGLU kernel, cuBLAS
    SGEMM. The two GEMMs are NOT folded into the Triton kernel.

    A fully fused single-kernel version was written and measured. At fp32 it
    is a disaster, because true fp32 ``tl.dot`` (``input_precision="ieee"``)
    is scalar FMA on CUDA cores with no tensor-core path, while eager gets
    cuBLAS SGEMM. At B=5 A=7328: 11030 us (L40) and 7037 us (H100) against
    416 us / 296 us eager -- 18-27x slower -- on top of a 19 s one-off JIT
    compile. TF32 or bf16 ``tl.dot`` would fix the speed and is exactly what
    the fp32-diffusion rule forbids, so the GEMMs stay in cuBLAS.

    Inference only. fp32, or bf16 operands with fp32 accumulation in cuBLAS
    and fp32 math inside the Triton kernel.

    Parameters
    ----------
    x : torch.Tensor
        [..., K] fp32 or bf16, contiguous, CUDA; the weights must match.
    w_up : torch.Tensor
        [2H, K] fp32, ``nn.Linear`` layout (``SwiGLUFFN.w_up.weight``).
    w_down : torch.Tensor
        [D_out, H] fp32, ``nn.Linear`` layout (``SwiGLUFFN.w_down.weight``).

    Returns
    -------
    torch.Tensor
        [..., D_out] fp32.

    Raises
    ------
    AtomBlockKernelUnavailable
        If Triton is missing or an input is unsupported.
    """
    if x.dtype is torch.bfloat16:
        for _n, _t in (("x", x), ("w_up", w_up), ("w_down", w_down)):
            if _t.dtype is not torch.bfloat16:
                raise AtomBlockKernelUnavailable(
                    f"{_n}: expected bfloat16 to match x, got {_t.dtype}"
                )
            _check_activation_bf16(_n, _t)
    else:
        _check_activation("x", x)
        _check_activation("w_up", w_up)
        _check_activation("w_down", w_down)
    if w_up.dim() != 2 or w_down.dim() != 2:
        raise AtomBlockKernelUnavailable("w_up/w_down must be 2D")
    two_h, K = w_up.shape
    d_out, H = w_down.shape
    if two_h != 2 * H or x.shape[-1] != K:
        raise AtomBlockKernelUnavailable(
            f"shape mismatch: w_up {tuple(w_up.shape)}, w_down "
            f"{tuple(w_down.shape)}, x {tuple(x.shape)}"
        )

    out_shape = (*x.shape[:-1], d_out)
    x_2d = x.reshape(-1, K)
    M = x_2d.shape[0]

    lin = torch.nn.functional.linear(x_2d, w_up)
    hidden = torch.empty((M, H), device=x.device, dtype=lin.dtype)
    grid = (triton.cdiv(M, _FFN_BLOCK_M),)
    _swiglu_act_kernel[grid](
        lin,
        hidden,
        M,
        H=H,  # ty:ignore[invalid-argument-type]
        BLOCK_H=_next_pow2(H),  # ty:ignore[invalid-argument-type]
        BLOCK_M=_FFN_BLOCK_M,  # ty:ignore[invalid-argument-type]
        num_warps=_FFN_NUM_WARPS,  # ty:ignore[unknown-argument]
        num_stages=_FFN_NUM_STAGES,  # ty:ignore[unknown-argument]
    )
    return torch.nn.functional.linear(hidden, w_down).view(*out_shape)


def segment_mean_atom_to_token(
    atom_features: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    n_tokens: int,
    atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-token mean of atom features -- drop-in for ``scatter_atom_to_token``.

    Same semantics as
    ``scatter_reduce_(1, idx, src, reduce="mean", include_self=False)``:
    a token with no (unmasked) atoms keeps a zero row.

    Implemented as a masked fp32 atomic accumulate plus a divide, so it makes
    no assumption that each token's atoms are contiguous in A. Like the eager
    scatter it is therefore run-to-run non-deterministic in the low bits.

    Inference only; fp32 only.

    Parameters
    ----------
    atom_features : torch.Tensor
        [B, A, D] fp32, contiguous, CUDA.
    atom_to_token_idx : torch.Tensor
        [B, A] int32/int64, contiguous, CUDA. Indices outside
        ``[0, n_tokens)`` are dropped.
    n_tokens : int
        L.
    atom_mask : torch.Tensor or None
        [B, A] bool, contiguous, CUDA. False atoms contribute nothing.

    Returns
    -------
    torch.Tensor
        [B, L, D] fp32.

    Raises
    ------
    AtomBlockKernelUnavailable
        If Triton is missing or an input is unsupported.
    """
    _check_activation("atom_features", atom_features)
    if atom_features.dim() != 3:
        raise AtomBlockKernelUnavailable(
            f"atom_features: expected [B, A, D], got {tuple(atom_features.shape)}"
        )
    B, A, D = atom_features.shape
    if not (
        atom_to_token_idx.is_cuda
        and atom_to_token_idx.dtype in (torch.int32, torch.int64)
        and atom_to_token_idx.is_contiguous()
        and atom_to_token_idx.shape == atom_features.shape[:2]
    ):
        raise AtomBlockKernelUnavailable(
            f"atom_to_token_idx: expected a contiguous CUDA int32/int64 "
            f"[{B}, {A}], got {atom_to_token_idx.dtype} "
            f"{tuple(atom_to_token_idx.shape)} on {atom_to_token_idx.device}"
        )
    if atom_mask is not None and not (
        atom_mask.is_cuda
        and atom_mask.is_contiguous()
        and atom_mask.shape == atom_features.shape[:2]
    ):
        raise AtomBlockKernelUnavailable(
            f"atom_mask: expected a contiguous CUDA [{B}, {A}], got "
            f"{tuple(atom_mask.shape)} on {atom_mask.device}"
        )

    out = torch.zeros(
        (B, n_tokens, D), device=atom_features.device, dtype=torch.float32
    )
    cnt = torch.zeros((B, n_tokens), device=atom_features.device, dtype=torch.float32)

    grid = (triton.cdiv(A, _SEG_BLOCK_A), B)
    _segment_accum_kernel[grid](
        atom_features,
        atom_to_token_idx,
        atom_mask if atom_mask is not None else atom_to_token_idx,
        out,
        cnt,
        A,
        n_tokens,
        HAS_MASK=atom_mask is not None,  # ty:ignore[invalid-argument-type]
        D=D,  # ty:ignore[invalid-argument-type]
        BLOCK_D=_next_pow2(D),  # ty:ignore[invalid-argument-type]
        BLOCK_A=_SEG_BLOCK_A,  # ty:ignore[invalid-argument-type]
        num_warps=_SEG_NUM_WARPS,  # ty:ignore[unknown-argument]
    )

    R = B * n_tokens
    div_grid = (triton.cdiv(R, _SEG_DIV_BLOCK_R),)
    _segment_divide_kernel[div_grid](
        out,
        cnt,
        R,
        D=D,  # ty:ignore[invalid-argument-type]
        BLOCK_D=_next_pow2(D),  # ty:ignore[invalid-argument-type]
        BLOCK_R=_SEG_DIV_BLOCK_R,  # ty:ignore[invalid-argument-type]
        num_warps=_SEG_DIV_NUM_WARPS,  # ty:ignore[unknown-argument]
    )
    return out


@triton.jit
def _qk_norm_rope_kernel(
    qkv_ptr,  # [B, N, 3, H, D] fp32 contiguous
    cos_ptr,  # [B, N, D/2] fp32 contiguous
    sin_ptr,
    q_ptr,  # [B, N, H, D] out
    k_ptr,
    M,  # B * N * H
    H,
    HD,  # H * D
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EPS: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    mask_m = offs_m < M
    mask_d = offs_d < D
    mask = mask_m[:, None] & mask_d[None, :]

    bn = offs_m // H  # b * N + n, which indexes cos/sin
    h = offs_m % H
    q_base = (bn * 3 + 0) * HD + h * D
    k_base = (bn * 3 + 1) * HD + h * D

    q = tl.load(qkv_ptr + q_base[:, None] + offs_d[None, :], mask=mask, other=0.0)
    k = tl.load(qkv_ptr + k_base[:, None] + offs_d[None, :], mask=mask, other=0.0)
    q_rstd = 1.0 / tl.sqrt(tl.sum(q * q, axis=1) / D + EPS)
    k_rstd = 1.0 / tl.sqrt(tl.sum(k * k, axis=1) / D + EPS)

    # cos/sin are [D/2] tiled twice, so lane i reads index i % (D/2) and pairs
    # with lane i +/- D/2 -- that pairing is rotate_half.
    half: tl.constexpr = D // 2
    is_low = offs_d < half
    cs = tl.where(is_low, offs_d, offs_d - half)
    cosv = tl.load(cos_ptr + bn[:, None] * half + cs[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    sinv = tl.load(sin_ptr + bn[:, None] * half + cs[None, :], mask=mask, other=0.0).to(
        tl.float32
    )

    partner = tl.where(is_low, offs_d + half, offs_d - half)
    qp = tl.load(qkv_ptr + q_base[:, None] + partner[None, :], mask=mask, other=0.0)
    kp = tl.load(qkv_ptr + k_base[:, None] + partner[None, :], mask=mask, other=0.0)

    qn = q * q_rstd[:, None]
    kn = k * k_rstd[:, None]
    qpn = qp * q_rstd[:, None]
    kpn = kp * k_rstd[:, None]
    q_out = qn * cosv + tl.where(is_low[None, :], -qpn, qpn) * sinv
    k_out = kn * cosv + tl.where(is_low[None, :], -kpn, kpn) * sinv

    if OUT_BF16:
        q_out = q_out.to(tl.bfloat16)
        k_out = k_out.to(tl.bfloat16)
    out_off = offs_m[:, None] * D + offs_d[None, :]
    tl.store(q_ptr + out_off, q_out, mask=mask)
    tl.store(k_ptr + out_off, k_out, mask=mask)


def qk_norm_rope(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_heads: int,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rms_norm`` then 3D RoPE on q and k, read straight out of ``Wqkv``'s output.

    Replaces four elementwise passes (two ``qk_norm``, two ``apply_rotary_emb_3d``)
    and their temporaries with one. Emits ``out_dtype`` directly, so the usual
    ``.bfloat16()`` before flash-attn is already paid for.

    Parameters
    ----------
    qkv : torch.Tensor
        [B, N, 3 * n_heads * D] fp32, contiguous, CUDA -- ``Wqkv(x)`` unreshaped.
    cos, sin : torch.Tensor
        [B, N, D / 2] fp32 or bf16, contiguous. ``build_3d_rope`` emits bf16 and
        the eager path promotes it, so the kernel upcasts to fp32 to match.
        ``D / 2`` fixes the rotated width, which must be the whole head: a
        partial rotation is not supported.
    n_heads : int
        Head count; D is inferred as ``qkv.shape[-1] // (3 * n_heads)``.
    out_dtype : torch.dtype
        ``torch.float32`` or ``torch.bfloat16``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        q and k, each [B, N, n_heads, D] contiguous in ``out_dtype``.

    Raises
    ------
    AtomBlockKernelUnavailable
        If Triton is missing or an input is unsupported.
    """
    _check_activation("qkv", qkv)
    for _name, _t in (("cos", cos), ("sin", sin)):
        if not (
            _t.is_cuda
            and _t.dtype in (torch.float32, torch.bfloat16)
            and _t.is_contiguous()
            and not _grad_tracked(_t)
        ):
            _require(_t.is_cuda, f"{_name}: expected a CUDA tensor, got {_t.device}")
            _require(
                _t.dtype in (torch.float32, torch.bfloat16),
                f"{_name}: expected float32 or bfloat16, got {_t.dtype}",
            )
            _require(_t.is_contiguous(), f"{_name}: expected a contiguous tensor")
            raise AtomBlockKernelUnavailable(f"{_name}: unsupported input")
    if cos.dtype is not sin.dtype:
        raise AtomBlockKernelUnavailable(
            f"cos/sin dtype mismatch: {cos.dtype} vs {sin.dtype}"
        )
    if qkv.dim() != 3:
        raise AtomBlockKernelUnavailable(
            f"qkv: expected [B, N, 3 * H * D], got {tuple(qkv.shape)}"
        )
    if out_dtype not in (torch.float32, torch.bfloat16):
        raise AtomBlockKernelUnavailable(f"out_dtype: {out_dtype} is not supported")
    B, N, three_hd = qkv.shape
    if three_hd % (3 * n_heads):
        raise AtomBlockKernelUnavailable(
            f"qkv: last dim {three_hd} is not 3 * {n_heads} * D"
        )
    D = three_hd // (3 * n_heads)
    if D % 2:
        raise AtomBlockKernelUnavailable(f"head dim {D} is odd")
    if cos.shape != (B, N, D // 2) or sin.shape != cos.shape:
        raise AtomBlockKernelUnavailable(
            f"cos/sin: expected {(B, N, D // 2)}, got {tuple(cos.shape)} "
            f"and {tuple(sin.shape)}"
        )

    q = torch.empty((B, N, n_heads, D), dtype=out_dtype, device=qkv.device)
    k = torch.empty_like(q)
    M = B * N * n_heads
    grid = (triton.cdiv(M, _ELEM_BLOCK_M),)
    _qk_norm_rope_kernel[grid](
        qkv,
        cos,
        sin,
        q,
        k,
        M,
        n_heads,
        n_heads * D,
        D=D,  # ty:ignore[invalid-argument-type]
        BLOCK_D=_next_pow2(D),  # ty:ignore[invalid-argument-type]
        BLOCK_M=_ELEM_BLOCK_M,  # ty:ignore[invalid-argument-type]
        EPS=RMS_NORM_EPS_FP32,  # ty:ignore[invalid-argument-type]
        OUT_BF16=out_dtype is torch.bfloat16,  # ty:ignore[invalid-argument-type]
        num_warps=_ELEM_NUM_WARPS,  # ty:ignore[unknown-argument]
        num_stages=_ELEM_NUM_STAGES,  # ty:ignore[unknown-argument]
    )
    return q, k


@triton.jit
def _rms_rope_halves(a1, a2, cs, sn, HD: tl.constexpr, EPS: tl.constexpr):
    rstd = 1.0 / tl.sqrt((tl.sum(a1 * a1, axis=1) + tl.sum(a2 * a2, axis=1)) / HD + EPS)
    a1 = a1 * rstd[:, None]
    a2 = a2 * rstd[:, None]
    return (a1 * cs - a2 * sn).to(tl.bfloat16), (a2 * cs + a1 * sn).to(tl.bfloat16)


@triton.jit
def _swa_attention_kernel(
    qkvg_ptr,
    cos_ptr,
    sin_ptr,
    seqlen_ptr,
    o_ptr,
    N,
    n_tiles,
    scale,
    H: tl.constexpr,
    HD: tl.constexpr,
    HW: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    b = pid // n_tiles
    p0 = (pid % n_tiles) * BM
    D: tl.constexpr = H * HD
    HALF: tl.constexpr = HD // 2
    prow = p0 + tl.arange(0, BM)
    in_row = prow < N
    rows = (b * N + prow).to(tl.int64)
    n_valid = tl.load(seqlen_ptr + b)
    c = tl.arange(0, HALF)
    d = tl.arange(0, HD)

    q_ptr = qkvg_ptr + rows[:, None] * (4 * D) + h * HD
    cs = tl.load(
        cos_ptr + rows[:, None] * HALF + c[None, :], mask=in_row[:, None], other=0.0
    )
    sn = tl.load(
        sin_ptr + rows[:, None] * HALF + c[None, :], mask=in_row[:, None], other=0.0
    )
    q1, q2 = _rms_rope_halves(
        tl.load(q_ptr + c[None, :], mask=in_row[:, None], other=0.0).to(tl.float32),
        tl.load(q_ptr + HALF + c[None, :], mask=in_row[:, None], other=0.0).to(
            tl.float32
        ),
        cs.to(tl.float32),
        sn.to(tl.float32),
        HD,
        EPS,
    )

    m_i = tl.full((BM,), -1e30, dtype=tl.float32)
    l_i = tl.zeros((BM,), dtype=tl.float32)
    acc = tl.zeros((BM, HD), dtype=tl.float32)
    hi = tl.minimum(p0 + BM + HW, n_valid)
    for j0 in range(tl.maximum(p0 - HW, 0), hi, BK):
        kpos = j0 + tl.arange(0, BK)
        k_in = kpos < hi
        krows = (b * N + kpos).to(tl.int64)
        k_ptr = qkvg_ptr + krows[:, None] * (4 * D) + D + h * HD
        csk = tl.load(
            cos_ptr + krows[:, None] * HALF + c[None, :], mask=k_in[:, None], other=0.0
        )
        snk = tl.load(
            sin_ptr + krows[:, None] * HALF + c[None, :], mask=k_in[:, None], other=0.0
        )
        k1, k2 = _rms_rope_halves(
            tl.load(k_ptr + c[None, :], mask=k_in[:, None], other=0.0).to(tl.float32),
            tl.load(k_ptr + HALF + c[None, :], mask=k_in[:, None], other=0.0).to(
                tl.float32
            ),
            csk.to(tl.float32),
            snk.to(tl.float32),
            HD,
            EPS,
        )
        v = tl.load(k_ptr + D + d[None, :], mask=k_in[:, None], other=0.0).to(
            tl.bfloat16
        )
        s = (tl.dot(q1, tl.trans(k1)) + tl.dot(q2, tl.trans(k2))) * scale
        diff = kpos[None, :] - prow[:, None]
        allowed = k_in[None, :] & (diff <= HW) & (diff >= -HW)
        s = tl.where(allowed, s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.where(allowed, tl.exp(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    g = tl.load(q_ptr + 3 * D + d[None, :], mask=in_row[:, None], other=0.0).to(
        tl.float32
    )
    o = acc / tl.maximum(l_i, 1e-30)[:, None] * tl.sigmoid(g)
    o = tl.where((prow < n_valid)[:, None], o, 0.0)
    tl.store(
        o_ptr + rows[:, None] * D + h * HD + d[None, :],
        o.to(o_ptr.type.element_ty),
        mask=in_row[:, None],
    )


def swa_attention(
    qkvg: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen: torch.Tensor,
    n_heads: int,
    half_window: int,
    scale: float,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Gated sliding-window attention with qk rms-norm and 3D RoPE, off one fused projection.

    Parameters
    ----------
    qkvg : torch.Tensor
        [B, N, 4 * D] fp32 or bf16 ``q | k | v | gate`` projection, contiguous, CUDA.
    cos, sin : torch.Tensor
        [B, N, D / n_heads / 2] fp32 or bf16, contiguous.
    seqlen : torch.Tensor
        [B] int32; each row's valid atoms are its first ``seqlen[b]``.
    n_heads, half_window : int
        Head count and one-sided window.
    scale : float
        Softmax scale.
    out_dtype : torch.dtype
        ``torch.float32`` or ``torch.bfloat16``.

    Returns
    -------
    torch.Tensor
        [B, N, D] ``sigmoid(gate) * attention``, zero past ``seqlen``.

    Raises
    ------
    AtomBlockKernelUnavailable
        If Triton is missing or an input is unsupported.
    """
    _require(HAS_TRITON, "Triton is not installed")
    for _name, _t in (("qkvg", qkvg), ("cos", cos), ("sin", sin)):
        _require(
            _t.is_cuda
            and _t.dtype in (torch.float32, torch.bfloat16)
            and _t.is_contiguous()
            and not _grad_tracked(_t),
            f"{_name}: expected a contiguous CUDA fp32/bf16 inference tensor",
        )
    _require(qkvg.dim() == 3, f"qkvg: expected [B, N, 4 * D], got {tuple(qkvg.shape)}")
    B, N, four_d = qkvg.shape
    _require(four_d % (4 * n_heads) == 0, f"qkvg: last dim {four_d} is not 4 * H * HD")
    hd = four_d // (4 * n_heads)
    _require(
        hd >= 32 and hd & (hd - 1) == 0, f"head dim {hd} must be a power of 2 >= 32"
    )
    _require(
        cos.shape == (B, N, hd // 2) and sin.shape == cos.shape,
        f"cos/sin: expected {(B, N, hd // 2)}, got {tuple(cos.shape)}",
    )
    _require(
        seqlen.is_cuda and seqlen.dtype is torch.int32 and seqlen.shape == (B,),
        f"seqlen: expected a CUDA int32 [{B}]",
    )
    _require(out_dtype in (torch.float32, torch.bfloat16), f"out_dtype {out_dtype}")

    o = torch.empty((B, N, four_d // 4), dtype=out_dtype, device=qkvg.device)
    n_tiles = triton.cdiv(N, _SWA_BLOCK_M)
    _swa_attention_kernel[(B * n_tiles, n_heads)](
        qkvg,
        cos,
        sin,
        seqlen,
        o,
        N,
        n_tiles,
        scale,
        H=n_heads,  # ty:ignore[invalid-argument-type]
        HD=hd,  # ty:ignore[invalid-argument-type]
        HW=half_window,  # ty:ignore[invalid-argument-type]
        BM=_SWA_BLOCK_M,  # ty:ignore[invalid-argument-type]
        BK=_SWA_BLOCK_K,  # ty:ignore[invalid-argument-type]
        EPS=RMS_NORM_EPS_FP32,  # ty:ignore[invalid-argument-type]
        num_warps=_SWA_NUM_WARPS,  # ty:ignore[unknown-argument]
    )
    return o
