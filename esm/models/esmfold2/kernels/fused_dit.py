"""Fused fp32 Triton kernels for the elementwise chains of the DiT step.

The token diffusion transformer runs 12 blocks per diffusion step, each one
attention half plus one transition half. Outside the GEMMs and the attention
every half is a short bandwidth-bound chain on the ``[B * L, 768]`` residual
stream: an adaptive LayerNorm, a sigmoid gate, a residual add, a SwiGLU. Eager
runs each link as its own kernel with its own HBM round-trip.

What is fused here, and why:

  * :func:`adaln` -- ``sigmoid(cs) * LayerNorm(x) + ch``. Eager is 4 kernels
    (layer_norm, sigmoid, mul, add) and ~10 tensor passes; this is 1 kernel
    and 4 passes. The ``LayerNorm(s)`` that produces ``cs``/``ch`` is hoisted
    to the caller, which runs it once for all 24 adaLNs instead of 24 times.
  * :func:`gate_residual_adaln` -- ``x += sigmoid(og) * y`` followed by the
    *next* sublayer's adaLN of that sum, returning both. The block needs
    ``x`` for the next residual and the normed copy for the next sublayer, so
    emitting both from one pass means ``x`` is never re-read from HBM. Eager
    is 7 kernels and ~18 passes; this is 1 kernel and 7 passes.
  * :func:`swiglu` -- ``silu(a) * b`` over the two halves of a packed
    up-projection, collapsing ``chunk`` + ``silu`` + ``mul`` into one kernel
    between two cuBLAS SGEMMs. The GEMMs are deliberately NOT fused in; see
    below.

Conditioning is shared across diffusion samples. Within one diffusion step
every sample of an input sees the same noise level, so the conditioning ``s``
is one tensor repeated ``S`` times and its projections are identical. The
caller therefore computes them at ``[B_cond * L, *]`` while the residual
stream stays ``[B * L, D]`` with ``B = B_cond * S``. These kernels index the
conditioning accordingly: stream row ``r`` reads conditioning row
``(r // (S * L)) * L + r % L``. That is the whole reason the kernels take
``seq_len`` -- they need it to recover the sample structure from a flat row
index.

By default everything is fp32 in, fp32 out, fp32 accumulate. With
``ESMFOLD2_BF16_DIT_GEMM=1`` set before import, the GEMM and SDPA operands
are bf16. The residual stream, the normalisation statistics and the kernel
arithmetic stay fp32, which is what keeps rounding from compounding across
sampler steps. The GEMMs stay in cuBLAS either way: an IEEE-fp32 ``tl.dot``
has no tensor-core path on NVIDIA hardware.

These kernels are NOT bitwise against the eager path (the LayerNorm
reduction order differs, and ``tl.sigmoid`` is not ATen's), so they belong
behind the "fast" backend and never behind "fused".

INFERENCE ONLY. None of these entry points is a ``torch.autograd.Function``
and none saves anything for backward. Calling them where a backward would be
recorded silently detaches the graph, so the wrappers refuse that case. Note
that ``requires_grad`` alone is not a refusal reason: model parameters carry
it even in ``.eval()``, and refusing on the flag by itself made an earlier
kernel reject every call and cost a fold 4-8%.

Measured per call in microseconds, B=5 L=1000 D=768 fp32, median of 25 reps
of 25 back-to-back calls. The eager arm carries the conditioning expanded per
sample, because that is what the eager block actually holds:

    op                      L40 eager  L40 fused   H100 eager  H100 fused
    adaln                         41.9       16.5        58.9        16.7
    gate_residual_adaln          188.4       29.1       102.9        38.0
    swiglu  (H=1536)             141.1       17.8        67.9        34.0

Over a whole 12-block step at that shape the elementwise work drops from 216
eager launches to 50, and total kernel device time goes 77.5 -> 59.0 ms on L40
and 39.3 -> 30.2 ms on H100 -- most of which is the conditioning GEMMs the
caller saves, not these kernels.
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
        raises :class:`DitKernelUnavailable` before it gets that far.
        """

        constexpr = None

        def jit(self, fn):
            return fn

        def __getattr__(self, name):
            return None

    triton = _TritonStub()  # ty:ignore[invalid-assignment]
    tl = _TritonStub()  # ty:ignore[invalid-assignment]
    HAS_TRITON = False


class DitKernelUnavailable(RuntimeError):
    """These inputs cannot run through the fused DiT kernels.

    Callers that want a silent fallback should catch this and run the eager
    block; it is raised for a missing Triton, a non-CUDA / non-fp32 /
    non-contiguous tensor, a tensor whose gradient would be recorded, or a
    row count that is not a whole number of conditioning rows.
    """


# ---------------------------------------------------------------------------
# Launch constants (static; there is no runtime autotune, because a cold
# autotune pass on the first diffusion step is unshippable for inference).
#
# One program per row of the residual stream. Swept offline at B=5 L=1000 over
# num_warps in {2,4,8,16}. At D=768 the three low settings are flat within 4%
# on both cards (L40 16.9/16.6/16.3 us, H100 16.8/16.7/16.6) and 16 regresses
# 25-48%. At H=1536 L40 mildly prefers 16 (17.1 vs 18.1 us at 8) and H100 is
# flat within 1%; the SwiGLU is 2.6% of a step, so the 6% is not worth a
# card-dependent constant.
# ---------------------------------------------------------------------------


def _row_warps(width: int) -> int:
    return 4 if width <= 1024 else 8


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _cond_row(row, L, SAMPLES):
    """Conditioning row feeding stream row ``row``.

    The stream is ``[B_cond, SAMPLES, L, D]`` flattened over the first three
    dims; the conditioning is ``[B_cond, L, D]`` flattened over the first two.
    """
    return (row // (SAMPLES * L)) * L + row % L


@triton.jit
def _adaln_row(x, cs_ptr, ch_ptr, crow, stride_c, offs, d_mask, D: tl.constexpr, EPS):
    """``sigmoid(cs) * LayerNorm(x) + ch`` for one row (LayerNorm, no affine)."""
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(d_mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    x_hat = xc * (1.0 / tl.sqrt(var + EPS))
    cs = tl.load(cs_ptr + crow * stride_c + offs, mask=d_mask, other=0.0)
    ch = tl.load(ch_ptr + crow * stride_c + offs, mask=d_mask, other=0.0)
    return tl.sigmoid(cs) * x_hat + ch


@triton.jit
def _adaln_kernel(
    x_ptr,  # [B * L, D] fp32 contiguous
    cs_ptr,  # [B_cond * L, D] fp32, column slice of a wider tensor
    ch_ptr,
    out_ptr,  # [B * L, D] fp32 contiguous
    L,
    SAMPLES,
    stride_c,
    EPS,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    crow = _cond_row(row, L, SAMPLES)
    offs = tl.arange(0, BLOCK_D)
    d_mask = offs < D
    x = tl.load(x_ptr + row * D + offs, mask=d_mask, other=0.0)
    y = _adaln_row(x, cs_ptr, ch_ptr, crow, stride_c, offs, d_mask, D, EPS)
    tl.store(out_ptr + row * D + offs, y.to(out_ptr.type.element_ty), mask=d_mask)


@triton.jit
def _gate_residual_adaln_kernel(
    x_ptr,  # [B * L, D] residual stream, updated in place
    y_ptr,  # [B * L, D] sublayer output
    og_ptr,  # [B_cond * L, D] output-gate logits
    cs_ptr,  # [B_cond * L, D] next adaLN gate logits (unused if not HAS_ADALN)
    ch_ptr,  # [B_cond * L, D] next adaLN shift
    x_out_ptr,
    a_out_ptr,
    L,
    SAMPLES,
    stride_og,
    stride_c,
    EPS,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_ADALN: tl.constexpr,
):
    """``x += sigmoid(og) * y``, then optionally the next sublayer's adaLN of ``x``."""
    row = tl.program_id(0).to(tl.int64)
    crow = _cond_row(row, L, SAMPLES)
    offs = tl.arange(0, BLOCK_D)
    d_mask = offs < D
    x = tl.load(x_ptr + row * D + offs, mask=d_mask, other=0.0)
    y = tl.load(y_ptr + row * D + offs, mask=d_mask, other=0.0).to(tl.float32)
    og = tl.load(og_ptr + crow * stride_og + offs, mask=d_mask, other=0.0)
    x = x + tl.sigmoid(og) * y
    tl.store(x_out_ptr + row * D + offs, x, mask=d_mask)
    if HAS_ADALN:
        a = _adaln_row(x, cs_ptr, ch_ptr, crow, stride_c, offs, d_mask, D, EPS)
        tl.store(
            a_out_ptr + row * D + offs, a.to(a_out_ptr.type.element_ty), mask=d_mask
        )


@triton.jit
def _swiglu_kernel(ab_ptr, out_ptr, HIDDEN: tl.constexpr, BLOCK_H: tl.constexpr):
    """``silu(a) * b`` for ``[a | b]`` packed on the last dim."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_H)
    h_mask = offs < HIDDEN
    a = tl.load(ab_ptr + row * 2 * HIDDEN + offs, mask=h_mask, other=0.0).to(tl.float32)
    b = tl.load(ab_ptr + row * 2 * HIDDEN + HIDDEN + offs, mask=h_mask, other=0.0).to(
        tl.float32
    )
    y = a * tl.sigmoid(a) * b
    tl.store(out_ptr + row * HIDDEN + offs, y.to(out_ptr.type.element_ty), mask=h_mask)


# ---------------------------------------------------------------------------
# Input checking
#
# These run 37 times per diffusion step. Keep the happy path to one boolean
# chain and build the (f-string) diagnostics only on failure: formatting them
# eagerly costs ~20 us of CPU per call, more than these kernels spend on the
# GPU at short sequence lengths.
# ---------------------------------------------------------------------------


def _grad_tracked(t: torch.Tensor) -> bool:
    """Whether this tensor would actually record a backward here.

    A model parameter has ``requires_grad=True`` even in ``.eval()``, so the
    flag alone is not a reason to refuse: under ``no_grad`` / ``inference_mode``
    nothing is recorded and the kernels are safe. Checking the flag by itself
    made an earlier kernel refuse every call and cost a fold 4-8%.
    """
    return t.requires_grad and torch.is_grad_enabled()


def _raise_rows(name: str, t: torch.Tensor) -> None:
    if not HAS_TRITON:
        raise DitKernelUnavailable("Triton is not installed")
    if not t.is_cuda:
        raise DitKernelUnavailable(f"{name}: expected a CUDA tensor, got {t.device}")
    if t.dtype is not torch.float32:
        raise DitKernelUnavailable(f"{name}: expected float32, got {t.dtype}")
    if t.dim() != 2:
        raise DitKernelUnavailable(f"{name}: expected [rows, D], got {tuple(t.shape)}")
    if t.stride(1) != 1:
        raise DitKernelUnavailable(f"{name}: last dim must be contiguous")
    if _grad_tracked(t):
        raise DitKernelUnavailable(f"{name}: inference-only (grad is recording)")
    raise DitKernelUnavailable(f"{name}: unsupported input")


def _check_stream(name: str, t: torch.Tensor) -> None:
    """A ``[rows, D]`` fp32 CUDA residual-stream tensor, fully contiguous."""
    if not (
        HAS_TRITON
        and t.is_cuda
        and t.dtype is torch.float32
        and t.dim() == 2
        and t.is_contiguous()
        and not _grad_tracked(t)
    ):
        _raise_rows(name, t)


def _check_stream_bf16(name: str, t: torch.Tensor) -> None:
    """A ``[rows, D]`` bf16 CUDA tensor: what the bf16-GEMM path hands the kernels."""
    if not (
        HAS_TRITON
        and t.is_cuda
        and t.dtype is torch.bfloat16
        and t.dim() == 2
        and t.is_contiguous()
        and not _grad_tracked(t)
    ):
        if t.dtype is not torch.bfloat16:
            raise DitKernelUnavailable(f"{name}: expected bfloat16, got {t.dtype}")
        _raise_rows(name, t)


def _check_cond(name: str, t: torch.Tensor, rows: int, d: int) -> None:
    """A ``[rows, D]`` fp32 CUDA conditioning slice; only the last dim need be dense.

    The caller slices these out of one wide packed projection with
    ``cs[:, i : i + D]``, so ``stride(0)`` is the packed width, not ``D``.
    """
    if not (
        HAS_TRITON
        and t.is_cuda
        and t.dtype is torch.float32
        and t.dim() == 2
        and t.shape == (rows, d)
        and t.stride(1) == 1
        and not _grad_tracked(t)
    ):
        if t.dim() == 2 and t.shape != (rows, d):
            raise DitKernelUnavailable(
                f"{name}: expected [{rows}, {d}], got {tuple(t.shape)}"
            )
        _raise_rows(name, t)


def _cond_samples(rows: int, cond_rows: int) -> int:
    if cond_rows <= 0 or rows % cond_rows:
        raise DitKernelUnavailable(
            f"{rows} stream rows are not a whole multiple of {cond_rows} "
            f"conditioning rows"
        )
    return rows // cond_rows


# ---------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------


@torch._dynamo.disable
def adaln(
    x: torch.Tensor,
    cs: torch.Tensor,
    ch: torch.Tensor,
    seq_len: int,
    eps: float,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``sigmoid(cs) * LayerNorm(x) + ch`` row-wise, in one kernel.

    ``cs`` / ``ch`` broadcast over the diffusion samples of an input.
    Inference only; fp32 only.

    Parameters
    ----------
    x : torch.Tensor
        ``[B * seq_len, D]`` fp32, contiguous, CUDA.
    cs, ch : torch.Tensor
        ``[B_cond * seq_len, D]`` fp32 CUDA column slices of one wider packed
        projection; only the last dim must be stride-1. ``cs`` is the gate
        *logit*: the sigmoid is applied inside the kernel.
    seq_len : int
        L. Needed to map a stream row to its conditioning row.
    eps : float
        LayerNorm epsilon.

    Returns
    -------
    torch.Tensor
        ``[B * seq_len, D]`` fp32.

    Raises
    ------
    DitKernelUnavailable
        If Triton is missing or an input is unsupported.
    """
    _check_stream("x", x)
    rows, d = x.shape
    if seq_len <= 0 or rows % seq_len:
        raise DitKernelUnavailable(f"{rows} rows is not a multiple of L={seq_len}")
    cond_rows = cs.shape[0] if cs.dim() == 2 else -1
    _check_cond("cs", cs, cond_rows, d)
    _check_cond("ch", ch, cond_rows, d)
    if cond_rows % seq_len:
        raise DitKernelUnavailable(
            f"{cond_rows} conditioning rows is not a multiple of L={seq_len}"
        )
    if cs.stride(0) != ch.stride(0):
        raise DitKernelUnavailable("cs and ch must share a row stride")
    samples = _cond_samples(rows // seq_len, cond_rows // seq_len)

    out = torch.empty_like(x, dtype=out_dtype or x.dtype)
    _adaln_kernel[(rows,)](
        x,
        cs,
        ch,
        out,
        seq_len,
        samples,
        cs.stride(0),
        eps,
        D=d,  # ty:ignore[invalid-argument-type]
        BLOCK_D=triton.next_power_of_2(d),
        num_warps=_row_warps(d),  # ty:ignore[unknown-argument]
    )
    return out


@torch._dynamo.disable
def gate_residual_adaln(
    x: torch.Tensor,
    y: torch.Tensor,
    og: torch.Tensor,
    seq_len: int,
    eps: float,
    cs: torch.Tensor | None = None,
    ch: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``x += sigmoid(og) * y``, plus the next adaLN of that sum when asked.

    Both outputs come from one pass, so the updated ``x`` is never re-read
    from HBM. ``x`` is updated IN PLACE and also returned.

    Inference only; fp32 only.

    Parameters
    ----------
    x, y : torch.Tensor
        ``[B * seq_len, D]`` fp32, contiguous, CUDA. ``x`` is written.
    og : torch.Tensor
        ``[B_cond * seq_len, D]`` fp32 output-gate *logits*; sigmoid inside.
    seq_len : int
        L.
    eps : float
        LayerNorm epsilon for the trailing adaLN.
    cs, ch : torch.Tensor or None
        The next sublayer's adaLN gate logits and shift, same layout as
        ``og``. Pass ``None`` for both to skip the trailing adaLN (the last
        block of the stack has no next sublayer).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor or None]
        ``(x, adaln(x))``, the second being ``None`` when ``cs`` is ``None``.

    Raises
    ------
    DitKernelUnavailable
        If Triton is missing or an input is unsupported.
    """
    _check_stream("x", x)  # the residual stream is fp32, always
    if y.dtype is torch.bfloat16:
        _check_stream_bf16("y", y)  # a bf16 GEMM output, upcast on load
    else:
        _check_stream("y", y)
    if x.shape != y.shape:
        raise DitKernelUnavailable(
            f"x and y must match, got {tuple(x.shape)} and {tuple(y.shape)}"
        )
    rows, d = x.shape
    if seq_len <= 0 or rows % seq_len:
        raise DitKernelUnavailable(f"{rows} rows is not a multiple of L={seq_len}")
    cond_rows = og.shape[0] if og.dim() == 2 else -1
    _check_cond("og", og, cond_rows, d)
    if cond_rows % seq_len:
        raise DitKernelUnavailable(
            f"{cond_rows} conditioning rows is not a multiple of L={seq_len}"
        )
    has_adaln = cs is not None
    if has_adaln != (ch is not None):
        raise DitKernelUnavailable("cs and ch must both be given or both be None")
    if has_adaln:
        assert cs is not None and ch is not None
        _check_cond("cs", cs, cond_rows, d)
        _check_cond("ch", ch, cond_rows, d)
        if cs.stride(0) != ch.stride(0):
            raise DitKernelUnavailable("cs and ch must share a row stride")
    samples = _cond_samples(rows // seq_len, cond_rows // seq_len)

    a_out = torch.empty_like(x, dtype=out_dtype or x.dtype) if has_adaln else x
    _gate_residual_adaln_kernel[(rows,)](
        x,
        y,
        og,
        cs if has_adaln else og,
        ch if has_adaln else og,
        x,
        a_out,
        seq_len,
        samples,
        og.stride(0),
        cs.stride(0) if cs is not None else 0,
        eps,
        D=d,  # ty:ignore[invalid-argument-type]
        BLOCK_D=triton.next_power_of_2(d),
        HAS_ADALN=has_adaln,  # ty:ignore[invalid-argument-type]
        num_warps=_row_warps(d),  # ty:ignore[unknown-argument]
    )
    return x, (a_out if has_adaln else None)


@torch._dynamo.disable
def swiglu(ab: torch.Tensor) -> torch.Tensor:
    """``silu(a) * b`` where ``ab = [a | b]`` is ``[M, 2H]`` contiguous.

    One kernel between the two cuBLAS SGEMMs of the transition's FFN, not a
    fusion of them: an IEEE-fp32 ``tl.dot`` has no tensor-core path and
    measured 24-27x slower than cuBLAS SGEMM in this codebase.

    Inference only; fp32 only.

    Parameters
    ----------
    ab : torch.Tensor
        ``[M, 2H]`` fp32 or bf16, contiguous, CUDA. The output matches.

    Returns
    -------
    torch.Tensor
        ``[M, H]`` fp32.

    Raises
    ------
    DitKernelUnavailable
        If Triton is missing or an input is unsupported.
    """
    if ab.dtype is torch.bfloat16:
        _check_stream_bf16("ab", ab)
    else:
        _check_stream("ab", ab)
    rows, two_h = ab.shape
    if two_h % 2:
        raise DitKernelUnavailable(f"ab last dim must be even, got {two_h}")
    hidden = two_h // 2
    out = torch.empty(rows, hidden, device=ab.device, dtype=ab.dtype)
    _swiglu_kernel[(rows,)](
        ab,
        out,
        HIDDEN=hidden,  # ty:ignore[invalid-argument-type]
        BLOCK_H=triton.next_power_of_2(hidden),
        num_warps=_row_warps(hidden),  # ty:ignore[unknown-argument]
    )
    return out
