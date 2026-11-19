"""Row-sharded pair stack for ``set_memory_mode("big", shard_rows=True)``.

Every rank of the process group runs the whole model. Only pair-shaped
``[B, N, N, C]`` tensors are split: rank ``q`` holds rows ``[r0, r1)`` of each,
from the statement that creates it to the last one that reads it. Everything
else (single representation, MSA ``m``, atoms, diffusion state) is replicated.

The data movement comes from ``opt_core.mem.rowpair``: the TriMul ring, the banded transposes, row gathers
and rows of the stock Philox draws. This module binds those primitives to the
ESMFold2 modules; ``layers.py`` and ``model.py`` consult :func:`current_layout`
to pick the row form of a statement.
"""

import contextvars
import os
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

try:
    from opt_core.mem.rowpair import RowpairRefused  # ty: ignore[unresolved-import]
    from opt_core.mem.rowpair import bcast as _rbc  # ty: ignore[unresolved-import]
    from opt_core.mem.rowpair import dist as _rd  # ty: ignore[unresolved-import]
    from opt_core.mem.rowpair import pairstack as _rps  # ty: ignore[unresolved-import]
    from opt_core.mem.rowpair import ring as _ring  # ty: ignore[unresolved-import]
    from opt_core.mem.rowpair import rng as _rng  # ty: ignore[unresolved-import]
    from opt_core.mem.rowpair import trimul as _rt  # ty: ignore[unresolved-import]
    from opt_core.mem.rowpair import (  # ty: ignore[unresolved-import]
        trimul_fused as _rtf,
    )

    ROWPAIR_AVAILABLE = True
except ImportError:
    _rbc: Any = None
    _rd: Any = None
    _rps: Any = None
    _ring: Any = None
    _rng: Any = None
    _rt: Any = None
    _rtf: Any = None

    class RowpairRefused(RuntimeError):
        """Stand-in so call sites can name the error with opt_core absent."""

    ROWPAIR_AVAILABLE = False

if TYPE_CHECKING:
    from opt_core.mem.rowpair.dist import Layout  # ty: ignore[unresolved-import]
else:
    Layout = Any

# t16 loads its cubin through cuda-bindings; without it every call fails over to Triton.
OPT_CORE_HINT = (
    'pip install "https://github.com/evolutionaryscale/wheels/releases/download/'
    'opt-core/v0.5.228.0/opt_core-0.5.228.0-py3-none-any.whl" '
    '"cuda-bindings>=13.3,<14"'
)

# Columns per in-place TriMul chunk.
_TRIMUL_INPLACE_CHUNK = 256
_TRIMUL_GRID = "rank"

# Rows per block of the in-place pair transition on a shard: caps the [rows, N, 4C] hidden.
TRANSITION_ROWS = 256

# Draws up to this size are taken whole and sliced.
_WHOLE_DRAW_MAX_ELEMS = 1 << 27

_LAYOUT: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "esmfold2_rowpair_layout", default=None
)


def require_opt_core() -> None:
    if not ROWPAIR_AVAILABLE:
        raise ImportError(
            "ESMFold2: set_memory_mode('big', shard_rows=True) needs opt_core's "
            f"row-sharded pair stack. Install it with {OPT_CORE_HINT}"
        )


def require_process_group(device: torch.device) -> None:
    """Raise unless a process group is initialised and this rank holds its own GPU."""
    require_opt_core()
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "ESMFold2: shard_rows=True needs a process group. Call "
            "torch.distributed.init_process_group() before folding."
        )
    # NCCL uses the current device. LOCAL_RANK is no index when each rank sees one GPU.
    if device.type == "cuda":
        current = torch.cuda.current_device()
        index = current if device.index is None else device.index
        if index != current:
            raise RuntimeError(
                f"ESMFold2: the model is on {device} but the current CUDA device "
                f"is cuda:{current}. Call torch.cuda.set_device(...) before "
                "moving the model to CUDA, so each rank uses its own GPU."
            )


def world() -> tuple[int, int]:
    """``(P, rank)`` of the running group; ``(1, 0)`` without one."""
    require_opt_core()
    comm = _rd.comm()
    return int(comm.P), int(comm.rank)


def layout_for(n_tokens: int) -> Layout | None:
    """The row layout of an ``n_tokens`` fold, or ``None`` to fold it whole.

    ``None`` on one rank, and below the sharding floor: an input too small to
    give every rank a row block is folded whole on every rank, with no collective.
    """
    P, rank = world()
    if P == 1:
        return None
    try:
        return _rd.Layout.auto(int(n_tokens), P, rank)
    except RowpairRefused:
        warnings.warn(
            f"ESMFold2: {n_tokens} tokens cannot be split over {P} GPUs; "
            "every rank folds the input whole.",
            stacklevel=2,
        )
        return None


def current_layout() -> Layout | None:
    """The layout of the sharded statement being issued, or ``None``."""
    return _LAYOUT.get()


@contextmanager
def sharded(layout: Layout) -> Iterator[None]:
    """Inside, pair-shaped tensors are this rank's rows of ``layout``."""
    token = _LAYOUT.set(layout)
    try:
        yield
    finally:
        _LAYOUT.reset(token)


def rows_of(t: Tensor, layout: Layout, dim: int = 1) -> Tensor:
    """This rank's rows of a whole tensor along ``dim`` (a view)."""
    return t.narrow(dim, layout.r0, layout.R)


# ---------------------------------------------------------------------------
# Rows of the single-device random draws
# ---------------------------------------------------------------------------


def uniform_rows(
    layout: Layout,
    channels: int,
    low: float,
    high: float,
    device: torch.device,
    *,
    transform: Callable[[Tensor], Tensor] = lambda t: t,
    dtype: torch.dtype = torch.float32,
    rows: int = TRANSITION_ROWS,
) -> Tensor:
    """Rows of ``transform(torch.empty(1, N, N, C).uniform_(low, high)).to(dtype)``.

    ``transform`` must be elementwise. Bit-identical to the single-device
    draw: on CUDA the stock launches covering each block of rows are re-issued
    at the Philox offsets the whole draw would see, and the generator is left
    where the whole draw leaves it. Only one fp32 block exists at a time. On
    CPU the whole draw is taken and sliced.
    """
    N, r0, R = layout.N, layout.r0, layout.R
    out = torch.empty(1, R, N, channels, dtype=dtype, device=device)
    blocks = [(lo, min(lo + rows, R)) for lo in range(0, R, rows)]
    if device.type != "cuda":
        full = torch.empty(1, N, N, channels, dtype=torch.float32).uniform_(low, high)
        for lo, hi in blocks:
            out[:, lo:hi] = transform(full[:, r0 + lo : r0 + hi].clone())
        return out
    gen = _rng.cuda_default_generator(device)
    base = _rng.get_offset(gen)
    plan = _rng.launch_plan(N * N * channels, device, 4)
    for lo, hi in blocks:
        f0, f1 = (r0 + lo) * N * channels, (r0 + hi) * N * channels
        band = torch.empty(f1 - f0, dtype=torch.float32, device=device)
        for start, n, rel in plan.launches:
            a, b = max(f0, start), min(f1, start + n)
            if a >= b:
                continue
            _rng.set_offset(gen, base + rel)
            piece = torch.empty(n, dtype=torch.float32, device=device)
            piece.uniform_(low, high, generator=gen)
            band[a - f0 : b - f0].copy_(piece[a - start : b - start])
            del piece
        out[:, lo:hi] = transform(band.view(1, hi - lo, N, channels))
        del band
    _rng.set_offset(gen, base + plan.round_increment)
    return out


def dropout_rows(
    make_rows: Callable[[int, int], Tensor],
    layout: Layout,
    p: float,
    channels: int,
    dtype: torch.dtype,
    device: torch.device,
    rows: int = TRANSITION_ROWS,
) -> Tensor:
    """Rows of ``F.dropout(x, p, training=True).to(dtype)`` for a whole ``x [1, N, N, C]``.

    ``make_rows(g0, g1)`` builds global rows ``[g0, g1)`` of the fp32 ``x``.
    Rows are built, dropped out and cast one block at a time, so no fp32
    shard exists. Bit-identical to the single-device call; the generator
    advances by the same amount.
    """
    N, R = layout.N, layout.R
    out = torch.empty(1, R, N, channels, dtype=dtype, device=device)
    blocks = [(lo, min(lo + rows, R)) for lo in range(0, R, rows)]
    if p == 0.0:
        for lo, hi in blocks:
            out[:, lo:hi] = make_rows(layout.r0 + lo, layout.r0 + hi)
        return out
    if device.type != "cuda" or N * N * channels <= _WHOLE_DRAW_MAX_ELEMS:
        # x * keep * scale with x == 1 is the scale itself, so rows of it are exact.
        keep = F.dropout(
            torch.ones(1, N, N, channels, device=device), p=p, training=True
        )
        for lo, hi in blocks:
            g0, g1 = layout.r0 + lo, layout.r0 + hi
            out[:, lo:hi] = make_rows(g0, g1) * keep[:, g0:g1]
        return out
    gen = _rng.cuda_default_generator(device)
    base = _rng.get_offset(gen)
    final = base
    for lo, hi in blocks:
        g0, g1 = layout.r0 + lo, layout.r0 + hi
        band, _, info = _rng.dropout_rows(
            make_rows(g0, g1).contiguous(),
            N,
            channels,
            g0,
            g1,
            p=p,
            base_offset=base,
            device=device,
            return_mask=False,
            finish=False,
        )
        out[:, lo:hi] = band
        final = info["final_offset"]
    _rng.set_offset(gen, final)
    return out


# ---------------------------------------------------------------------------
# Pair-stack statements on a shard
# ---------------------------------------------------------------------------


def _trimul_fns(block: Any, fused: bool) -> Any:
    """``TriangleMultiplicativeBlock``'s statements as opt_core ``TriMulFns``."""
    C_z = block.input_channels
    C_h = block.latent_channels
    w = block.proj_bundle.weight
    # Cached on the block, so it dies with the model; the key drops it when the weights move.
    key = (fused, w.data_ptr(), w.dtype, w.device)
    cache = block.__dict__.setdefault("_rowpair_trimul_fns", {})
    if key in cache:
        return cache[key]
    ln_in, ln_mix = block.norm_start, block.norm_mix
    # proj_bundle rows: a-signal | b-signal | a-gate | b-gate, C_h each.
    w_ap, w_bp, w_ag, w_bg = (
        w[:C_h],
        w[C_h : 2 * C_h],
        w[2 * C_h : 3 * C_h],
        w[3 * C_h :],
    )
    w_o, w_og = block.proj_emit.weight, block.proj_gate.weight

    def proj(zb: Tensor, mb: Tensor, is_a: bool) -> Tensor:
        n = F.layer_norm(zb, (C_z,), ln_in.weight, ln_in.bias, ln_in.eps)
        wp, wg = (w_ap, w_ag) if is_a else (w_bp, w_bg)
        p = F.linear(n, wp) * torch.sigmoid(F.linear(n, wg))
        # The ring sizes its buffers from z's dtype, so the operand must match it.
        return (p * mb.to(p.dtype)).to(zb.dtype)

    def out(x: Tensor) -> Tensor:
        return F.linear(
            F.layer_norm(x, (C_h,), ln_mix.weight, ln_mix.bias, ln_mix.eps), w_o
        ).to(x.dtype)

    def gate(zb: Tensor) -> Tensor:
        n = F.layer_norm(zb, (C_z,), ln_in.weight, ln_in.bias, ln_in.eps)
        return torch.sigmoid(F.linear(n, w_og)).to(zb.dtype)

    fns = _rt.TriMulFns(proj, out, gate, C_h)
    if fused and w.is_cuda:
        weights = {
            "ln_in_w": ln_in.weight,
            "ln_in_b": ln_in.bias,
            "w_ap": w_ap,
            "w_bp": w_bp,
            "w_ag": w_ag,
            "w_bg": w_bg,
            "ln_out_w": ln_mix.weight,
            "ln_out_b": ln_mix.bias,
            "w_o": w_o,
            "w_og": w_og,
        }
        # On sm_90+ the fused row blocks beat the statements at every size.
        min_tokens = 0 if torch.cuda.get_device_capability(w.device)[0] >= 9 else None
        fns = _rtf.fused_trimul_fns(
            weights, fns, eps=float(ln_in.eps), min_tokens=min_tokens
        )
    cache.clear()
    cache[key] = fns
    return fns


def trimul_residual_rows_(
    block: Any, pair: Tensor, mask: Tensor | None, layout: Layout, fused: bool
) -> Tensor:
    """``pair += block(pair, mask)`` on this rank's rows ``[B, R, N, C]``, in place.

    The column operand travels round a ring of row blocks (outgoing) or comes
    from banded transposes (incoming); no ``N x N`` tensor exists on a rank.
    ``fused`` serves the row blocks with opt_core's fused TriMul kernels.
    """
    if not pair.is_contiguous():
        raise RowpairRefused("trimul_residual_rows_: the shard must be contiguous")
    fns = _trimul_fns(block, fused)
    for b in range(pair.shape[0]):
        m = None if mask is None else mask[b].to(torch.float32).contiguous()
        _rt.trimul_update_(
            fns,
            pair[b],
            m,
            layout,
            outgoing=block.flow == "outgoing",
            add=True,
            grid=_TRIMUL_GRID,
            inplace_chunk=_TRIMUL_INPLACE_CHUNK,
        )
    return pair


def row_blocks_(
    fn: Callable[[Tensor], Tensor],
    pair: Tensor,
    *,
    add: bool,
    rows: int = TRANSITION_ROWS,
) -> Tensor:
    """A row-local ``fn`` on ``[B, R, N, C]`` rows, per block of ``rows``, in place.

    ``add=True``: ``fn`` returns a delta that is added. ``add=False``: ``fn``
    returns the updated rows, written back over the block.
    """
    if not pair.is_contiguous():
        raise RowpairRefused("row_blocks_: the shard must be contiguous")
    for b in range(pair.shape[0]):
        _rps.transition_update_(
            lambda zb, _mb: fn(zb.unsqueeze(0))[0], pair[b], None, rows, add=add
        )
    return pair


def transpose_band_rows(
    x_rows: Tensor, layout: Layout, lo: int, hi: int
) -> Tensor | None:
    """Local rows ``[lo, hi)`` of ``x^T`` (token axes swapped), from rows of ``x``.

    Collective: every rank calls with the same ``lo, hi``. Returns ``None``
    once ``lo`` is past this rank's rows.
    """
    windows = [(q0 + lo, min(q0 + hi, q1)) for q0, q1 in layout.bounds]
    return _ring.transpose_band(x_rows.contiguous(), layout, windows)


# ---------------------------------------------------------------------------
# Gathers and replicated-state guards
# ---------------------------------------------------------------------------


def all_gather_rows(x_rows: Tensor, layout: Layout, dim: int = 1) -> Tensor:
    """The whole tensor on every rank from each rank's rows along ``dim``."""
    if dim == 1 and x_rows.shape[0] == 1:
        # Batch 1 gathers dim 0 of x[0] directly, so the result stays contiguous.
        return _rd.all_gather_rows(x_rows[0].contiguous(), layout).unsqueeze(0)
    moved = x_rows.movedim(dim, 0).contiguous()
    return _rd.all_gather_rows(moved, layout).movedim(0, dim)


def gather_rows_to_rank0_host(x_rows: Tensor, layout: Layout) -> Tensor | None:
    """A whole ``[1, N, N, *rest]`` tensor in rank 0's host memory; ``None`` elsewhere.

    Assembled one column block at a time, so no device holds it whole.
    """
    # Pin only device tensors: a CPU rank forked on a GPU machine cannot init CUDA.
    out = _rd.gather_rows_to_rank0_host(x_rows[0], layout, pin=x_rows.is_cuda)
    return None if out is None else out.unsqueeze(0)


SAMPLER_ROUTES = ("auto", "whole", "rows")
# Headroom over the whole route's estimate before it is chosen over rows.
_WHOLE_ROUTE_HEADROOM = 1.25


def sampler_route(need_bytes: int, device: torch.device) -> str:
    """``"whole"`` or ``"rows"``, the same on every rank.

    Whole gathers the conditioned pair once and runs the single-device DiT:
    faster, but ``need_bytes`` on every rank. Rows keep the pair bias on rows
    and scale with the rank count. ``ESMFOLD2_ROWPAIR_SAMPLER`` forces one.
    """
    word = os.environ.get("ESMFOLD2_ROWPAIR_SAMPLER", "auto")
    if word not in SAMPLER_ROUTES:
        raise ValueError(f"ESMFOLD2_ROWPAIR_SAMPLER={word!r}: one of {SAMPLER_ROUTES}")
    if word != "auto":
        return word
    free = _rd.agreed_free_bytes(device) if device.type == "cuda" else None
    if free is None:
        return "whole"
    return "whole" if _WHOLE_ROUTE_HEADROOM * need_bytes <= free else "rows"


def broadcast_features(features: dict[str, Any]) -> dict[str, Any]:
    """Rank 0's feature dict on every rank (unseeded featurisation may differ)."""
    return _rbc.broadcast_tensordict(features, src=0)


def sync_rng_state(device: torch.device) -> None:
    """Give every rank rank 0's CPU and ``device`` generator states."""
    cpu = _rd.broadcast_obj(torch.get_rng_state(), src=0)
    torch.set_rng_state(cpu)
    if device.type == "cuda":
        state = _rd.broadcast_obj(torch.cuda.get_rng_state(device), src=0)
        torch.cuda.set_rng_state(state, device)


def sync_replicated(t: Tensor) -> Tensor:
    """Rank 0's copy, so all ranks condition on one state."""
    return broadcast_(t.contiguous())


def broadcast_(t: Tensor, src: int = 0) -> Tensor:
    """Overwrite ``t`` on every rank with rank ``src``'s copy, in place."""
    _rd.comm().bcast_(t, src)
    return t


def guard_replicated(t: Tensor, name: str) -> None:
    """Raise unless ``t`` is bit-identical on every rank."""
    _rd.allreduce_checksum(t, name)
