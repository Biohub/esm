# Single-GPU ESMFold2 — kernel backends

ESMFold2 runs the expensive Pairformer/attention math through a selectable
**kernel backend**, chosen with:

```python
model.set_kernel_backend(None | "fused" | "cuequivariance")
```

This fans out to every module that runs Pairformer-style blocks (`folding_trunk`,
`lm_encoder`, `parcae_coda`, `confidence_head`, `structure_head`). It only swaps the
*implementation* of a few hot ops — the **triangle multiplicative update** (the
dominant `L×L` Pairformer op), the transition FFN (LayerNorm+SwiGLU), the
dropout-residual, and the attention pair-bias. Weights are untouched and the three
backends are numerically equivalent to bf16 rounding (same pLDDT/pTM). They trade
**speed, memory, first-call compilation, and dependencies** — not accuracy.

> This page covers the single-GPU backends. For multi-GPU context parallelism see
> `fold-cp.md` (`wrap_model_with_cp`), which is an orthogonal choice.

## The three backends

| | `None` | `"fused"` | `"cuequivariance"` |
|---|---|---|---|
| Implementation | pure PyTorch | vendored **Triton** kernels | **cuEquivariance** kernel |
| Ops accelerated | none (reference) | tri-mul **+ LN+SwiGLU + dropout-residual + pair-bias** | **tri-mul only** (rest = reference) |
| Extra dependency | none | `triton>=3` | `cuequivariance-torch` (CUDA-matched build) |
| First-call compilation | none | **yes — Triton JIT + autotune, per shape** | **no — ships precompiled kernels** |
| Steady-state speed | slowest (reference) | **fastest** | fast (tri-mul only) |
| Missing-dependency behavior | n/a | **silent no-op** → reference path | **raises** at `set_kernel_backend` |
| Runtime fallback | n/a | per-op reference fallback | logs + falls back to chunked einsum if the kernel throws |

## What to install

Optional extras are declared on the `esm` package:

```bash
pip install "esm[fast]"      # -> triton>=3,<4         (the "fused" backend)
pip install "esm[cueq12]"    # -> cuequivariance build for CUDA 12   (stub — fill in)
pip install "esm[cueq13]"    # -> cuequivariance build for CUDA 13   (stub — fill in)
```

- **`None`** needs nothing — it's always available and is the reference/fallback.
- **`"fused"`** needs **Triton 3** (`esm[fast]`). It's GPU-only (Triton JIT-compiles
  to PTX) and inference-only (falls back to reference under autograd). If Triton is
  not importable, `set_kernel_backend("fused")` installs nothing and silently runs
  the reference path — so if `"fused"` is unexpectedly slow, check that Triton
  imported.
- **`"cuequivariance"`** needs `cuequivariance-torch` matched to your CUDA toolkit
  (`esm[cueq12]` / `esm[cueq13]` — currently stubs, fill in the right build). If it's
  not installed, `set_kernel_backend("cuequivariance")` **raises** (unlike `"fused"`,
  which degrades silently).

## Performance (measured, L=1168, single H100, **steady-state after warm-up**)

| backend | elapsed | pLDDT | note |
|---|---|---|---|
| `None` | 216.8 s | 0.793 | reference; ~7× slower |
| `"fused"` | 28.9 s | 0.793 | fastest steady-state |
| `"cuequivariance"` | 36.3 s | 0.793 | ~6× over reference; tri-mul only |

Identical pLDDT confirms the backends are numerically equivalent. `"fused"` edges
out `"cuequivariance"` because it accelerates the *whole* block (FFN + dropout +
tri-mul), whereas `"cuequivariance"` only replaces the tri-mul and leaves the rest
on the reference path.

## Memory

Peak VRAM is **roughly the same across all three backends** at a given length — the
model dtype (bf16) dominates, and the kernel choice is a second-order effect:

| backend | peak VRAM (L=1168) |
|---|---|
| `None` | ~48.3 GB |
| `"cuequivariance"` | ~48.3 GB |
| `"fused"` | ~48.4 GB (marginally higher — Triton autotune scratch) |

So **pick the backend for speed and compilation behavior, not memory.** (To reduce
memory at a given length you want context parallelism — `fold-cp.md` — not a
different kernel backend.)

## The compilation trade-off

There are **two** distinct one-time costs on the first fold(s) — don't conflate them
(measured with `esmc_jit_bench.py`):

1. **A process-global first-fold cost (~10 s), paid once by *any* backend.** The very
   first fold in a process pays cuDNN/cuBLAS algorithm selection, flash-attn init,
   allocator growth, and the first ESM-C / atom-encoder / diffusion run. This is
   **not** a backend property — whichever backend you run first absorbs it. Both
   `"fused"` and `"cuequivariance"` pay it once; `None` too.

2. **A backend-specific kernel init on first use (~2 s), paid once by *both*
   backends.** The first fold with a given backend either JIT-compiles the Triton
   kernel set (`"fused"`) or loads the precompiled cuEquivariance kernels
   (`"cuequivariance"`). Measured ~2.2 s for each.

3. **A per-new-sequence-length cost — this is the only place the backends differ:**
   - **`"fused"` (Triton)** recompiles its shape-specialized kernels for each new
     length → **~0.5 s** per new length.
   - **`"cuequivariance"`** is precompiled / shape-independent → **~0 s** (~0.2 s).
   - **`None`** compiles nothing.

Measured with `esmc_jit_bench.py` (cost = first-fold minus warm, after the global
warm-up), two lengths:

| backend | L=772 (first use) | L=1024 (new length) |
|---|---|---|
| `"cuequivariance"` | ~2.2 s (one-time init) | **~0.2 s** |
| `"fused"` | ~2.2 s (one-time init) | **~0.5 s** |

So the earlier worry that fused pays "seconds-to-minutes per shape" was wrong: the
big cost is the shared ~10 s global first fold, both backends then pay a ~2 s
one-time init, and fused's *extra* per-new-length compile is only ~0.5 s (cueq ~0).
The steady-state timings in the first table exclude all of this (measured after
warm-up).

- **`apply_torch_compile()` does not stack with the Triton kernels** — call
  `set_kernel_backend(None)` before compiling. (torch.compile adds its *own* compile
  cost with the same warm-up caveat.)

## Decision guide

- **Default / fastest throughput →** `"fused"` (`esm[fast]`). Fastest steady state,
  and the incremental per-new-length compile is only ~0.5 s — cheap even for one-shot
  or variable-length workloads once the global first fold + ~2 s init are paid.
- **No Triton available, or you want zero per-shape compile (e.g. huge length
  variety) →** `"cuequivariance"` (`esm[cueq12]`/`esm[cueq13]`) — precompiled, ~0 s
  compile, steady state a bit slower than fused.
- **No extra deps, portability, or a bit-exact reference for debugging →** `None`.

> Tip: whatever backend you pick, do **one throwaway warm-up fold** at startup to pay
> the ~10 s global cost off the critical path (that's what the benchmark's warm-up
> fold does).

## Gotchas recap

- `"fused"` missing Triton → silent reference fallback (looks like `None` speed).
- `"cuequivariance"` missing the package → raises immediately.
- `"cuequivariance"` has a *runtime* safety net: if the kernel throws (odd
  shape/dtype), it logs a warning and uses the chunked einsum for that call.
- `"fused"` is inference-only and GPU-only; under autograd or on CPU it falls back.
- Backends are **not additive** on the tri-mul — `set_kernel_backend` selects one path.
