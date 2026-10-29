# Single-GPU ESMFold2 — kernel backends

The reference implementation in pure PyTorch is accurate but not the fastest way to run ESMFold2.
Most of the single-GPU runtime sits in a few repeated pair-representation operations, and there are alternate **kernel backends** that keep the same weights and outputs while swapping in faster implementations of those hot paths.
The result is the same ESMFold2 model, but with different speed, warm-up, and dependency trade-offs.

ESMFold2 runs the expensive Pairformer/attention math through a selectable
**kernel backend**, chosen with:

```python
model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
model.set_kernel_backend(None | "fused")
```

This propagates out to every module that runs Pairformer-style blocks (`folding_trunk`,
`lm_encoder`, `parcae_coda`, `confidence_head`, `structure_head`). It only swaps the
*implementation* of a few hot ops. Weights are untouched and both
backends are numerically equivalent to bf16 rounding (same pLDDT/pTM).

> This page covers the single-GPU backends. For multi-GPU context parallelism see
> `fold-cp.md` (`wrap_model_with_cp`), which is an orthogonal choice.

## Table of contents

- [The two backends](#the-two-backends)
- [None (default experience)](#none-default-experience)
- [Fused](#fused)
- [Performance summary](#performance-summary)
- [Decision guide](#decision-guide)

## The two backends

| | None | fused |
|---|---|---|
| Implementation | PyTorch | Triton kernels |
| Accelerated Operations | none (reference) | tri-mul + LN+SwiGLU + dropout-residual + pair-bias |
| Extra dependency | none | `triton>=3` |
| Training / autograd | yes | inference-only |
| First-call compilation | none | yes — Triton JIT + autotune |

## None (default experience)

### Dependencies

None beyond this ESM package

### Usage

```python
model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
model.set_kernel_backend(None) # optional
```

### Performance

Fold a 644-residue protein using the default backend with [esmfold2-none.py](esmfold2-none.py) on a single H100 SXM

```shell
python esmfold2-none.py

/usr/local/lib/python3.12/dist-packages/torch/jit/_script.py:1487: DeprecationWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
  warnings.warn(
🚨 No checkpoint found for ESMCForSequenceClassification.forward. Please add a `checkpoint` arg to `auto_docstring` or add one in ESMCConfig's docstring
🚨 No checkpoint found for ESMCForTokenClassification.forward. Please add a `checkpoint` arg to `auto_docstring` or add one in ESMCConfig's docstring
Loading checkpoint shards: 100%|█████████████████████████████████| 6/6 [00:00<00:00, 151.51it/s]
Loading CCD dictionary from /root/.cache/huggingface/hub/models--biohub--ESMFold2/snapshots/1ebf0e3481a5184eb6171d40615c79e384b48796/ccd.pkl
pLDDT mean: 0.751, pTM: 0.458, ipTM: 0.096
Elapsed: 58.02 sec
Max VRAM: 19682.0 MB
```

## Fused

The "fused" backend uses Triton, a Python-based language for writing custom GPU kernels, to fuse several ESMFold2 hot-path operations into fewer CUDA launches for the fastest single-GPU inference while preserving the same model weights and outputs.

### Dependencies

The "fused" backend needs Triton 3, which `pip install esm` installs on Linux x86-64. It's GPU-only (Triton JIT-compiles to PTX) and **inference-only** (falls back to reference under autograd).

> If Triton is not importable, `set_kernel_backend("fused")` silently runs the reference path. Check that Triton can be imported if `"fused"` is unexpectedly slow.

### Usage

```python
model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
model.set_kernel_backend("fused")
```

### Performance

Fold a 644-residue protein using the "fused" backend with [esmfold2-fused.py](esmfold2-fused.py) on a single H100 SXM

```shell
python esmfold2-fused.py

pLDDT mean: 0.750, pTM: 0.458, ipTM: 0.096
Elapsed: 16.56 sec
Max VRAM: 19795.1 MB
```

## Performance summary

These results fold the same 644-residue `7ysz` input used in the per-backend
examples above.

| backend | elapsed | max VRAM | pLDDT | pTM | ipTM | speedup vs `None` |
|---|---:|---:|---:|---:|---:|---:|
| `None` | 58.02 s | 19682.0 MB | 0.751 | 0.458 | 0.096 | 1.0× |
| "fused" | 16.56 s | 19795.1 MB | 0.750 | 0.458 | 0.096 | 3.5× |

## Decision guide

If you want:

- **Fastest throughput →** `"fused"`.
  - Fastest steady state, and the incremental per-new-length compile is only ~0.5 s
- **Bit-exact reference for debugging →** `None`.

> Tip: For any benchmarking, include a warm-up fold at startup to pay the ~10s global cost
