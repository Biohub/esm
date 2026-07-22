# Single-GPU ESMFold2 — kernel backends

The reference implementation in pure PyTorch is accurate but not the fastest way to run ESMFold2.
Most of the single-GPU runtime sits in a few repeated pair-representation operations, and there are alternate **kernel backends** that keep the same weights and outputs while swapping in faster implementations of those hot paths.
The result is the same ESMFold2 model, but with different speed, warm-up, and dependency trade-offs.

ESMFold2 runs the expensive Pairformer/attention math through a selectable
**kernel backend**, chosen with:

```python
model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
model.set_kernel_backend(None | "fused" | "cuequivariance")
```

This propagates out to every module that runs Pairformer-style blocks (`folding_trunk`,
`lm_encoder`, `parcae_coda`, `confidence_head`, `structure_head`). It only swaps the
*implementation* of a few hot ops. Weights are untouched and the three
backends are numerically equivalent to bf16 rounding (same pLDDT/pTM).

> This page covers the single-GPU backends. For multi-GPU context parallelism see
> `fold-cp.md` (`wrap_model_with_cp`), which is an orthogonal choice.

## Table of contents

- [The three backends](#the-three-backends)
- [None (default experience)](#none-default-experience)
- [cuEquivariance](#cuequivariance)
- [Fused](#fused)
- [Performance summary](#performance-summary)
- [Decision guide](#decision-guide)

## The three backends

| | None | fused | cuequivariance |
|---|---|---|---|
| Implementation | PyTorch | Triton kernels | [cuEquivariance](https://github.com/nvidia/cuequivariance) |
| Accelerated Operations | none (reference) | tri-mul + LN+SwiGLU + dropout-residual + pair-bias | tri-mul |
| Extra dependency | none | `triton>=3` | `cuequivariance-torch` (CUDA-matched build) |
| Training / autograd | yes | inference-only | yes |
| First-call compilation | none | yes — Triton JIT + autotune | no — precompiled kernels |

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

## cuEquivariance

[cuEquivariance](https://github.com/nvidia/cuequivariance) is NVIDIA’s precompiled CUDA kernel backend for ESMFold2’s triangle-multiplication hot path, giving a large single-GPU speedup without Triton JIT or accuracy changes.

### Dependencies

* `cuequivariance-torch`
* Depending on your CUDA version
  * `cuequivariance-ops-torch-cu13`
  * `cuequivariance-ops-torch-cu12`

Install the CUDA-matched extra:

```bash
pip install "esm[cueq12]"  # CUDA 12 build
```
or
```bash
pip install "esm[cueq13]"  # CUDA 13 build
```

You can figure out which one you need with `nvidia-smi`

```shell
$ nvidia-smi

Thu Jul 16 13:03:23 2026
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.216.03             Driver Version: 535.216.03   CUDA Version: 13.2     |
|-----------------------------------------+----------------------+----------------------+
```

In this case, the CUDA Version is 13.2, so you would need to install `esm[cueq13]`.

### Usage

```python
model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()
model.set_kernel_backend("cuequivariance")
```

### Performance

Fold a 644-residue protein using the cuEquivariance backend with [esmfold2-cueq.py](esmfold2-cueq.py) on a single H100 SXM

```shell
python esmfold2-cueq.py

pLDDT mean: 0.751, pTM: 0.459, ipTM: 0.096
Elapsed: 19.90 sec
Max VRAM: 19680.1 MB
```

## Fused

The "fused" backend uses Triton, a Python-based language for writing custom GPU kernels, to fuse several ESMFold2 hot-path operations into fewer CUDA launches for the fastest single-GPU inference while preserving the same model weights and outputs.

### Dependencies

The "fused" backend needs Triton 3. It's GPU-only (Triton JIT-compiles to PTX) and **inference-only** (falls back to reference under autograd).

```bash
pip install "esm[fused]"  # triton>=3,<4
```

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
| "cuequivariance" | 19.90 s | 19680.1 MB | 0.751 | 0.459 | 0.096 | 2.9× |
| "fused" | 16.56 s | 19795.1 MB | 0.750 | 0.458 | 0.096 | 3.5× |

## Decision guide

If you want:

- **Fastest throughput →** `"fused"`.
  - Fastest steady state, and the incremental per-new-length compile is only ~0.5 s
- **Huge length variety →** `"cuequivariance"`
  — precompiled, steady state a bit slower than fused.
- **Bit-exact reference for debugging →** `None`.

> Tip: For any benchmarking, include a warm-up fold at startup to pay the ~10s global cost
