# The BioNeMo Contessa — ESMFold2 folding examples

This cookbook shows how to run Biohub's ESMFold2 structure-prediction model with
[NVIDIA BioNeMo libraries and methods](https://github.com/NVIDIA-BioNeMo). It
covers single-GPU accelerated kernel backends for faster inference and context
parallelism (CP) for inputs whose `L x L` pair representation does not fit on one
GPU.

Tested on 1x and 4x H100 80GB GPUs.

| Script | Input | GPUs | Result |
| --- | --- | --- | --- |
| `esmfold2-none.py` | 7ysz (1 protein chain ×2) | 1 | Reference single-GPU path |
| `esmfold2-cueq.py` | 7ysz (1 protein chain ×2) | 1 | cuEquivariance backend |
| `esmfold2-fused.py` | 7ysz (1 protein chain ×2) | 1 | Triton fused backend |
| `esmfold2-cp.py` | 7ysz (1 protein chain ×2) | 4 | Same fold API with CP setup |
| `cuequivariance_cp.py` | 5xgo (1 protein chain ×12, CL ligand) | 4 | Larger input via CP |
| `will_fail.py` | 5xgo (same as CP) | 1 | **OOMs** — too large for 1xH100 |

## Environment setup

Start from the NVIDIA PyTorch container, then install the dependencies:

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:26.03-py3 bash -l

# Main ESM package
pip install "git+https://github.com/Biohub/esm.git@main"
# fused dependency
pip install "esm[fused] @ git+https://github.com/Biohub/esm.git@main"
# cuEquivariance dependency (cueq12 also exists)
pip install "esm[cueq13] @ git+https://github.com/Biohub/esm.git@main"
# fold-cp dependency for larger inputs
pip install "esm[fold-cp] @ git+https://github.com/Biohub/esm.git@main"
```

## Single-GPU accelerated backends

[single-gpu.md](single-gpu.md) compares the three single-GPU backend choices:
`None` for the pure PyTorch reference path, `"cuequivariance"` for NVIDIA
cuEquivariance triangle-multiplication kernels, and `"fused"` for Triton kernels
that fuse several ESMFold2 hot-path operations. These backends keep the same model
weights and outputs while trading dependencies, warm-up behavior, and speed.

```bash
python esmfold2-none.py
python esmfold2-cueq.py
python esmfold2-fused.py
```

## Fold-CP for larger inputs

[fold-cp.md](fold-cp.md) documents how `wrap_model_with_cp(model, dm, ...)`
spreads one ESMFold2 fold across a square grid of GPUs using the [Fold-CP methodology](https://github.com/NVIDIA-BioNeMo/boltz-cp).
CP shards the large `L x L` pair activations so longer proteins and complexes fit in memory; it is a
capability feature rather than a general speedup.

Launch CP examples with `torchrun` on a perfect-square number of GPUs.

```bash
torchrun --nproc-per-node=4 esmfold2-cp.py

# Larger 5xgo example.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc-per-node=4 cuequivariance_cp.py
```
