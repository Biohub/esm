# The BioNeMo Contessa — ESMFold2 folding examples

Three scripts that fold protein/ligand complexes with `ESMFold2Model`, showing
when a single GPU suffices and when **context parallelism (CP)** is needed to fit
a large complex. All use the `cuequivariance` backend, `bf16` ESM-C,
`torch.inference_mode()`, and the same fold settings (`num_loops=10`,
`num_sampling_steps=50`, `num_diffusion_samples=1`, `seed=0`).

Tested on 1× and 4× H100 80GB GPUs.

| Script | Input | GPUs | Result |
| --- | --- | --- | --- |
| `cuequivariance.py` | 7ysz (1 protein chain ×2, GDP + TRS ligands) | 1 | Folds successfully |
| `cuequivariance_cp.py` | 5xgo (1 protein chain ×12, CL ligand) | n² (e.g. 4) | Folds successfully via CP |
| `will_fail.py` | 5xgo (same as CP) | 1 | **OOMs** — too large for one GPU |

## Environment setup

Start from the NVIDIA PyTorch container, then install the dependencies (the
active commands in [`install.sh`](../../../install.sh)):

```bash
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:26.03-py3

pip install git+https://github.com/Biohub/esm.git@main
pip install cuequivariance-torch cuequivariance-ops-torch-cu13
pip uninstall -y transformers
pip install git+https://github.com/zyndagj/transformers.git@zyndagj/foldcp_support
```

The `zyndagj/foldcp_support` transformers branch ships `ESMFold2Model` and its CP
utilities. (Commented-out lines in `install.sh` show an alternative `uv`
virtualenv setup.)

## `cuequivariance.py` — single-GPU baseline

Folds a small complex (7ysz) on one GPU and writes `esmfold2_output.cif`.

```bash
python cuequivariance.py
```

## `cuequivariance_cp.py` — context-parallel for large complexes

Folds a large 12-chain complex (5xgo) by sharding across an **n×n CP grid** of
GPUs. Both the trunk *and* the MSA encoder are sharded via `wrap_model_with_cp`
(trunk-only sharding still OOMs on the full L×L pair). Launch with `torchrun` on
a **perfect-square** number of GPUs (1, 4, 9, …); output goes to
`esmfold2_cp_output.cif`.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc-per-node=4 cuequivariance_cp.py
```

> **CP is a capability enabler, not a speedup.** Splitting the L×L pair across
> GPUs lets you fold complexes too large for one GPU, but the per-layer cross-rank
> communication means it won't fold faster than a single GPU that already fits the
> complex. Reach for CP on out-of-memory, not for performance.

## `will_fail.py` — what happens without CP

The single-GPU script run on the large 5xgo complex (same input as the CP
script). The 12-chain L×L pair representation exceeds one GPU's memory, so it
OOMs — the fix is the CP sharding in `cuequivariance_cp.py`.

```bash
python will_fail.py   # expected to OOM
```

## Outputs

Successful runs write an mmCIF file (`esmfold2_output.cif` /
`esmfold2_cp_output.cif`) and print `pLDDT mean`, `pTM`, `ipTM`, elapsed time,
and peak VRAM.
