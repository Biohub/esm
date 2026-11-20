# Fold large inputs with "big" mode, on one GPU or several

`set_memory_mode("big")` trades speed for lower peak GPU memory, so larger
complexes fit. Add `shard_rows=True` and run under `torchrun`, and "big" also
splits the pair representation across the GPUs. Without it, each rank folds its
own input whole, as in any data-parallel script.

## Install

"big" on one GPU needs only `esm`. On several GPUs it also needs
`opt_core`, which provides GPU-to-GPU communication. The 3,350-token example
also uses `opt_core` for its H100 triangle multiplication kernels.

```bash
pip install esm
pip install "https://github.com/evolutionaryscale/wheels/releases/download/opt-core/v0.5.228.0/opt_core-0.5.228.0-py3-none-any.whl" "cuda-bindings>=13.3,<14"
```

## Run

```python
import os

import torch
import torch.distributed as dist

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    EsmFold2Model,
    ProteinInput,
    StructurePredictionInput,
)

# Under torchrun every rank must use its own GPU, in one process group.
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
sharded = int(os.environ.get("WORLD_SIZE", "1")) > 1
if sharded:
    dist.init_process_group("nccl")

model = EsmFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
model.set_kernel_backend("fast")
model.set_memory_mode("big", shard_rows=sharded)
model.set_chunk_size(None)

spi = StructurePredictionInput(
    sequences=[ProteinInput(id="A", sequence=("MQIFVKTLTGKT" * 280)[:3350])]
)
result = ESMFold2InputBuilder().fold(
    model, spi, num_loops=10, num_sampling_steps=100, num_diffusion_samples=1, seed=0
)

if int(os.environ.get("RANK", "0")) == 0:
    print(result.ptm)
```

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ESMFOLD2_BF16_DIT_GEMM=1
python fold_big.py                          # one GPU
torchrun --nproc-per-node=2 fold_big.py     # two GPUs; any number works
```

Sharded, every rank folds rank 0's input and returns the same coordinates and scores. `pae`, `pde` and the
distogram exist on rank 0 only (`None` elsewhere), so read results and write
files from rank 0.

## How it splits the work

Each GPU holds a block of rows of every `N x N` pair tensor, from the moment it
is created to the confidence head. Everything else, including the language
model, the MSA, atoms and the diffusion sampler, runs whole on every GPU.

| Step | On several GPUs |
| --- | --- |
| Triangle multiplication | The column operand is passed GPU to GPU in a ring |
| Pair transition, recycling | Each GPU updates its own rows |
| MSA module | Pair rows local; the MSA update is gathered |
| Diffusion | The pair conditioning is gathered once if it fits on every GPU; otherwise each GPU keeps its rows and the sampler attends from its own query rows |
| Confidence, distogram | Rows local; the small results are gathered |

The diffusion route is picked from the free memory every GPU agrees on. Set
`ESMFOLD2_ROWPAIR_SAMPLER=whole` or `rows` to force one (default `auto`).

Random draws are the single-GPU draws, so a seeded fold sees the same pair
initialisation and dropout on any number of GPUs.

## What to expect

- **Size, not speed.** Two GPUs are slower than one on inputs that fit on one.
- **Close to one GPU, not identical.** Splitting changes the order of sums. On
  600-token complexes, two GPUs differ from one by 1-5x the run-to-run noise
  of one GPU, and about 20x less than two seeds differ.
- **One complex per call.** Batched inputs are refused.
- **Small inputs fold whole.** An input too small to split is folded whole on
  every GPU, with a warning.
- **Not returned on several GPUs:** `pae_logits` and `pde_logits` (the
  `[samples, N, N, bins]` logits; `pae` and `pde` are returned), and
  `include_embeddings=True`.

Largest input that completed, 5 diffusion samples, one trunk pass:

| GPUs | ESMFold2-Fast | ESMFold2 |
| --- | ---: | ---: |
| 2x L40 48 GB | 4,000 tokens | |
| 2x H100 80 GB | 6,000 tokens | 5,000 tokens |
| 4x H100 80 GB | 12,000 tokens | |

The H100 sizes also complete at full settings (20 loops, 200-step schedule,
5 samples): ESMFold2-Fast at 6,000 tokens in 7.8 min with a 70 GiB peak per
GPU, ESMFold2 at 5,000 tokens in 12.3 min with a 63 GiB peak.
