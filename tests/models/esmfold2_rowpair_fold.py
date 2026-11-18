"""One fold of the real ESMFold2-Fast checkpoint under "big"; run by torchrun.

Usage: ``torchrun --nproc-per-node=P esmfold2_rowpair_fold.py OUT``. Rank 0 saves
each sample's coordinates. The manual test in ``esmfold2_rowpair_test.py`` runs it.
"""

import os
import sys

import torch
import torch.distributed as dist

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    EsmFold2Model,
    MolecularComplexResult,
    ProteinInput,
    StructurePredictionInput,
)
from tests.conftest import _protein_of_length

GPU_MODEL = "biohub/ESMFold2-Fast"
GPU_CHAINS = (200, 200, 200)


def main(out: str) -> None:
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    sharded = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if sharded:
        dist.init_process_group("nccl")
    model = EsmFold2Model.from_pretrained(GPU_MODEL).cuda().eval()
    model.set_kernel_backend("fast")
    model.set_memory_mode("big", shard_rows=sharded)
    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id=chr(65 + i), sequence=_protein_of_length(n))
            for i, n in enumerate(GPU_CHAINS)
        ]
    )
    results = ESMFold2InputBuilder().fold(
        model, spi, num_loops=3, num_sampling_steps=20, num_diffusion_samples=5, seed=0
    )
    assert not isinstance(results, MolecularComplexResult)  # one per sample
    if int(os.environ.get("RANK", "0")) == 0:
        torch.save([torch.as_tensor(r.complex.atom_positions) for r in results], out)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1])
