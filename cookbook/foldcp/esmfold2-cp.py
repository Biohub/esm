import math
import os
from collections import OrderedDict
from time import time

import torch

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    EsmFold2Model,
    MolecularComplexResult,
    ProteinInput,
    StructurePredictionInput,
)
from esm.models.esmfold2.distributed import DistributedManager, wrap_model_with_cp

spi = StructurePredictionInput(  # 7ysz
    sequences=[
        ProteinInput(
            id=["A1", "B1"],
            sequence=(
                "MDNFDNYEQVASIKVIGIGGAGNNAVNRMIEAGVQGVEFIVANTDAQIISVSKSKNKIVLGKETSKGLGA"
                "GANPDVGRQAAIESAEEIKDALKGADMVFVAAGMGGGTGTGAAPIIAKLAREQGALTVGIITTPFSFEGR"
                "ARNSYAIQGTEELRKHVDSLIIISNDRLLEVIGGVPLKDSFKEADNILRQGVQTITDLIAVPSLINLDFA"
                "DIKTVMKNKGNALFGIGIGSGKDKAIEAANKAIISPLLEASIRGARDAIINVTGGNTLTLNDANDAVDIV"
                "KQAIGGEVNIIFGTAVNEHLDDEMIVTVIATGFDGSHHHHHH"
            ),
        )
    ]
)

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
n = math.isqrt(world_size)
assert n * n == world_size, "CP requires a square number of GPUs: 1, 4, 9, 16, ..."

torch.cuda.set_device(local_rank)
torch.cuda.reset_peak_memory_stats()

DistributedManager.initialize(
    grid_group_sizes=OrderedDict([("dp", 1), ("cp", (n, n))]),
    device_type="cuda",
    backend="nccl",
)
dm = DistributedManager()

model = (
    EsmFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision="bf16")
    .cuda()
    .eval()
)
wrap_model_with_cp(model, dm, comm="ring")

start = time()
with torch.inference_mode():
    result = ESMFold2InputBuilder().fold(
        model, spi, num_loops=10, num_sampling_steps=50, num_diffusion_samples=1, seed=0
    )
end = time()
peak_mib = torch.cuda.max_memory_allocated() / (1024**2)

# num_diffusion_samples=1 returns a single result with confidence heads populated.
assert isinstance(result, MolecularComplexResult)
assert result.plddt is not None and result.ptm is not None and result.iptm is not None

if local_rank == 0:
    print(
        f"pLDDT mean: {float(result.plddt.mean()):.3f}, "
        f"pTM: {float(result.ptm):.3f}, ipTM: {float(result.iptm):.3f}"
    )
    print(f"Elapsed: {end - start:.2f} sec")
    print(f"Max VRAM: {peak_mib:.1f} MB")

DistributedManager.cleanup()
