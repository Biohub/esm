import os, math, gc
from collections import OrderedDict
from time import time

import torch
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmfold2.distributed import (
    DistributedManager,
    wrap_model_with_cp,
)
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    LigandInput,
    ProteinInput,
    StructurePredictionInput,
)

spi = StructurePredictionInput(  # 5xgo
    sequences=[
        ProteinInput(
            id=["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1", "I1", "J1", "K1", "L1"],
            sequence=(
                "MAHHHHHHVDDDDKMSTAKLVKSKATNLLYTRNDVSDSEKKATVELLNRQVIQFIDLSLITKQAHWNMRG"
                "ANFIAVHEMLDGFRTALIDHLDTMAERAVQLGGVALGTTQVINSKTPLKSYPLDIHNVQDHLKELADRYA"
                "IVANDVRKAIGEAKDDDTADILTAASRDLDKFLWFIECNLDLIQKMGLQNYLQAQIREEG"
            ),
        ),
        LigandInput(id=["M1", "N1"], ccd=["CL"]),
    ]
)

# torchrun entrypoint:  torchrun --nproc_per_node=4 cuequivariance_cp.py
# torchrun sets LOCAL_RANK / WORLD_SIZE / RANK / MASTER_ADDR / MASTER_PORT.
# world_size must be a perfect square (the CP grid is n×n).
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

os.environ["RANK"] = str(local_rank)
torch.cuda.set_device(local_rank)
torch.cuda.reset_peak_memory_stats()

n = math.isqrt(world_size)
DistributedManager.initialize(
    grid_group_sizes=OrderedDict([("dp", 1), ("cp", (n, n))]),
    device_type="cuda",
    backend="nccl",
)
dm = DistributedManager()

model = ESMFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision='bf16').cuda().eval()
# Shard BOTH the folding trunk and the MSA encoder across the n×n CP grid.
# (Trunk-only sharding leaves the MSA encoder running the full L×L pair on
# every rank, which OOMs for large complexes.)
#
# Defaults applied here (override via args): bf16=True (bf16 distributed trunk)
# and offload_esmc=True (move the ~12 GB ESM-C LM to CPU after its one-shot use
# so the trunk reuses that memory). Together: ~2.14x faster, ~18.5 GB lower
# peak vs fp32, quality-neutral. comm="gather" (bit-exact) | "ring" (n>=3).
wrap_model_with_cp(model, dm, comm="ring")
# Fused kernels accelerate the modules *outside* the CP region (notably the
# diffusion sampler / structure head). The CP-wrapped trunk and MSA encoder
# no-op this call, so it never touches the distributed ring math.
model.set_kernel_backend("cuequivariance")

start = time()
with torch.inference_mode():
    result = ESMFold2InputBuilder().fold(
        model, spi, num_loops=10, num_sampling_steps=50,
        num_diffusion_samples=1, seed=0,
    )
end = time()
peak_mib = torch.cuda.max_memory_allocated() / (1024**2)

if local_rank == 0:
    print(f"pLDDT mean: {float(result.plddt.mean()):.3f}, "
          f"pTM: {float(result.ptm):.3f}, ipTM: {float(result.iptm):.3f}")
    print(f"Elapsed: {end - start} sec")
    print(f"Max VRAM: {peak_mib} MB")
    with open("esmfold2_cp_output.cif", "w") as f:
        f.write(result.complex.to_mmcif())

DistributedManager.cleanup()
DistributedManager._state.clear()
gc.collect()
torch.cuda.empty_cache()
