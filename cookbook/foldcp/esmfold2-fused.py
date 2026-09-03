from time import time

import torch

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    EsmFold2Model,
    MolecularComplexResult,
    ProteinInput,
    StructurePredictionInput,
)

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

torch.cuda.reset_peak_memory_stats()

model = (
    EsmFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision="bf16")
    .cuda()
    .eval()
)
model.set_kernel_backend("fused")

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

print(
    f"pLDDT mean: {float(result.plddt.mean()):.3f}, "
    f"pTM: {float(result.ptm):.3f}, ipTM: {float(result.iptm):.3f}"
)
print(f"Elapsed: {end - start:.2f} sec")
print(f"Max VRAM: {peak_mib:.1f} MB")
