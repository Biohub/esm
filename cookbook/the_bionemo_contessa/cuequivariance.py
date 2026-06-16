import gc
from time import time

import torch
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    LigandInput,
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
        ),
        LigandInput(id=["C1", "E1"], ccd=["GDP"]),
        LigandInput(id="D1", ccd=["TRS"]),
    ]
)

torch.cuda.reset_peak_memory_stats()

model = ESMFold2Model.from_pretrained("biohub/ESMFold2", esmc_precision='bf16').cuda().eval()
model.set_kernel_backend("cuequivariance")

start = time()
with torch.inference_mode():
    result = ESMFold2InputBuilder().fold(
        model, spi, num_loops=10, num_sampling_steps=50,
        num_diffusion_samples=1, seed=0,
    )
end = time()
peak_mib = torch.cuda.max_memory_allocated() / (1024**2)

print(f"pLDDT mean: {float(result.plddt.mean()):.3f}, "
      f"pTM: {float(result.ptm):.3f}, ipTM: {float(result.iptm):.3f}")
print(f"Elapsed: {end - start} sec")
print(f"Max VRAM: {peak_mib} MB")
with open("esmfold2_output.cif", "w") as f:
    f.write(result.complex.to_mmcif())
