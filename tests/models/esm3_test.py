"""ESM3 in reduced precision."""

import torch

from esm.models.esm3 import EncodeInputs
from esm.utils.constants import esm3 as C

D_MODEL = 32
B, L = 2, 10


def test_encode_inputs_runs_in_bf16_with_the_default_plddt():
    """Issue #178: ``ESM3.forward`` fills pLDDT in with ``.float()``.

    Those fp32 values go through ``rbf`` into the bf16 pLDDT projections, so the
    first forward of a bf16 model fails with a dtype mismatch.
    """
    torch.manual_seed(0)
    encoder = EncodeInputs(D_MODEL).to(torch.bfloat16).eval()

    # The inputs ESM3.forward builds for a sequence-only call.
    sequence_tokens = torch.randint(4, 24, (B, L))
    structure_tokens = torch.full((B, L), C.STRUCTURE_MASK_TOKEN)
    average_plddt = torch.ones(B, L)
    per_res_plddt = torch.zeros(B, L)
    ss8_tokens = torch.full((B, L), C.SS8_PAD_TOKEN)
    sasa_tokens = torch.full((B, L), C.SASA_PAD_TOKEN)
    function_tokens = torch.full((B, L, 8), C.INTERPRO_PAD_TOKEN)
    residue_annotation_tokens = torch.full((B, L, 16), C.RESIDUE_PAD_TOKEN)

    with torch.no_grad():
        out = encoder(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )

    assert out.dtype == torch.bfloat16
    assert out.shape == (B, L, D_MODEL)
    assert torch.isfinite(out.float()).all()
