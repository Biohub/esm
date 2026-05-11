"""Test loading ESMC 6B locally and running a forward pass.

Usage:
    python tests/test_load_esmc_6b.py
"""

import torch

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, ESMProteinTensor, LogitsConfig, LogitsOutput
from esm.utils.constants.models import ESMC_6B, ESMC_600M

TEST_SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPEDLDA"


def test_load_and_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading ESMC 6B via from_pretrained ...")
    model = ESMC.from_pretrained(ESMC_600M, device=device, use_flash_attn=torch.cuda.is_available())
    assert isinstance(model, ESMC), f"Expected ESMC but got {type(model)}"
    print(f"  d_model={model.embed.embedding_dim}, layers={len(model.transformer.blocks)}")

    print("Testing encode ...")
    protein = ESMProtein(sequence=TEST_SEQUENCE)
    protein_tensor = model.encode(protein)
    breakpoint()
    assert isinstance(protein_tensor, ESMProteinTensor)
    print(f"  sequence tokens: {protein_tensor.sequence.shape}")

    print("Testing decode ...")
    decoded = model.decode(protein_tensor)
    breakpoint()
    assert isinstance(decoded, ESMProtein)
    assert decoded.sequence == TEST_SEQUENCE, f"Round-trip mismatch: {decoded.sequence}"
    print(f"  decoded sequence matches: OK")


if __name__ == "__main__":
    test_load_and_forward()
