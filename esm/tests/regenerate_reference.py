"""Regenerate the numerical reference values used by ``esmc_test.py``.

Run from the repo root when a deliberate numerical change lands::

    python -m esm.tests.regenerate_reference

CPU + fp32, so the file reproduces anywhere; every other build configuration is
checked against it rather than against a reference of its own.
"""

import gzip
import pickle

import torch

from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
from esm.tests.conftest import (
    ESMC_300M_REPO,
    ESMC_300M_STAGED_DIR,
    REFERENCE_DIR,
    SEQUENCES,
    staged_model_dir,
)
from esm.tests.esmc_test import (
    PPL_MAX_POSITIONS,
    REFERENCE_FILE,
    pseudo_perplexity,
)

# One residue's logit row plus the argmax path moves on any real numerical
# change, so there is no need to store the full logits tensor.
N_LOGITS = 16


def main() -> None:
    torch.manual_seed(0)
    source = staged_model_dir(ESMC_300M_STAGED_DIR) or ESMC_300M_REPO
    model = EsmcForMaskedLM.from_pretrained(source, device="cpu")
    tokenizer = EsmcTokenizer()

    cases = {}
    for name, sequence in SEQUENCES.items():
        enc = tokenizer(sequence, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits.float()
        measured = pseudo_perplexity(model, tokenizer, sequence)
        cases[name] = {
            "sequence": sequence,
            "n_residues": len(sequence),
            "logits_shape": list(logits.shape),
            "first_residue_logits": logits[0, 1, :N_LOGITS].tolist(),
            "argmax": logits.argmax(-1)[0].tolist(),
            # Per-position L2 norms of the post-LayerNorm output: O(1) in
            # magnitude, so one tolerance is meaningful at every dtype.
            "last_hidden_state_norms": out.last_hidden_state.float()[0]
            .norm(dim=-1)
            .tolist(),
            "pseudo_perplexity": measured.perplexity,
            "top1_recovery": measured.recovery,
        }
        print(
            f"  {name:<7} L={len(sequence):<4} "
            f"ppl={measured.perplexity:.6f} recovery={measured.recovery:.4f}"
        )

    payload = {
        "repo": ESMC_300M_REPO,
        "device": "cpu",
        "dtype": "float32",
        "n_logits": N_LOGITS,
        "ppl_max_positions": PPL_MAX_POSITIONS,
        "cases": cases,
    }

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(REFERENCE_FILE, "wb", compresslevel=9) as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {REFERENCE_FILE}")


if __name__ == "__main__":
    main()
