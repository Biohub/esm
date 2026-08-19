"""Regenerate the numerical reference values the test suites compare against.

Run from the repo root when a deliberate numerical change lands::

    python -m esm.tests.regenerate_reference
    python -m esm.tests.regenerate_reference esmfold2

CPU + fp32, so the files reproduce anywhere; every other build configuration is
checked against them rather than against a reference of its own. The writers
import the helpers the tests use rather than restating them, so a stored number
cannot come from a different computation than the one checking it.
"""

import gzip
import pickle
import sys
from pathlib import Path

import torch

from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
from esm.tests.conftest import (
    ESMC_300M_REPO,
    ESMC_300M_STAGED_DIR,
    ESMFOLD2_SEQUENCES,
    REFERENCE_DIR,
    SEQUENCES,
    staged_model_dir,
)
from esm.tests.esmc_test import (
    PPL_MAX_POSITIONS,
    pseudo_perplexity,
)
from esm.tests.esmc_test import (
    REFERENCE_FILE as ESMC_REFERENCE_FILE,
)
from esm.tests.esmfold2_execution_test import (
    DISTANCE_DRIFT_TOLERANCE,
    FORWARD_SEED,
    MODEL_SEED,
    N_DISTOGRAM_TOKENS,
    NUM_DIFFUSION_SAMPLES,
    NUM_LOOPS,
    NUM_SAMPLING_STEPS,
    REP_ATOM_DISTANCE_BIN_EDGES,
    config_fingerprint,
    execution_case,
    reference_config,
)
from esm.tests.esmfold2_execution_test import (
    REFERENCE_FILE as ESMFOLD2_REFERENCE_FILE,
)

# One residue's logit row plus the argmax path moves on any real numerical
# change, so there is no need to store the full logits tensor.
N_LOGITS = 16


def _write(path: Path, payload: dict) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=9) as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {path}")


def esmc_300m() -> None:
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
            f"  {name:<12} L={len(sequence):<4} "
            f"ppl={measured.perplexity:.6f} recovery={measured.recovery:.4f}"
        )

    _write(
        ESMC_REFERENCE_FILE,
        {
            "repo": ESMC_300M_REPO,
            "device": "cpu",
            "dtype": "float32",
            "n_logits": N_LOGITS,
            "ppl_max_positions": PPL_MAX_POSITIONS,
            "cases": cases,
        },
    )


def esmfold2_tiny() -> None:
    """A tiny randomly-initialised ESMFold2 at a fixed seed, one entry per length
    regime. No checkpoint and no network, so it regenerates in seconds.
    """
    config = reference_config()

    cases = {}
    for name, sequence in ESMFOLD2_SEQUENCES.items():
        case = execution_case(config, sequence)
        cases[name] = {"sequence": sequence, **case}
        print(
            f"  {name:<12} L={case['n_residues']:<4} "
            f"atoms={case['n_atoms_valid']:<5} ptm={case['ptm']:.6f}"
        )

    _write(
        ESMFOLD2_REFERENCE_FILE,
        {
            "architecture": "EsmFold2Model",
            "device": "cpu",
            "dtype": "float32",
            "model_seed": MODEL_SEED,
            # Not just the sampler's: the trunk draws from the global generator
            # too, so this seeds the whole forward.
            "forward_seed": FORWARD_SEED,
            "sampler": {
                "num_loops": NUM_LOOPS,
                "num_sampling_steps": NUM_SAMPLING_STEPS,
                "num_diffusion_samples": NUM_DIFFUSION_SAMPLES,
                "chunk_size": None,
                "kernel_backend": None,
            },
            "n_distogram_tokens": N_DISTOGRAM_TOKENS,
            "rep_atom_distance_bin_edges": list(REP_ATOM_DISTANCE_BIN_EDGES),
            "distance_drift_tolerance": DISTANCE_DRIFT_TOLERANCE,
            # The fingerprint is the loud failure when the architecture moves;
            # the config itself is what makes that failure diagnosable.
            "config_fingerprint": config_fingerprint(config),
            "config": config.to_dict(),
            "cases": cases,
        },
    )


REGENERATORS = {"esmc": esmc_300m, "esmfold2": esmfold2_tiny}


def main() -> None:
    for name in sys.argv[1:] or list(REGENERATORS):
        REGENERATORS[name]()


if __name__ == "__main__":
    main()
