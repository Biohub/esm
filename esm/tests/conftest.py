"""Shared fixtures for the ESMC test suite.

Tiny randomly-initialised models cover the structural contracts on CPU in
milliseconds. The published-weight fixtures download once per session and skip
when the Hub is unreachable, so an offline runner degrades to the structural
tests instead of failing.
"""

import os
import random
from pathlib import Path

import pytest
import torch

REFERENCE_DIR = Path(__file__).parent / "reference"

# Small enough to build instantly, wide enough that the head dim is sane. Every
# field the architecture depends on is present: `from_dict` defaults none of them.
TINY_ESMC = dict(
    vocab_size=64,
    hidden_size=32,
    num_attention_heads=4,
    num_hidden_layers=2,
    pad_token_id=1,
    mask_token_id=32,
)
TINY_SEQUENCES = ["MQIFVKTLTGKT", "MKV"]

ESMC_300M_REPO = "biohub/ESMC-300M"
#: Subdirectory holding this model under ``$ESM_ESMC_WEIGHTS_ROOT``.
ESMC_300M_STAGED_DIR = "ESMC-300M"


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

# Two real sequences, spliced into three length regimes. ESMC has no
# length-dependent branch other than the RoPE cache, so the regimes are picked
# where numerics and kernels change behaviour:
#
#   short  (10 aa, 12 tokens)   Every softmax runs over a handful of keys.
#   medium (129 aa, 131 tokens) Past ESMC-300M's head_dim (64), and 131 is prime,
#       so it is not a multiple of any fused-attention tile (8/16/32/64/128) and
#       every kernel has to handle a ragged tail.
#   long   (410 aa, 412 tokens) ~3x medium. Forces the RoPE cache to take its
#       grow branch when a model sees medium first, and puts ~35x as many terms
#       in each softmax denominator as short - which is where the CPU/GPU and
#       fused/reference tolerances separate.
UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
LYSOZYME = (
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLC"
    "NIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
)

SHORT_SEQUENCE = UBIQUITIN[:10]
MEDIUM_SEQUENCE = LYSOZYME
# A tandem two-domain construct rather than a natural protein: length is the
# point, and repeating real domains keeps the tokens in-distribution so
# perplexity stays a meaningful number.
LONG_SEQUENCE = UBIQUITIN + LYSOZYME + UBIQUITIN + LYSOZYME

# Ordered short -> long; tests parametrize over the keys.
SEQUENCES = {"short": SHORT_SEQUENCE, "medium": MEDIUM_SEQUENCE, "long": LONG_SEQUENCE}
LENGTH_NAMES = tuple(SEQUENCES)

# A typo in either literal above would silently change every reference value.
assert [len(s) for s in SEQUENCES.values()] == [10, 129, 410]

# Only the 20 canonical residues: X/B/U/Z/O and the gap, insertion and
# chain-break tokens are in the vocabulary but are not amino acids, and a random
# sequence containing them tests the vocabulary rather than the model.
CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
RANDOM_SEQUENCE_SEED = 20241203
# Spans the three regimes and the awkward edges: a single residue, an exact power
# of two and both its neighbours.
RANDOM_SEQUENCE_LENGTHS = (1, 2, 5, 31, 63, 64, 65, 127, 200, 410)


def random_amino_acid_sequences(
    lengths: tuple[int, ...] | list[int] = RANDOM_SEQUENCE_LENGTHS,
    seed: int = RANDOM_SEQUENCE_SEED,
) -> list[str]:
    """Seeded random sequences over the canonical amino acids."""
    rng = random.Random(seed)
    return ["".join(rng.choices(CANONICAL_AMINO_ACIDS, k=n)) for n in lengths]


@pytest.fixture(scope="session")
def random_sequences() -> list[str]:
    return random_amino_acid_sequences()


# ---------------------------------------------------------------------------
# Published weights
# ---------------------------------------------------------------------------


def staged_model_dir(subdirectory: str) -> str | None:
    """A local copy of a model, when ``ESM_ESMC_WEIGHTS_ROOT`` points at one.

    Read straight from the environment: this package is the bottom import layer.
    """
    root = os.environ.get("ESM_ESMC_WEIGHTS_ROOT")
    if not root:
        return None
    directory = Path(root) / subdirectory
    return str(directory) if (directory / "config.json").exists() else None


def _skip_if_unreachable(repo: str) -> str:
    """Resolve a Hub snapshot, or skip when the runner genuinely has no access.

    Only offline and transport failures are a skip. Anything else - a bad token, a
    repo that has moved, a resolver bug - has to fail, or a regression there would
    quietly turn the published-weight coverage green.
    """
    from huggingface_hub.errors import LocalEntryNotFoundError, OfflineModeIsEnabled
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import Timeout

    from esm.models.hub import resolve_model_dir

    try:
        return resolve_model_dir(repo)
    except (
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        RequestsConnectionError,
        Timeout,
    ) as exc:
        pytest.skip(
            f"{repo} unreachable ({type(exc).__name__}); skipping"
        )  # ty:ignore[too-many-positional-arguments]


#: Fixtures that resolve a real published checkpoint. Anything depending on one
#: needs network (or staged weights) and takes seconds rather than milliseconds.
WEIGHTS_FIXTURES = frozenset({"esmc_300m_dir"})


def pytest_collection_modifyitems(items):
    """Mark every published-weight test ``merge_only``.

    Derived from the fixture graph rather than written on each test, so a
    weights-backed test cannot be added without the marker.
    """
    for item in items:
        if not WEIGHTS_FIXTURES.isdisjoint(getattr(item, "fixturenames", ())):
            # `gpu` already routes to its own job; a second marker would only
            # deselect it there.
            if not item.get_closest_marker("gpu"):
                item.add_marker(pytest.mark.merge_only)


@pytest.fixture(scope="session")
def tiny_esmc_config():
    from esm.models.esmc import EsmcConfig

    return EsmcConfig(
        vocab_size=TINY_ESMC["vocab_size"],
        hidden_size=TINY_ESMC["hidden_size"],
        num_attention_heads=TINY_ESMC["num_attention_heads"],
        num_hidden_layers=TINY_ESMC["num_hidden_layers"],
    )


@pytest.fixture
def tiny_esmc(tiny_esmc_config):
    from esm.models.esmc import EsmcModel

    torch.manual_seed(0)
    return EsmcModel(tiny_esmc_config).eval()


@pytest.fixture
def tiny_esmc_mlm(tiny_esmc_config):
    from esm.models.esmc import EsmcForMaskedLM

    torch.manual_seed(0)
    return EsmcForMaskedLM(tiny_esmc_config).eval()


@pytest.fixture(scope="session")
def esmc_tokenizer():
    from esm.models.esmc import EsmcTokenizer

    return EsmcTokenizer()


@pytest.fixture(scope="session")
def esmc_300m_dir():
    """A local copy of the weights when one is staged, else the published repo."""
    return staged_model_dir(ESMC_300M_STAGED_DIR) or _skip_if_unreachable(
        ESMC_300M_REPO
    )


@pytest.fixture(scope="session")
def esmc_300m(esmc_300m_dir):
    from esm.models.esmc import EsmcForMaskedLM

    return EsmcForMaskedLM.from_pretrained(esmc_300m_dir, device="cpu")


@pytest.fixture(scope="session")
def esmc_300m_cpu_bf16(esmc_300m_dir):
    from esm.models.esmc import EsmcForMaskedLM

    return EsmcForMaskedLM.from_pretrained(
        esmc_300m_dir, device="cpu", dtype=torch.bfloat16
    )


@pytest.fixture(scope="session")
def esmc_300m_cpu_reference(esmc_300m, esmc_tokenizer):
    """CPU / fp32 logits and hidden states for each length, computed once.

    The runtime counterpart of the checked-in reference file: that pins the
    numbers across commits, this is the full tensor other configurations are
    compared against rather than a stored summary.
    """
    reference = {}
    for name, sequence in SEQUENCES.items():
        enc = esmc_tokenizer(sequence, return_tensors="pt")
        with torch.no_grad():
            out = esmc_300m(**enc)
        reference[name] = {
            "logits": out.logits.float(),
            "last_hidden_state": out.last_hidden_state.float(),
        }
    return reference
