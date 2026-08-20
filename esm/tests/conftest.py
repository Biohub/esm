"""Shared fixtures for the ESMC and ESMFold2 test suites.

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
ESMFOLD2_REPO = "biohub/ESMFold2"
ESMFOLD2_EXPERIMENTAL_REPO = "biohub/ESMFold2-Experimental"
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
        pytest.skip(f"{repo} unreachable ({type(exc).__name__}); skipping")  # ty:ignore[too-many-positional-arguments]


#: Fixtures that resolve a real published checkpoint. Anything depending on one
#: needs network (or staged weights) and takes seconds rather than milliseconds.
WEIGHTS_FIXTURES = frozenset(
    {"esmc_300m_dir", "esmfold2_dir", "esmfold2_experimental_dir"}
)


def pytest_configure(config):
    """Point ``conformers.py`` at the mounted CCD before it is imported."""


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


# ---------------------------------------------------------------------------
# ESMFold2 length regimes
# ---------------------------------------------------------------------------

# ESMC's only length-dependent branch is the RoPE cache, but ESMFold2 also bands
# attention at ``sliding_window``, chunks trimul and Transition at
# ``_DEFAULT_CHUNK_SIZE``, pads the atom axis to a multiple of 32, and saturates
# the relative-position bins. Each regime sits on one of those boundaries:
#
#   tiny         12   The length the rest of the ESMFold2 tests use.
#   sub_window   96   Shorter than sliding_window=128, so the band never wraps.
#   window_edge 128   Exactly the window; an off-by-one in the band bounds is
#       visible here and at no other length.
#   over_window 197   The band wraps. Prime, so no fused tile divides it.
#   chunk_edge   64   Exactly _DEFAULT_CHUNK_SIZE: one full chunk, no remainder.
#   chunk_split  65   One past it: two chunks, the second holding one row.
#   long        384   Past n_relative_residx_bins=32 by an order of magnitude, so
#       relpos saturates, and ~9x window_edge's pair terms.
ESMFOLD2_LENGTHS = {
    "tiny": 12,
    "sub_window": 96,
    "chunk_edge": 64,
    "chunk_split": 65,
    "window_edge": 128,
    "over_window": 197,
    "long": 384,
}

_LENGTH_BASE = UBIQUITIN + LYSOZYME


def _protein_of_length(n: int) -> str:
    """A real-residue sequence of exactly ``n`` residues.

    Sliced from tandem ubiquitin/lysozyme rather than generated, so the residue
    composition - and so the atom count per token - stays realistic.
    """
    repeats = -(-n // len(_LENGTH_BASE))
    return (_LENGTH_BASE * repeats)[:n]


ESMFOLD2_SEQUENCES = {
    name: _protein_of_length(n) for name, n in ESMFOLD2_LENGTHS.items()
}
ESMFOLD2_LENGTH_NAMES = tuple(ESMFOLD2_SEQUENCES)

# A typo in a literal above would move every stored reference value at once.
assert [len(s) for s in ESMFOLD2_SEQUENCES.values()] == list(ESMFOLD2_LENGTHS.values())
assert ESMFOLD2_SEQUENCES["tiny"] == "MQIFVKTLTGKT"

# Glycine is the only residue with exactly 4 heavy atoms, so poly-glycine is the
# one sequence family whose atom count is predictable without the residue
# tables: 8 residues is exactly one 32-atom block, and one alanine (5 atoms) past
# it forces 27 rows of padding.
ATOM_ALIGNED_SEQUENCE = "G" * 8
ATOM_PADDED_SEQUENCE = "G" * 8 + "A"


# ---------------------------------------------------------------------------
# ESMFold2 complexes
# ---------------------------------------------------------------------------

# Distinct enough that entity grouping is unambiguous, short enough that a
# 4-chain complex is still a fast CPU forward.
CHAIN_A = UBIQUITIN[:24]
CHAIN_B = LYSOZYME[:18]
CHAIN_C = UBIQUITIN[30:52]

# Chain multiplicities are the point: a homodimer has one entity and two sym_ids,
# A2B2 has two entities with two copies each, and the mixed case has a repeated
# entity that is not adjacent - which is where an implementation assigning
# sym_id by position rather than by entity goes wrong.
ESMFOLD2_COMPLEXES = {
    "homodimer": (CHAIN_A, CHAIN_A),
    "heterodimer": (CHAIN_A, CHAIN_B),
    "a2b2": (CHAIN_A, CHAIN_A, CHAIN_B, CHAIN_B),
    "mixed4": (CHAIN_A, CHAIN_B, CHAIN_A, CHAIN_C),
}
ESMFOLD2_COMPLEX_NAMES = tuple(ESMFOLD2_COMPLEXES)


def protein_complex_input(chains):
    """A ``StructurePredictionInput`` of protein chains, ids A, B, C, ...

    Chain ids are positional and always distinct; entity identity is derived from
    the sequence by the featurizer, not from the id.
    """
    from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput

    return StructurePredictionInput(
        sequences=[
            ProteinInput(id=chr(ord("A") + i), sequence=seq)
            for i, seq in enumerate(chains)
        ]
    )


def _synthetic_lm_states(model, features, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return (
        torch.randn(
            1,
            features["res_type"].shape[-1],
            model.config.lm_num_layers + 1,
            model.config.lm_d_model,
        )
        * 0.02
    )


def esmfold2_inputs(model, seq: str = "MQIFVKTLTGKT", seed: int = 0):
    """Protein features plus synthetic LM states, so no PLM backbone is needed."""
    from esm.models.esmfold2.protein_utils import prepare_protein_features

    features = prepare_protein_features(seq)
    return features, _synthetic_lm_states(model, features, seed)


def esmfold2_complex_features(model, chains, seed: int = 0):
    """Multi-chain features plus synthetic LM states.

    Goes through the full featurizer rather than ``prepare_protein_features``,
    because chain identity tensors only exist on that path.
    """
    from esm.models.esmfold2.prepare_input import prepare_esmfold2_input

    raw, chain_infos = prepare_esmfold2_input(protein_complex_input(chains))
    # prepare_esmfold2_input returns unbatched tensors; only
    # ESMFold2InputBuilder.prepare_input adds the leading axis forward requires.
    features = {k: v[None] for k, v in raw.items() if isinstance(v, torch.Tensor)}
    return features, _synthetic_lm_states(model, features, seed), chain_infos


# ---------------------------------------------------------------------------
# ESMFold2 models and weights
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ccd_pickle() -> str:
    """The CCD pickle every non-protein featurization path needs.

    ``load_ccd`` reaches for this on its own, so a test that looks offline is
    not; depending on this fixture makes that reach explicit and skips the test
    when the pickle is not mounted, which is the case on a CI runner.

    Deliberately no Hub fallback. ``biohub/ESMFold2:ccd.pkl`` is a *different*
    CCD release from the mounted one, and the release changes per-residue atom
    ordering - enough to move the reversed-atom controls in
    ``esmfold2_sampler_test.py`` by several Angstrom. Silently substituting it
    would make those numbers depend on which copy the runner happened to find.
    """
    path = os.environ.get("ESMCFOLD_CCD_PATH", "")
    if not path or not Path(path).exists():
        pytest.skip(f"CCD pickle not mounted at {path}")  # ty:ignore[too-many-positional-arguments]
    return path


@pytest.fixture(scope="session")
def esmfold2_dir() -> str:
    return _skip_if_unreachable(ESMFOLD2_REPO)


@pytest.fixture(scope="session")
def esmfold2_experimental_dir() -> str:
    return _skip_if_unreachable(ESMFOLD2_EXPERIMENTAL_REPO)


@pytest.fixture(scope="session")
def tiny_esmfold2_config():
    """Smallest ESMFold2 that still exercises trunk, structure and confidence heads."""
    from esm.models.esmfold2 import EsmFold2Config

    return EsmFold2Config(
        hidden_size=32,
        pairwise_hidden_size=16,
        num_loops=1,
        num_diffusion_samples=1,
        lm_d_model=32,
        lm_num_layers=2,
        folding_trunk_num_hidden_layers=1,
        folding_trunk_num_attention_heads=2,
        # The diffusion module dominates parameter count, so shrink its token and
        # atom stacks too; single_inputs_size / c_s_inputs are fixed by the featurizer.
        structure_head={
            "inference_num_steps": 2,
            "distogram_bins": 8,
            "diffusion_module": {
                # 3D RoPE needs head_dim >= 2 * (3 * spatial_pairs + uid_pairs) = 32.
                "atom_encoder": {
                    "hidden_size": 64,
                    "num_attention_heads": 2,
                    "num_hidden_layers": 1,
                },
                "token_hidden_size": 32,
                "c_z": 16,
                "fourier_dim": 16,
                "token_num_blocks": 1,
                "token_num_heads": 2,
            },
        },
        confidence_head={
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "distogram_bins": 8,
            "num_plddt_bins": 8,
            "num_pae_bins": 8,
            "num_pde_bins": 8,
        },
        parcae_num_coda_layers=1,
        lm_encoder={"num_hidden_layers": 1},
    )


@pytest.fixture
def tiny_esmfold2(tiny_esmfold2_config):
    from esm.models.esmfold2 import EsmFold2Model

    torch.manual_seed(0)
    model = EsmFold2Model(tiny_esmfold2_config).eval()
    model.set_chunk_size(None)
    return model


@pytest.fixture(scope="session")
def kernel_esmfold2_config():
    """Like the tiny config but with tile-aligned pair dims.

    The fused Triton kernels assert ``N % TILE_N == 0``, so a 16-wide pair dim
    trips an assertion inside fused_dual_gemm. 128 is aligned for every tile.
    """
    from esm.models.esmfold2 import EsmFold2Config

    return EsmFold2Config(
        hidden_size=128,
        pairwise_hidden_size=128,
        num_loops=1,
        num_diffusion_samples=1,
        lm_d_model=32,
        lm_num_layers=2,
        folding_trunk_num_hidden_layers=1,
        folding_trunk_num_attention_heads=2,
        structure_head={
            "inference_num_steps": 2,
            "distogram_bins": 8,
            "diffusion_module": {
                "atom_encoder": {
                    "hidden_size": 64,
                    "num_attention_heads": 2,
                    "num_hidden_layers": 1,
                },
                "token_hidden_size": 128,
                "c_z": 128,
                "fourier_dim": 16,
                "token_num_blocks": 1,
                "token_num_heads": 2,
            },
        },
        confidence_head={
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "distogram_bins": 8,
            "num_plddt_bins": 8,
            "num_pae_bins": 8,
            "num_pde_bins": 8,
        },
        parcae_num_coda_layers=1,
        lm_encoder={"num_hidden_layers": 1},
    )
