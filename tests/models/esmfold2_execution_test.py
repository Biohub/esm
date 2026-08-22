"""ESMFold2 execution correctness against the stored tiny-model reference.

A randomly-initialised model at a fixed seed on CPU in fp32, checked against
``reference/esmfold2_tiny_cpu_fp32.pkl.gz``. No checkpoint and no network, so
this runs in the fast tier and is what catches architecture drift.

**Nothing in an ESMFold2 forward is deterministic given inputs.** Two sources,
both live under ``eval()``:

1. ``_init_pair_state`` draws a fresh ``trunc_normal_`` on every forward - the
   release architecture starts its parcae recurrence from noise by construction.
2. ``_run_one_loop`` applies the per-loop LM dropout as
   ``F.dropout(..., training=True)`` with ``training`` hardcoded, and
   ``EsmFold2LmEncoderConfig`` defaults to ``lm_dropout=0.25``.

So ``torch.manual_seed(FORWARD_SEED)`` immediately before every forward is the
only thing that makes any of these numbers reproducible, and ``FORWARD_SEED`` is
in the reference envelope for that reason. Switching the dropout off is not an
alternative: source 1 remains.

The model is split at the sampler boundary, because the two halves need
different kinds of reference value:

* **Trunk.** Given the seed, the distogram logits and the pair and single
  representations are reproducible exactly, so they get the ESMC treatment: a
  fixed-width logit slice, the exact argmax, and per-position L2 norms.
* **Sampler.** ``DiffusionStructureHead.sample`` re-centres and re-rotates the
  structure at every ODE step, so the pose is defined only up to a rigid motion
  even at a fixed seed. Nothing here stores raw coordinates: the stored
  quantities are rigid-invariant (per-atom radii, representative-atom distances)
  or pose-free (pLDDT, pTM).

``regenerate_reference.py`` imports the helpers below rather than restating them,
so a stored number cannot come from a different computation than the one
checking it.
"""

import contextlib
import functools
import gzip
import hashlib
import json
import math
import pickle

import pytest
import torch
from torch import Tensor

from esm.models.esmfold2 import EsmFold2Config, EsmFold2Model
from esm.models.esmfold2 import layers as _layers
from tests import conftest as _conftest
from tests.conftest import (
    ESMFOLD2_LENGTH_NAMES,
    ESMFOLD2_SEQUENCES,
    REFERENCE_DIR,
    esmfold2_inputs,
)

REFERENCE_FILE = REFERENCE_DIR / "esmfold2_tiny_cpu_fp32.pkl.gz"

#: Seeds ``EsmFold2Model``'s random initialisation. Matches the ``tiny_esmfold2``
#: fixture, which ``test_the_reference_model_is_the_fixture_model`` pins.
MODEL_SEED = 0
#: Reset immediately before every forward, not once at import: the trunk draws
#: from the global generator twice per forward, so a stored number means nothing
#: unless the generator is at a known state at the call.
FORWARD_SEED = 0

# A forward is O(L^2); these keep the whole module in the fast tier. They live in
# the reference envelope too, so a test cannot compare against a file generated
# with different ones.
NUM_LOOPS = 1
NUM_SAMPLING_STEPS = 2
NUM_DIFFUSION_SAMPLES = 1

# Width of the stored distogram slice: the full [L, L, bins] tensor is far too
# large to check in, while one token's row still moves on any real numerical
# change.
N_DISTOGRAM_TOKENS = 16

#: 2 Å bins to 32 Å. The measured distances reach 30.6 Å, so nothing is clamped
#: today; the clamp exists so a change pushing a distance past the top edge still
#: leaves the counts summing to the pair count.
REP_ATOM_DISTANCE_BIN_EDGES = tuple(float(x) for x in torch.linspace(0.0, 32.0, 17))

#: Budget for how far a representative-atom distance may move. Twice the 1e-2 Å
#: the per-atom radii are compared at, because a pairwise distance moves by up to
#: the sum of its two endpoints' drift.
DISTANCE_DRIFT_TOLERANCE = 2e-2


@functools.lru_cache(maxsize=1)
def reference_values() -> dict:
    # A missing file is a broken checkout, not a reason to skip.
    assert REFERENCE_FILE.exists(), (
        f"{REFERENCE_FILE} is missing. It is checked in; regenerate with "
        f"`python -m tests.regenerate_reference esmfold2`."
    )
    with gzip.open(REFERENCE_FILE, "rb") as f:
        return pickle.load(f)


def reference_config() -> EsmFold2Config:
    """The exact config the reference was generated from.

    Reached through the ``tiny_esmfold2_config`` fixture rather than restated, so
    the regenerator and the tests cannot describe different models. pytest 8.3
    hands back the plain function and 8.4 wraps it; both set ``__wrapped__``.
    """
    fixture = _conftest.tiny_esmfold2_config
    return getattr(fixture, "__wrapped__", fixture)()


def config_fingerprint(config: EsmFold2Config) -> str:
    """SHA-256 over the config's canonical JSON.

    A one-line failure when the architecture moves, so the reference file is
    invalidated loudly instead of being compared against stale numbers.
    """
    canonical = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@contextlib.contextmanager
def reference_attention():
    """Force the pure-PyTorch attention path in the atom encoder.

    ``flash_attn`` imports on a CPU box but its kernels are CUDA-only, so the
    varlen call raises ``NotImplementedError`` mid-forward. Patched here rather
    than in an autouse fixture because ``regenerate_reference`` needs the same
    path and gets no fixtures.
    """
    previous = _layers.FLASH_ATTN_AVAILABLE
    _layers.FLASH_ATTN_AVAILABLE = False
    try:
        yield
    finally:
        _layers.FLASH_ATTN_AVAILABLE = previous


def build_reference_model(config: EsmFold2Config) -> EsmFold2Model:
    """The CPU / fp32 model every stored number comes from."""
    torch.manual_seed(MODEL_SEED)
    model = EsmFold2Model(config).eval()
    model.set_chunk_size(None)
    model.set_kernel_backend(None)
    return model


@contextlib.contextmanager
def capture_representations(model: EsmFold2Model):
    """Hook the trunk's final pair state and the single representation.

    A forward hook rather than a new output flag: neither tensor is returned by
    ``forward``, and widening the published API to serve a test is the wrong
    trade. ``parcae_coda`` is the last module to touch ``z`` before the distogram
    head and the sampler; ``inputs_embedder`` produces the single representation
    the sampler receives as ``s_inputs``.
    """
    captured: dict[str, Tensor] = {}

    def record(key: str):
        def hook(module, args, output):
            captured[key] = output

        return hook

    handles = [
        model.parcae_coda.register_forward_hook(record("pair")),
        model.inputs_embedder.register_forward_hook(record("single")),
    ]
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def pair_row_norms(pair: Tensor) -> Tensor:
    """``sqrt(mean_j ||z[i, j]||^2)`` - one number per token, not per token pair.

    A per-token-pair norm would be L^2 numbers. Averaging over the second token
    axis keeps it O(L) and O(1) in magnitude, and stays asymmetric in the two
    token axes - transposing them moves it, which is the axis-permutation bug an
    argmax over 8 distogram bins would not see.
    """
    return pair.float().flatten(-2).norm(dim=-1) / math.sqrt(pair.shape[-2])


def single_norms(single: Tensor) -> Tensor:
    """Per-position L2 norm of the single representation."""
    return single.float().norm(dim=-1)


def atom_radii(coords: Tensor, atom_attention_mask: Tensor) -> Tensor:
    """Per-atom distance from the centroid of the valid atoms.

    Rigid-invariant, which is the point: the sampler re-centres and re-rotates
    the structure at every ODE step. Padded atoms are dropped rather than centred
    in, so a padding-mask bug moves every radius.
    """
    valid = coords.float()[atom_attention_mask.bool()]
    return (valid - valid.mean(dim=0)).norm(dim=-1)


def rep_atom_distances(coords: Tensor, distogram_atom_idx: Tensor) -> Tensor:
    """Upper triangle of the representative-atom distance matrix.

    "Representative" is the featurizer's own choice - CB where the residue has
    one, CA for glycine - so this is the atom pairing the distogram head predicts
    over, and the two halves of the reference describe one geometry.
    """
    points = coords.float()[distogram_atom_idx.long()]
    distances = torch.cdist(points.unsqueeze(0), points.unsqueeze(0))[0]
    rows, cols = torch.triu_indices(len(points), len(points), offset=1)
    return distances[rows, cols]


def n_distances_near_bin_edge(distances: Tensor) -> int:
    """How many distances sit within ``DISTANCE_DRIFT_TOLERANCE`` of a bin edge.

    The histogram's error budget, computed per case rather than guessed at once:
    only these distances can cross an edge under a legitimate numerical change,
    and each crossing moves exactly two counts by one. A single flat fraction
    cannot do the job - the tiny regime has only 66 pairs, so one crossing there
    is already 1.5% of them, while the same fraction at L=384 would be 1100
    counts of slack.
    """
    edges = torch.tensor(REP_ATOM_DISTANCE_BIN_EDGES)
    clamped = distances.clamp(max=float(edges[-1]))
    nearest = (clamped[:, None] - edges[None, :]).abs().amin(dim=1)
    return int((nearest < DISTANCE_DRIFT_TOLERANCE).sum())


def rep_atom_distance_histogram(distances: Tensor) -> Tensor:
    """Bin counts over ``REP_ATOM_DISTANCE_BIN_EDGES``, clamped into range.

    A histogram rather than the L^2 distances themselves, for the same size
    reason as ``pair_row_norms`` - and unlike a mean it moves when the structure
    collapses or inflates in a way that preserves its scale.
    """
    edges = torch.tensor(REP_ATOM_DISTANCE_BIN_EDGES)
    counts = torch.histogram(distances.clamp(max=float(edges[-1])), bins=edges).hist
    counts = counts.long()
    assert int(counts.sum()) == distances.numel(), "a distance fell outside every bin"
    return counts


def _stored(value):
    # Clone, or a view would drag its whole base storage into the pickle: the
    # distogram slice alone would carry the full [L, L, bins] tensor.
    return value.detach().clone() if isinstance(value, Tensor) else value


def execution_case(config: EsmFold2Config, sequence: str) -> dict:
    """One reference forward: every quantity the reference file stores.

    Builds its own model so no case can inherit state from the one before it.
    """
    model = build_reference_model(config)
    features, lm_hidden_states = esmfold2_inputs(model, sequence)

    with reference_attention(), capture_representations(model) as captured:
        torch.manual_seed(FORWARD_SEED)
        with torch.no_grad():
            out = model(
                **features,
                lm_hidden_states=lm_hidden_states,
                num_loops=NUM_LOOPS,
                num_diffusion_samples=NUM_DIFFUSION_SAMPLES,
                num_sampling_steps=NUM_SAMPLING_STEPS,
            )
    assert set(captured) == {"pair", "single"}, "a hooked module never ran"

    logits = out["distogram_logits"].float()[0]
    argmax = logits.argmax(-1)
    atom_mask = features["atom_attention_mask"][0]
    coords = out["sample_atom_coords"][0]
    distances = rep_atom_distances(coords, features["distogram_atom_idx"][0])

    case = {
        "n_residues": len(sequence),
        "n_tokens": logits.shape[0],
        "n_atoms_valid": int(atom_mask.sum()),
        "n_atoms_padded": atom_mask.shape[0],
        "distogram_logits_shape": list(out["distogram_logits"].shape),
        # First token's row against the first N tokens, all bins.
        "distogram_logits_slice": logits[0, :N_DISTOGRAM_TOKENS],
        "distogram_argmax_first_row": argmax[0],
        # The whole [L, L] argmax, losslessly summarised: L^2 integers do not
        # belong in a checked-in file, but their bin counts still change if any
        # one of them moves to a different bin.
        "distogram_argmax_bin_counts": torch.bincount(
            argmax.flatten(), minlength=logits.shape[-1]
        ),
        "pair_norms": pair_row_norms(captured["pair"])[0],
        "single_norms": single_norms(captured["single"])[0],
        "atom_radii": atom_radii(coords, atom_mask),
        "n_rep_atom_pairs": distances.numel(),
        "rep_atom_distance_histogram": rep_atom_distance_histogram(distances),
        "n_distances_near_bin_edge": n_distances_near_bin_edge(distances),
        "plddt": out["plddt"].float()[0],
        "ptm": float(out["ptm"][0]),
    }
    return {key: _stored(value) for key, value in case.items()}


@functools.lru_cache(maxsize=None)
def measured_case(name: str) -> dict:
    """The reference forward for one length regime, computed once per process.

    Memoised per regime rather than as one all-regimes fixture: ``addopts``
    carries ``-n``, so each worker would otherwise run all seven forwards to
    serve the single test it was handed.
    """
    return execution_case(reference_config(), ESMFOLD2_SEQUENCES[name])


# ---------------------------------------------------------------------------
# The file's envelope, and that it describes this build
# ---------------------------------------------------------------------------


def test_reference_envelope_describes_this_build(tiny_esmfold2_config):
    """A config change must invalidate the file loudly, not silently.

    Without this the suite would keep comparing a new architecture against
    numbers generated for the old one, and only the tolerances would decide
    whether anyone noticed.
    """
    payload = reference_values()

    assert payload["architecture"] == "EsmFold2Model"
    assert payload["device"] == "cpu"
    assert payload["dtype"] == "float32"
    assert payload["model_seed"] == MODEL_SEED
    assert payload["forward_seed"] == FORWARD_SEED
    assert payload["sampler"] == {
        "num_loops": NUM_LOOPS,
        "num_sampling_steps": NUM_SAMPLING_STEPS,
        "num_diffusion_samples": NUM_DIFFUSION_SAMPLES,
        "chunk_size": None,
        "kernel_backend": None,
    }
    assert payload["n_distogram_tokens"] == N_DISTOGRAM_TOKENS
    assert payload["rep_atom_distance_bin_edges"] == list(REP_ATOM_DISTANCE_BIN_EDGES)
    assert payload["distance_drift_tolerance"] == DISTANCE_DRIFT_TOLERANCE

    # The stored config, for diagnosing a fingerprint mismatch, and the
    # fingerprint, which is the one-line failure.
    assert payload["config"] == tiny_esmfold2_config.to_dict()
    assert payload["config_fingerprint"] == config_fingerprint(tiny_esmfold2_config)

    # Equality, not containment: a new length regime has to be regenerated in.
    assert tuple(payload["cases"]) == ESMFOLD2_LENGTH_NAMES


def test_the_reference_model_is_the_fixture_model(tiny_esmfold2, tiny_esmfold2_config):
    """``build_reference_model`` and the ``tiny_esmfold2`` fixture must agree.

    The reference is generated outside pytest, so nothing else forces the two
    constructions to stay the same model.
    """
    ours = build_reference_model(tiny_esmfold2_config).state_dict()
    theirs = tiny_esmfold2.state_dict()
    assert set(ours) == set(theirs)
    differing = [k for k in ours if not torch.equal(ours[k], theirs[k])]
    assert not differing, differing


# ---------------------------------------------------------------------------
# The trunk: distogram logits and the pair / single representations
# ---------------------------------------------------------------------------


def trunk_outputs(
    model: EsmFold2Model, features: dict, lm_hidden_states: Tensor, *, seed: int | None
) -> tuple[Tensor, Tensor]:
    """Distogram logits and pair norms from one forward, seeded or not.

    ``seed=None`` deliberately leaves the global generator wherever the previous
    call left it, which is the only way to observe the trunk's own stochasticity.
    """
    with reference_attention(), capture_representations(model) as captured:
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            out = model(
                **features,
                lm_hidden_states=lm_hidden_states,
                num_loops=NUM_LOOPS,
                num_diffusion_samples=NUM_DIFFUSION_SAMPLES,
                num_sampling_steps=NUM_SAMPLING_STEPS,
            )
    return out["distogram_logits"].float(), pair_row_norms(captured["pair"])


def test_the_trunk_is_reproducible_only_under_a_reseed(tiny_esmfold2_config):
    """The premise of every stored number in this file, pinned from both sides.

    Zeroing ``lm_dropout`` is asserted *not* to be an alternative, because that is
    the plausible-looking fix someone will reach for: it removes one source and
    leaves the pair-state draw untouched.
    """
    config = reference_config()
    model = build_reference_model(config)
    features, lm_hidden_states = esmfold2_inputs(
        model, ESMFOLD2_SEQUENCES["chunk_edge"]
    )

    seeded = trunk_outputs(model, features, lm_hidden_states, seed=FORWARD_SEED)
    again = trunk_outputs(model, features, lm_hidden_states, seed=FORWARD_SEED)
    assert torch.equal(seeded[0], again[0])
    assert torch.equal(seeded[1], again[1])

    # Unseeded, both sources live. Observed max|Δ| against the seeded run: 2.91 on
    # the distogram logits (O(5)) and 0.27 on the pair norms (~4.7-5.5).
    drifted = trunk_outputs(model, features, lm_hidden_states, seed=None)
    assert not torch.allclose(seeded[0], drifted[0], atol=1e-3)
    assert not torch.allclose(seeded[1], drifted[1], atol=1e-3)

    # With the dropout off, only ``_init_pair_state`` remains - and it is enough
    # on its own. Observed max|Δ| between two forwards: 0.21 and 0.029.
    assert config.lm_encoder.lm_dropout == 0.25, "the fixture no longer has dropout on"
    no_dropout = reference_config()
    no_dropout.lm_encoder.lm_dropout = 0.0
    quiet = build_reference_model(no_dropout)
    first = trunk_outputs(quiet, features, lm_hidden_states, seed=None)
    second = trunk_outputs(quiet, features, lm_hidden_states, seed=None)
    assert not torch.allclose(first[0], second[0], atol=1e-3)
    assert not torch.allclose(first[1], second[1], atol=1e-3)


@pytest.mark.parametrize("name", ESMFOLD2_LENGTH_NAMES)
def test_reference_trunk_outputs(name):
    """Distogram logits and the trunk's two representations, in one forward.

    One test rather than one per quantity: xdist puts each test on its own
    worker, and a separate test per quantity would make each regime's forward run
    several times over.
    """
    case = reference_values()["cases"][name]
    measured = measured_case(name)

    assert measured["distogram_logits_shape"] == case["distogram_logits_shape"]

    # Bit-identical across reruns at a fixed thread count. Two independent
    # reduction-order changes were measured against the stored file: chunk_size
    # None vs 4 moves the logits by 1.2e-6, and 2 or 4 CPU threads instead of 12
    # moves this slice by 6.0e-7. They are O(5), so atol=1e-3 is ~800x the worse.
    torch.testing.assert_close(
        measured["distogram_logits_slice"],
        case["distogram_logits_slice"],
        atol=1e-3,
        rtol=1e-3,
    )

    # Exact, at every length. The smallest top-1 / top-2 logit margin anywhere in
    # any regime's L x L matrix is 3.2e-3, ~2600x the numerical movement above.
    # The bin counts cover the whole matrix, so this is not a claim about one row.
    for key in ("distogram_argmax_first_row", "distogram_argmax_bin_counts"):
        assert torch.equal(measured[key], case[key]), key

    # The load-bearing half: an argmax over 8 distogram bins is a coarse readout
    # of an [L, L, c_z] tensor, while these norms are not, and they move on the
    # axis-permutation and mask bugs the argmax survives. Both are O(1) - pair
    # norms sit in 4.68-5.52 and single norms in 5.98-6.67 - so one absolute
    # tolerance means the same thing everywhere. Observed max|Δ| under both
    # perturbations: 9.5e-7 on the pair norms, exactly 0 on the single norms.
    torch.testing.assert_close(
        measured["pair_norms"], case["pair_norms"], atol=1e-3, rtol=1e-4
    )
    torch.testing.assert_close(
        measured["single_norms"], case["single_norms"], atol=1e-3, rtol=1e-4
    )


# ---------------------------------------------------------------------------
# The sampler, on rigid-invariant quantities only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ESMFOLD2_LENGTH_NAMES)
def test_reference_sampled_structure(name):
    """The sampler's rigid-invariant geometry, plus pLDDT and pTM.

    pLDDT is exactly 0.5 at every position here, and that is a real assertion
    rather than a vacuous one: ``ConfidenceHead.plddt_weight`` is zero-initialised
    and no ``_init_weights`` pass touches it, so the per-atom bin distribution is
    exactly uniform and its mean is the midpoint of the [0, 1] bin range. What the
    value pins is the arithmetic around it - the bin range, and the atom -> token
    scatter dividing by the valid atom count rather than the padded one. It does
    not pin what the head reads; that needs trained weights.
    """
    case = reference_values()["cases"][name]
    measured = measured_case(name)

    # Padded atoms are excluded from the radii, so the count is part of the
    # contract: an atom-padding bug that leaked 32 zero rows into the centroid
    # would otherwise only show up as a diffuse shift in every radius.
    assert measured["n_atoms_valid"] == case["n_atoms_valid"]
    assert measured["n_atoms_padded"] == case["n_atoms_padded"]

    # Radii run 0.37-25.4 Å. Bit-identical across reruns; observed max|Δ| 5.9e-5 Å
    # between chunk_size None and 4 - the sampler amplifies the trunk's 9.5e-7 by
    # ~60x over its two ODE steps and two Kabsch alignments. atol=1e-2 Å is ~170x
    # that, and two orders of magnitude below the Ångström-scale difference any
    # real regression produces (2.3 Å mean between two sampler seeds). rtol=0:
    # these are Ångströms, so a relative term would swamp the absolute bound at
    # the outer atoms.
    torch.testing.assert_close(
        measured["atom_radii"], case["atom_radii"], atol=1e-2, rtol=0
    )

    assert measured["n_rep_atom_pairs"] == case["n_rep_atom_pairs"]
    # Identical counts at every length under both perturbations, so there is no
    # observed spread to widen from; the budget is the stored census instead,
    # which is an upper bound rather than an observation - see
    # ``n_distances_near_bin_edge``. Budgets run 3.6% of the pair count at
    # chunk_edge up to 9.1% at tiny (66 pairs is a coarse denominator), so the 20%
    # ceiling has ~2x headroom and still fails if the census balloons into
    # vacuity.
    budget = 2 * case["n_distances_near_bin_edge"]
    drift = int(
        (measured["rep_atom_distance_histogram"] - case["rep_atom_distance_histogram"])
        .abs()
        .sum()
    )
    assert drift <= budget, f"{drift} > {budget}"
    assert budget <= 0.2 * case["n_rep_atom_pairs"]

    # Exactly 0.0 at every length under both perturbations, which the
    # exactly-uniform bin distribution explains; 1e-6 is a floor, not a spread.
    torch.testing.assert_close(measured["plddt"], case["plddt"], atol=1e-6, rtol=0)

    # pTM spans 6.8e-4 (tiny) to 0.307 (long), so the bound has to work at both
    # ends. Observed max|Δ| 1.5e-8. abs=1e-6 binds at the tiny end, rel=1e-3
    # above ~1e-3.
    assert measured["ptm"] == pytest.approx(case["ptm"], rel=1e-3, abs=1e-6)


def test_the_stored_sample_is_the_seeded_one():
    """The reference would also pass against a sampler that ignored its noise.

    ``sample`` draws the initial x and its per-step augmentation from the global
    generator, so a reference taken at one seed only means something if a
    different seed produces a different structure.
    """
    config = reference_config()
    model = build_reference_model(config)
    features, lm_hidden_states = esmfold2_inputs(
        model, ESMFOLD2_SEQUENCES["chunk_edge"]
    )

    def sample(seed: int) -> Tensor:
        with reference_attention():
            torch.manual_seed(seed)
            with torch.no_grad():
                out = model(
                    **features,
                    lm_hidden_states=lm_hidden_states,
                    num_loops=NUM_LOOPS,
                    num_diffusion_samples=NUM_DIFFUSION_SAMPLES,
                    num_sampling_steps=NUM_SAMPLING_STEPS,
                )
        return atom_radii(
            out["sample_atom_coords"][0], features["atom_attention_mask"][0]
        )

    seeded = sample(FORWARD_SEED)
    assert torch.equal(seeded, sample(FORWARD_SEED))
    # On the per-atom radii, seed 0 against seeds 1/2/3: mean|Δ| ~2.3 Å and
    # max|Δ| 9.2-10.8 Å, against the 1e-2 Å the stored reference is checked at.
    assert not torch.allclose(seeded, sample(FORWARD_SEED + 1), atol=1e-2)
