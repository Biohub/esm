"""ESMFold2 public-API tests — ``fold``, ``infer_protein``, ``infer_protein_as_pdb``.

All cases run on CPU against a tiny randomly-initialised model with no PLM
backbone attached, so the assertions are about plumbing rather than structure
quality.

The ``sampler override`` group pins that ``fold``'s ``noise_scale`` /
``step_scale`` / ``max_inference_sigma`` reach the sampler:
``EsmFold2Model.forward`` has to declare each one rather than let a trailing
``**kwargs`` swallow it and make the override a silent no-op.
"""

import pytest
import torch

from esm.models.esmfold2 import layers as _layers
from esm.models.esmfold2.prepare_input import (
    prepare_esmfold2_input,
)
from esm.models.esmfold2.processor import ESMFold2InputBuilder
from esm.models.esmfold2.protein_utils import (
    OUTPUT_TO_PDB_FEATURE_KEYS,
    prepare_protein_features,
)
from esm.models.esmfold2.types import (
    ProteinInput,
    StructurePredictionInput,
)
from esm.tests.conftest import (
    ESMFOLD2_SEQUENCES,
    esmfold2_inputs,
)
from esm.utils.structure.molecular_complex import (
    MolecularComplexResult,
)

TINY_SEQUENCE = ESMFOLD2_SEQUENCES["tiny"]

# A forward is O(L^2); these keep the module fast.
FAST_FOLD = dict(num_loops=1, num_sampling_steps=2, num_diffusion_samples=1)

# The Karras schedule gives ``sigma_t > gamma_min`` — and therefore a non-zero
# churn step, which is the only place ``noise_scale`` enters — only from the
# fourth step down. At the fast tier's two steps every gamma is zero and
# ``noise_scale`` is a genuine no-op, so the override cases use four.
SAMPLER_OVERRIDE_STEPS = 4


#: ``ESMFold2InputBuilder.fold`` forwards ``early_exit`` and
#: ``msa_subsample_at_inference`` to ``EsmFold2Model.forward``, which #13995
#: deleted from the signature, so every ``fold`` call raises TypeError. Recorded
#: rather than fixed: this suite only adds tests, it does not change the
#: implementation it tests. Strict, so whoever drops those two arguments from the
#: ``fold`` call gets a failure telling them to delete this marker, rather than a
#: test that silently keeps passing.
FOLD_IS_BROKEN = pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason=(
        "fold() forwards early_exit and msa_subsample_at_inference, which "
        "EsmFold2Model.forward no longer accepts (#13995)"
    ),
)


@pytest.fixture(autouse=True)
def _force_reference_attention(monkeypatch):
    """flash-attn imports on a CPU box but needs a CUDA runtime to run."""
    if not torch.cuda.is_available():
        monkeypatch.setattr(_layers, "FLASH_ATTN_AVAILABLE", False)


@pytest.fixture(scope="session")
def builder(ccd_pickle) -> ESMFold2InputBuilder:
    """The real input builder.

    Constructing it loads the CCD dictionary — from ``ESMCFOLD_CCD_PATH`` when
    set, else one cached Hub download. Session-scoped because that load is the
    dominant cost of every case here; the builder itself is stateless.
    """
    return ESMFold2InputBuilder()


def protein_input(sequence: str = TINY_SEQUENCE) -> StructurePredictionInput:
    return StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])


def record_sample_kwargs(model) -> list[dict]:
    """Capture the keyword arguments of every ``DiffusionStructureHead.sample``."""
    calls: list[dict] = []
    real = model.structure_head.sample

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    model.structure_head.sample = spy
    return calls


# ---------------------------------------------------------------------------
# forward must not swallow keyword arguments
# ---------------------------------------------------------------------------


def test_forward_rejects_an_unknown_keyword(tiny_esmfold2):
    """A misspelled override has to fail rather than silently do nothing.

    ``forward`` takes 30-odd feature arguments, so a catch-all ``**kwargs`` on
    it will keep absorbing typos and renamed knobs; this is the assertion that
    stops the next one.
    """
    features, lm_hidden_states = esmfold2_inputs(tiny_esmfold2)
    with pytest.raises(TypeError, match="nosie_scale"):
        with torch.no_grad():
            tiny_esmfold2(
                **features,
                lm_hidden_states=lm_hidden_states,
                nosie_scale=2.0,
                **FAST_FOLD,
            )


def test_forward_accepts_the_featurizer_key_set(tiny_esmfold2, ccd_pickle):
    """``prepare_esmfold2_input`` emits training-time keys ``forward`` ignores.

    ``pocket_feature`` / ``gt_coords`` / ``is_resolved`` / ``frames_idx`` have no
    consumer at inference, so ``forward`` has to keep tolerating them: this is
    why the fix for the swallowed-kwargs bug validates against a known-unused
    set instead of dropping the catch-all outright.
    """
    features, _ = prepare_esmfold2_input(protein_input())
    batched = {k: v[None] for k, v in features.items() if isinstance(v, torch.Tensor)}
    assert {"pocket_feature", "gt_coords", "is_resolved", "frames_idx"} <= set(batched)

    torch.manual_seed(0)
    with torch.no_grad():
        out = tiny_esmfold2(**batched, **FAST_FOLD)
    assert torch.isfinite(out["sample_atom_coords"]).all()


@pytest.mark.parametrize(
    "override,expected",
    [
        ({"noise_scale": 2.0}, {"noise_scale": 2.0}),
        ({"step_scale": 2.5}, {"step_scale": 2.5}),
        ({"max_inference_sigma": 100.0}, {"max_inference_sigma": 100.0}),
    ],
    ids=["noise_scale", "step_scale", "max_inference_sigma"],
)
@FOLD_IS_BROKEN
def test_fold_forwards_sampler_overrides_to_the_sampler(
    tiny_esmfold2, builder, override, expected
):
    """Each override reaches ``DiffusionStructureHead.sample`` with its value."""
    calls = record_sample_kwargs(tiny_esmfold2)

    builder.fold(
        tiny_esmfold2,
        protein_input(),
        seed=0,
        lm_dropout=None,
        **{**FAST_FOLD, **override},
    )

    assert len(calls) == 1
    for name, value in expected.items():
        assert calls[0][name] == value


@pytest.mark.parametrize(
    "override",
    [{"noise_scale": 2.0}, {"step_scale": 2.5}, {"max_inference_sigma": 100.0}],
    ids=["noise_scale", "step_scale", "max_inference_sigma"],
)
@FOLD_IS_BROKEN
def test_fold_sampler_overrides_change_the_structure(tiny_esmfold2, builder, override):
    """The overrides are not just delivered, they move the coordinates.

    Delivery alone would still pass if the sampler ignored them, and the
    coordinate check alone would pass on any source of extra randomness, so both
    assertions are needed. The same ``seed`` makes the two folds differ in
    nothing but the override.
    """
    settings = dict(
        FAST_FOLD, num_sampling_steps=SAMPLER_OVERRIDE_STEPS, seed=0, lm_dropout=None
    )
    baseline = builder.fold(tiny_esmfold2, protein_input(), **settings)
    overridden = builder.fold(tiny_esmfold2, protein_input(), **settings, **override)

    assert isinstance(baseline, MolecularComplexResult)
    assert isinstance(overridden, MolecularComplexResult)
    before = torch.as_tensor(baseline.complex.atom_positions)
    after = torch.as_tensor(overridden.complex.atom_positions)
    assert before.shape == after.shape
    assert not torch.allclose(before, after, atol=1e-3)


@FOLD_IS_BROKEN
def test_fold_is_reproducible_under_a_seed(tiny_esmfold2, builder):
    """Two folds with the same seed agree bit-for-bit.

    This is what makes the override comparison above meaningful: without it, a
    difference in coordinates would only show that the sampler is stochastic.
    """
    settings = dict(FAST_FOLD, seed=7, lm_dropout=None)
    first = builder.fold(tiny_esmfold2, protein_input(), **settings)
    second = builder.fold(tiny_esmfold2, protein_input(), **settings)
    torch.testing.assert_close(
        torch.as_tensor(first.complex.atom_positions),
        torch.as_tensor(second.complex.atom_positions),
        atol=0,
        rtol=0,
    )


# ---------------------------------------------------------------------------
# The fold() return contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_diffusion_samples", [1, 2])
@FOLD_IS_BROKEN
def test_fold_return_shape_follows_the_sample_count(
    tiny_esmfold2, builder, num_diffusion_samples
):
    """One sample returns a result; more than one returns a list of that length."""
    result = builder.fold(
        tiny_esmfold2,
        protein_input(),
        seed=0,
        lm_dropout=None,
        num_loops=1,
        num_sampling_steps=2,
        num_diffusion_samples=num_diffusion_samples,
    )

    if num_diffusion_samples == 1:
        assert isinstance(result, MolecularComplexResult)
        results = [result]
    else:
        assert isinstance(result, list)
        assert len(result) == num_diffusion_samples
        results = result

    n_residues = len(TINY_SEQUENCE)
    for item in results:
        assert item.plddt.shape == (n_residues,)
        assert 0.0 <= float(item.plddt.min()) and float(item.plddt.max()) <= 1.0
        assert item.ptm is not None and 0.0 <= item.ptm <= 1.0
        assert item.complex.atom_positions.shape[-1] == 3

    if num_diffusion_samples > 1:
        # Independent noise per sample: identical coordinates would mean the
        # sample axis was broadcast rather than drawn.
        first, second = (
            torch.as_tensor(item.complex.atom_positions) for item in results[:2]
        )
        assert not torch.allclose(first, second, atol=1e-3)


def test_fold_rejects_an_unknown_sampler_override(tiny_esmfold2, builder):
    """``fold`` has no catch-all of its own, so a typo stops at its signature."""
    with pytest.raises(TypeError):
        builder.fold(
            tiny_esmfold2, protein_input(), seed=0, noize_scale=2.0, **FAST_FOLD
        )


# ---------------------------------------------------------------------------
# The convenience wrappers on the model itself
# ---------------------------------------------------------------------------


def test_infer_protein_reattaches_the_pdb_feature_keys(tiny_esmfold2):
    """``forward`` does not echo its inputs, so ``infer_protein`` puts them back."""
    torch.manual_seed(0)
    output = tiny_esmfold2.infer_protein(TINY_SEQUENCE, **FAST_FOLD)
    features = prepare_protein_features(TINY_SEQUENCE)

    for key in OUTPUT_TO_PDB_FEATURE_KEYS:
        assert key in output, key
        torch.testing.assert_close(output[key], features[key], atol=0, rtol=0)

    n_atoms = features["atom_attention_mask"].shape[-1]
    assert output["sample_atom_coords"].shape == (1, n_atoms, 3)
    assert output["plddt"].shape == (1, len(TINY_SEQUENCE))
    assert torch.isfinite(output["sample_atom_coords"]).all()


def test_infer_protein_as_pdb_renders_the_same_fold(tiny_esmfold2):
    """The one-liner is exactly ``output_to_pdb(infer_protein(...))``.

    Seeded on both sides, so the two PDB strings must be character-identical;
    anything the wrapper adds or drops shows up immediately.
    """
    torch.manual_seed(0)
    via_wrapper = tiny_esmfold2.infer_protein_as_pdb(TINY_SEQUENCE, **FAST_FOLD)
    torch.manual_seed(0)
    by_hand = tiny_esmfold2.output_to_pdb(
        tiny_esmfold2.infer_protein(TINY_SEQUENCE, **FAST_FOLD)
    )
    assert via_wrapper == by_hand

    atoms = [line for line in via_wrapper.splitlines() if line.startswith("ATOM")]
    n_real_atoms = int(
        prepare_protein_features(TINY_SEQUENCE)["atom_attention_mask"].sum()
    )
    assert len(atoms) == n_real_atoms


def test_infer_protein_rejects_an_unknown_keyword(tiny_esmfold2):
    """``infer_protein``'s ``**forward_kwargs`` inherits ``forward``'s strictness."""
    with pytest.raises(TypeError, match="num_recyles"):
        tiny_esmfold2.infer_protein(TINY_SEQUENCE, num_recyles=1, **FAST_FOLD)


def test_infer_protein_rejects_an_empty_sequence(tiny_esmfold2):
    with pytest.raises(ValueError, match="non-empty"):
        tiny_esmfold2.infer_protein("", **FAST_FOLD)


# ---------------------------------------------------------------------------
# The experimental architecture answers to the same contract
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_experimental(tiny_esmfold2_config):
    """The experimental architecture at the same tiny widths."""
    from dataclasses import asdict

    from esm.models.esmfold2 import (
        EsmFold2Config,
        EsmFold2ExperimentalModel,
    )

    fields = {
        key: asdict(value) if hasattr(value, "__dataclass_fields__") else value
        for key, value in vars(tiny_esmfold2_config).items()
    }
    torch.manual_seed(0)
    model = EsmFold2ExperimentalModel(
        EsmFold2Config(**{**fields, "type": "experimental"})
    ).eval()
    model.set_chunk_size(None)
    return model


@pytest.mark.parametrize(
    "override,expected",
    [
        ({"noise_scale": 2.0}, {"noise_scale": 2.0}),
        ({"step_scale": 2.5}, {"step_scale": 2.5}),
        ({"max_inference_sigma": 100.0}, {"max_inference_sigma": 100.0}),
    ],
    ids=["noise_scale", "step_scale", "max_inference_sigma"],
)
def test_experimental_forward_forwards_sampler_overrides(
    tiny_experimental, override, expected
):
    """The experimental model declares the overrides too, not just the release one.

    All three used to land in its ``**kwargs`` and be dropped, exactly as the
    release model's were.
    """
    features, lm_hidden_states = esmfold2_inputs(tiny_experimental, TINY_SEQUENCE)
    calls = record_sample_kwargs(tiny_experimental)

    with torch.no_grad():
        tiny_experimental(
            **features, lm_hidden_states=lm_hidden_states, **{**FAST_FOLD, **override}
        )

    assert len(calls) == 1
    for name, value in expected.items():
        assert calls[0][name] == value


def test_experimental_forward_rejects_an_unknown_keyword(tiny_experimental):
    features, lm_hidden_states = esmfold2_inputs(tiny_experimental, TINY_SEQUENCE)
    with pytest.raises(TypeError, match="num_recyles"):
        with torch.no_grad():
            tiny_experimental(
                **features,
                lm_hidden_states=lm_hidden_states,
                num_recyles=1,
                **FAST_FOLD,
            )
