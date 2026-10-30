"""ESMFold2 public-API tests — ``fold``, ``infer_protein``, ``infer_protein_as_pdb``.

All cases run on CPU against a tiny randomly-initialised model, so the
assertions are about plumbing rather than structure quality. ``fold`` refuses a
model with no PLM backbone, so the ``fold`` cases attach a tiny random ESMC
(``tiny_esmfold2_lm``); raw ``forward`` cases pass synthetic LM states instead.

The ``sampler override`` group pins that ``fold``'s ``noise_scale`` /
``step_scale`` / ``max_inference_sigma`` reach the sampler:
``EsmFold2Model.forward`` has to declare each one rather than let a trailing
``**kwargs`` swallow it and make the override a silent no-op.
"""

import pytest
import torch

from esm.models.esmfold2 import layers as _layers
from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
from esm.models.esmfold2.processor import ESMFold2InputBuilder, _lm_dropout_context
from esm.models.esmfold2.protein_utils import (
    OUTPUT_TO_PDB_FEATURE_KEYS,
    prepare_protein_features,
)
from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput
from esm.utils.structure.molecular_complex import MolecularComplexResult
from tests.conftest import ESMFOLD2_SEQUENCES, _synthetic_lm_states, esmfold2_inputs

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


@pytest.fixture
def tiny_esmfold2_lm(tiny_esmfold2, tiny_esmc):
    """``tiny_esmfold2`` with a tiny random ESMC attached as its backbone.

    The tiny ESMC's 2 layers x 32 wide match the tiny config's ``lm_num_layers``
    / ``lm_d_model``, so the shim accepts its hidden states unchanged.
    """
    tiny_esmfold2.esmc = tiny_esmc
    return tiny_esmfold2


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
def test_fold_forwards_sampler_overrides_to_the_sampler(
    tiny_esmfold2_lm, builder, override, expected
):
    """Each override reaches ``DiffusionStructureHead.sample`` with its value."""
    calls = record_sample_kwargs(tiny_esmfold2_lm)

    builder.fold(
        tiny_esmfold2_lm,
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
def test_fold_sampler_overrides_change_the_structure(
    tiny_esmfold2_lm, builder, override
):
    """The overrides are not just delivered, they move the coordinates.

    Delivery alone would still pass if the sampler ignored them, and the
    coordinate check alone would pass on any source of extra randomness, so both
    assertions are needed. The same ``seed`` makes the two folds differ in
    nothing but the override.
    """
    settings = dict(
        FAST_FOLD, num_sampling_steps=SAMPLER_OVERRIDE_STEPS, seed=0, lm_dropout=None
    )
    baseline = builder.fold(tiny_esmfold2_lm, protein_input(), **settings)
    overridden = builder.fold(tiny_esmfold2_lm, protein_input(), **settings, **override)

    assert isinstance(baseline, MolecularComplexResult)
    assert isinstance(overridden, MolecularComplexResult)
    before = torch.as_tensor(baseline.complex.atom_positions)
    after = torch.as_tensor(overridden.complex.atom_positions)
    assert before.shape == after.shape
    assert not torch.allclose(before, after, atol=1e-3)


def test_fold_is_reproducible_under_a_seed(tiny_esmfold2_lm, builder):
    """Two folds with the same seed agree bit-for-bit.

    This is what makes the override comparison above meaningful: without it, a
    difference in coordinates would only show that the sampler is stochastic.
    """
    settings = dict(FAST_FOLD, seed=7, lm_dropout=None)
    first = builder.fold(tiny_esmfold2_lm, protein_input(), **settings)
    second = builder.fold(tiny_esmfold2_lm, protein_input(), **settings)
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
def test_fold_return_shape_follows_the_sample_count(
    tiny_esmfold2_lm, builder, num_diffusion_samples
):
    """One sample returns a result; more than one returns a list of that length."""
    result = builder.fold(
        tiny_esmfold2_lm,
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
# The LM pathway: fold() cannot drop it, forward() still can
# ---------------------------------------------------------------------------


LM_FEATURE_KEYS = (
    "input_ids",
    "asym_id",
    "residue_index",
    "mol_type",
    "token_attention_mask",
)


def lm_inputs(features: dict) -> dict:
    """The feature tensors ``compute_lm_hidden_states`` takes, by name."""
    return {key: features[key] for key in LM_FEATURE_KEYS}


def record_shim_inputs(model) -> list[torch.Tensor]:
    """Capture the hidden states every call to the LM shim receives."""
    seen: list[torch.Tensor] = []
    model.language_model.register_forward_hook(
        lambda module, args, out: seen.append(args[0])
    )
    return seen


def synthetic_states_for(model, builder, input) -> torch.Tensor:
    features, _ = builder.prepare_input(input, seed=0, device=model.device)
    return _synthetic_lm_states(model, features, seed=0)


@pytest.mark.parametrize("fixture", ["tiny_esmfold2", "tiny_experimental"])
def test_fold_refuses_a_model_without_an_lm_backbone(fixture, request, builder):
    model = request.getfixturevalue(fixture)
    assert model.esmc is None
    calls = record_sample_kwargs(model)
    with pytest.raises(ValueError, match="no ESMC backbone"):
        builder.fold(model, protein_input(), seed=0, **FAST_FOLD)
    assert calls == []


def test_fold_forwards_supplied_lm_hidden_states(tiny_esmfold2, builder):
    states = synthetic_states_for(tiny_esmfold2, builder, protein_input())
    seen = record_shim_inputs(tiny_esmfold2)
    result = builder.fold(
        tiny_esmfold2,
        protein_input(),
        seed=0,
        lm_dropout=None,
        lm_hidden_states=states,
        **FAST_FOLD,
    )
    assert isinstance(result, MolecularComplexResult)
    assert len(seen) == 1
    assert torch.equal(seen[0], states)


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16, torch.float16])
def test_fold_casts_supplied_states_to_the_shim_dtype(tiny_esmfold2, builder, dtype):
    states = synthetic_states_for(tiny_esmfold2, builder, protein_input())
    seen = record_shim_inputs(tiny_esmfold2)
    builder.fold(
        tiny_esmfold2,
        protein_input(),
        seed=0,
        lm_hidden_states=states.to(dtype),
        **FAST_FOLD,
    )
    assert seen[0].dtype == tiny_esmfold2.language_model.base_z_linear[0].weight.dtype


def _wider(states: torch.Tensor, axis: int) -> torch.Tensor:
    shape = list(states.shape)
    shape[axis] += 1
    return torch.zeros(shape)


@pytest.mark.parametrize(
    "mutate,extra,match",
    [
        (lambda s: s, {"lm_mask_pct": 0.1}, "lm_mask_pct"),
        (lambda s: s[:, :-1], {}, "does not match the input"),
        (lambda s: _wider(s, 2), {}, "LM shim"),
        (lambda s: _wider(s, 3), {}, "LM shim"),
        (lambda s: s.long(), {}, "floating point"),
        (lambda s: s.to("meta"), {}, "is on meta"),
    ],
    ids=["lm_mask_pct", "tokens", "layers", "width", "dtype", "device"],
)
def test_fold_rejects_unusable_lm_hidden_states(
    tiny_esmfold2, builder, mutate, extra, match
):
    states = mutate(synthetic_states_for(tiny_esmfold2, builder, protein_input()))
    with pytest.raises(ValueError, match=match):
        builder.fold(
            tiny_esmfold2,
            protein_input(),
            seed=0,
            lm_hidden_states=states,
            **extra,
            **FAST_FOLD,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_standalone_lm_pass_on_cuda_offloads_and_feeds_fold(tiny_esmfold2_lm, builder):
    """With offloading on, ESMC is back on CPU after the call and its states fold."""
    model = tiny_esmfold2_lm.cuda()
    model._offload_esmc = True
    features, _ = builder.prepare_input(protein_input(), seed=0, device=model.device)

    states = model.compute_lm_hidden_states(**lm_inputs(features))

    assert states.is_cuda
    assert all(p.device.type == "cpu" for p in model.esmc.parameters())

    result = builder.fold(
        model,
        protein_input(),
        seed=0,
        lm_dropout=None,
        lm_hidden_states=states,
        **FAST_FOLD,
    )
    assert isinstance(result, MolecularComplexResult)
    assert torch.isfinite(torch.as_tensor(result.complex.atom_positions)).all()
    assert all(p.device.type == "cpu" for p in model.esmc.parameters())


def test_forward_without_an_lm_backbone_skips_the_lm_pathway(tiny_esmfold2, builder):
    """The low-level entry point keeps the no-LM path, for intentional ablations."""
    features, _ = builder.prepare_input(protein_input(), seed=0)
    seen = record_shim_inputs(tiny_esmfold2)

    torch.manual_seed(0)
    with torch.no_grad():
        output = tiny_esmfold2(**features, **FAST_FOLD)

    assert seen == []
    assert torch.isfinite(output["sample_atom_coords"]).all()


def test_compute_lm_hidden_states_is_what_forward_computes(tiny_esmfold2_lm, builder):
    """Folding with the public method's states reproduces the ordinary fold."""
    settings = dict(FAST_FOLD, seed=0, lm_dropout=None)
    features, _ = builder.prepare_input(protein_input(), seed=0)
    seen = record_shim_inputs(tiny_esmfold2_lm)

    states = tiny_esmfold2_lm.compute_lm_hidden_states(**lm_inputs(features))
    config = tiny_esmfold2_lm.config
    assert states.shape == (
        *features["token_attention_mask"].shape,
        config.lm_num_layers + 1,
        config.lm_d_model,
    )

    own = builder.fold(tiny_esmfold2_lm, protein_input(), **settings)
    supplied = builder.fold(
        tiny_esmfold2_lm, protein_input(), lm_hidden_states=states, **settings
    )

    assert len(seen) == 2
    torch.testing.assert_close(seen[0], seen[1], atol=0, rtol=0)
    torch.testing.assert_close(
        torch.as_tensor(own.complex.atom_positions),
        torch.as_tensor(supplied.complex.atom_positions),
        atol=0,
        rtol=0,
    )


def test_experimental_compute_lm_hidden_states_is_what_forward_computes(
    tiny_experimental, tiny_esmc, builder
):
    tiny_experimental.esmc = tiny_esmc
    features, _ = builder.prepare_input(protein_input(), seed=0)
    seen = record_shim_inputs(tiny_experimental)

    states = tiny_experimental.compute_lm_hidden_states(**lm_inputs(features))
    torch.manual_seed(0)
    with torch.no_grad():
        tiny_experimental(**features, **FAST_FOLD)

    assert len(seen) == 1
    torch.testing.assert_close(seen[0], states, atol=0, rtol=0)


@pytest.mark.parametrize("fixture", ["tiny_esmfold2", "tiny_experimental"])
def test_compute_lm_hidden_states_needs_a_backbone(fixture, request, builder):
    model = request.getfixturevalue(fixture)
    features, _ = builder.prepare_input(protein_input(), seed=0)
    with pytest.raises(ValueError, match="needs an ESMC backbone"):
        model.compute_lm_hidden_states(**lm_inputs(features))


# ---------------------------------------------------------------------------
# compute_lm_hidden_states: offload, fp8 and defaults, through a mock LM pass
# ---------------------------------------------------------------------------


class _PlacementRecorder(torch.nn.Module):
    """Stands in for ESMC; records every ``.to(device)`` instead of moving."""

    def __init__(self):
        super().__init__()
        self.moves: list[torch.device] = []

    def to(self, *args, **kwargs):
        self.moves.append(torch.device(args[0] if args else kwargs["device"]))
        return self


def _dummy_lm_inputs() -> dict:
    return {key: torch.zeros(1, 3, dtype=torch.long) for key in LM_FEATURE_KEYS}


@pytest.fixture
def recorded_lm_pass(monkeypatch):
    """Swap ``model.compute_lm_hidden_states`` (the layers helper) for a recorder."""
    from esm.models.esmfold2 import model as model_module

    calls: list[dict] = []

    def fake(esmc, *args, **kwargs):
        calls.append(
            {"moves": list(getattr(esmc, "moves", [])), "args": args, **kwargs}
        )
        return torch.zeros(1)

    monkeypatch.setattr(model_module, "compute_lm_hidden_states", fake)
    return calls


@pytest.fixture
def on_an_accelerator(monkeypatch, tiny_esmfold2):
    """Make ``model.device`` report a non-CPU device, so offload moves show."""
    from esm.models.esmfold2 import EsmFold2Model

    accelerator = torch.device("meta")
    monkeypatch.setattr(EsmFold2Model, "device", property(lambda self: accelerator))
    return accelerator


def test_standalone_lm_pass_offloads_the_backbone_again(
    tiny_esmfold2, recorded_lm_pass, on_an_accelerator
):
    """Resident on the model device for the pass, back on CPU after it."""
    esmc = _PlacementRecorder()
    tiny_esmfold2.esmc = esmc
    tiny_esmfold2._offload_esmc = True

    tiny_esmfold2.compute_lm_hidden_states(**_dummy_lm_inputs())

    (call,) = recorded_lm_pass
    assert call["moves"] == [on_an_accelerator]
    assert esmc.moves == [on_an_accelerator, torch.device("cpu")]


def test_standalone_lm_pass_offloads_even_when_the_pass_fails(
    tiny_esmfold2, monkeypatch, on_an_accelerator
):
    from esm.models.esmfold2 import model as model_module

    def boom(*args, **kwargs):
        raise RuntimeError("LM pass blew up")

    monkeypatch.setattr(model_module, "compute_lm_hidden_states", boom)
    esmc = _PlacementRecorder()
    tiny_esmfold2.esmc = esmc
    tiny_esmfold2._offload_esmc = True

    with pytest.raises(RuntimeError, match="blew up"):
        tiny_esmfold2.compute_lm_hidden_states(**_dummy_lm_inputs())
    assert esmc.moves == [on_an_accelerator, torch.device("cpu")]


def test_standalone_lm_pass_leaves_a_resident_backbone_alone(
    tiny_esmfold2, recorded_lm_pass, on_an_accelerator
):
    esmc = _PlacementRecorder()
    tiny_esmfold2.esmc = esmc
    assert tiny_esmfold2._offload_esmc is False

    tiny_esmfold2.compute_lm_hidden_states(**_dummy_lm_inputs())

    assert esmc.moves == []


@pytest.mark.parametrize("fp8,pad", [(True, 8), (False, None)], ids=["fp8", "bf16"])
def test_lm_pass_applies_the_fp8_context_and_padding(
    tiny_esmfold2, recorded_lm_pass, monkeypatch, fp8, pad
):
    from contextlib import contextmanager

    from esm.models.esmfold2 import model as model_module

    entered: list[bool] = []
    inside = {"now": False}

    @contextmanager
    def recording_context(fp8_flag):
        entered.append(fp8_flag)
        inside["now"] = True
        try:
            yield
        finally:
            inside["now"] = False

    real_fake = model_module.compute_lm_hidden_states

    def fake(*args, **kwargs):
        assert inside["now"], "LM pass ran outside the precision context"
        return real_fake(*args, **kwargs)

    monkeypatch.setattr(model_module, "_lm_precision_context", recording_context)
    monkeypatch.setattr(model_module, "compute_lm_hidden_states", fake)
    tiny_esmfold2.esmc = _PlacementRecorder()
    tiny_esmfold2._esmc_fp8 = fp8

    tiny_esmfold2.compute_lm_hidden_states(**_dummy_lm_inputs())

    assert entered == [fp8]
    (call,) = recorded_lm_pass
    assert call["pad_to_multiple"] == pad


@pytest.mark.parametrize("requested,expected", [(None, 0.37), (0.0, 0.0), (0.2, 0.2)])
def test_lm_mask_pct_defaults_to_the_config(
    tiny_esmfold2, recorded_lm_pass, monkeypatch, requested, expected
):
    """``None`` means the checkpoint's value; an explicit 0.0 is kept."""
    tiny_esmfold2.esmc = _PlacementRecorder()
    # The tiny config is session-scoped; monkeypatch puts the value back.
    monkeypatch.setattr(tiny_esmfold2.config, "lm_mask_pct", 0.37)

    tiny_esmfold2.compute_lm_hidden_states(**_dummy_lm_inputs(), lm_mask_pct=requested)

    (call,) = recorded_lm_pass
    assert call["lm_mask_pct"] == expected


def test_experimental_lm_mask_pct_defaults_to_the_config(
    tiny_experimental, monkeypatch
):
    from esm.models.esmfold2 import experimental as experimental_module

    seen: list[float] = []
    monkeypatch.setattr(
        experimental_module,
        "compute_lm_hidden_states",
        lambda *args, lm_mask_pct, **kwargs: seen.append(lm_mask_pct),
    )
    tiny_experimental.esmc = _PlacementRecorder()
    monkeypatch.setattr(tiny_experimental.config, "lm_mask_pct", 0.37)

    tiny_experimental.compute_lm_hidden_states(**_dummy_lm_inputs())

    assert seen == [0.37]


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

    from esm.models.esmfold2 import EsmFold2Config, EsmFold2ExperimentalModel

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


# ---------------------------------------------------------------------------
# include_embeddings
# ---------------------------------------------------------------------------


def _pair_width(config) -> int:
    return config.pairwise_hidden_size


def test_fold_omits_embeddings_by_default(tiny_esmfold2_lm, builder):
    """The pair export is opt-in: it costs an L x L reduction and an L x D tensor."""
    result = builder.fold(tiny_esmfold2_lm, protein_input(), seed=0, **FAST_FOLD)

    assert result.output_embedding_pair_pooled is None
    assert result.output_embedding_sequence is None


def test_fold_include_embeddings_returns_the_pooled_pair(
    tiny_esmfold2_lm, tiny_esmfold2_config, builder
):
    """Shape, dtype and device match what the SDK hands back from Forge."""
    result = builder.fold(
        tiny_esmfold2_lm, protein_input(), seed=0, include_embeddings=True, **FAST_FOLD
    )

    pooled = result.output_embedding_pair_pooled
    assert pooled is not None
    assert pooled.shape == (len(TINY_SEQUENCE), _pair_width(tiny_esmfold2_config))
    assert pooled.dtype == torch.float32
    assert pooled.device.type == "cpu"
    assert torch.isfinite(pooled).all()

    assert result.output_embedding_sequence is None


def test_pooled_pair_is_the_post_coda_pair_averaged_over_the_first_axis(tiny_esmfold2):
    """Pin the reduction axis against the tensor it is supposed to summarise."""
    captured: dict[str, torch.Tensor] = {}
    handle = tiny_esmfold2.parcae_coda.register_forward_hook(
        lambda module, args, out: captured.__setitem__("z", out.detach().float())
    )
    try:
        features, lm_hidden_states = esmfold2_inputs(tiny_esmfold2, TINY_SEQUENCE)
        with torch.no_grad():
            output = tiny_esmfold2(
                **features,
                lm_hidden_states=lm_hidden_states,
                include_embeddings=True,
                **FAST_FOLD,
            )
    finally:
        handle.remove()

    z = captured["z"]
    pooled = output["output_embedding_pair_pooled"]

    assert torch.allclose(pooled, z.mean(dim=1), atol=1e-6)
    assert not torch.allclose(pooled, z.mean(dim=2), atol=1e-4)


def test_pooled_pair_is_shared_across_diffusion_samples(tiny_esmfold2_lm, builder):
    """The trunk runs once, so every sample carries the same embedding."""
    results = builder.fold(
        tiny_esmfold2_lm,
        protein_input(),
        seed=0,
        num_loops=1,
        num_sampling_steps=2,
        num_diffusion_samples=2,
        include_embeddings=True,
    )

    assert isinstance(results, list) and len(results) == 2
    first, second = (item.output_embedding_pair_pooled for item in results)
    assert first is not None and second is not None
    assert torch.equal(first, second)


def test_experimental_forward_supports_include_embeddings(
    tiny_experimental, tiny_esmfold2_config
):
    """``fold`` takes either architecture, so the flag cannot be release-only."""
    features, lm_hidden_states = esmfold2_inputs(tiny_experimental, TINY_SEQUENCE)

    with torch.no_grad():
        output = tiny_experimental(
            **features,
            lm_hidden_states=lm_hidden_states,
            include_embeddings=True,
            **FAST_FOLD,
        )

    pooled = output["output_embedding_pair_pooled"]
    assert pooled.shape == (1, len(TINY_SEQUENCE), _pair_width(tiny_esmfold2_config))
    assert pooled.dtype == torch.float32


def test_lm_dropout_zero_disables_and_restores(tiny_esmfold2):
    lm_encoder = tiny_esmfold2.config.lm_encoder
    before = (lm_encoder.lm_dropout, lm_encoder.per_loop_lm_dropout)

    with _lm_dropout_context(tiny_esmfold2, 0):
        assert lm_encoder.lm_dropout == 0

    assert (lm_encoder.lm_dropout, lm_encoder.per_loop_lm_dropout) == before


def test_lm_dropout_none_keeps_the_checkpoint_rate(tiny_esmfold2):
    lm_encoder = tiny_esmfold2.config.lm_encoder
    before = lm_encoder.lm_dropout

    with _lm_dropout_context(tiny_esmfold2, None):
        assert lm_encoder.lm_dropout == before
