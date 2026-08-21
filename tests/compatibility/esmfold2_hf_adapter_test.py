"""``EsmFold2HFAdapter``: our ESMFold2 API presented over the upstream HF port.

The port is not importable here (transformers 5.16 vs the 4.x this repo pins), so
the adapter is driven through a stub reproducing its measured call contract:
``fold`` is the end-to-end entry point, it takes ``attention_mask``, it reads the
sampler settings off ``config.structure_head`` at call time, it returns no
``atom_pad_mask`` / ``residue_index`` / ``entity_id``, and it swallows unknown
kwargs. Behind the stub sits our own tiny ESMFold2, so the tensors are real. The
one case needing their classes is gated on :func:`upstream_available`.
"""

import copy
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from esm.models.esmfold2 import layers as _layers
from esm.models.esmfold2.hf_adapter import (
    EsmFold2HFAdapter,
    translate_features,
    upstream_available,
)
from esm.models.esmfold2.processor import ESMFold2InputBuilder
from esm.models.esmfold2.protein_utils import (
    OUTPUT_TO_PDB_FEATURE_KEYS,
    prepare_protein_features,
)
from esm.models.esmfold2.types import ProteinInput, StructurePredictionInput
from esm.utils.structure.molecular_complex import MolecularComplexResult
from tests.conftest import ESMFOLD2_SEQUENCES, esmfold2_inputs

TINY_SEQUENCE = ESMFOLD2_SEQUENCES["tiny"]

# A forward is O(L^2); these keep the module fast.
FAST_FOLD = dict(num_loops=1, num_sampling_steps=2, num_diffusion_samples=1)

# The Karras schedule's churn step, the only place noise_scale enters, is zero
# below four steps on both sides, so a two-step override test proves nothing.
SAMPLER_OVERRIDE_STEPS = 4

# In our output dict, absent from theirs, and read by ESMFold2InputBuilder.decode.
UPSTREAM_ABSENT_OUTPUT_KEYS = ("atom_pad_mask", "residue_index", "entity_id")

# Emitted by our featurizer, consumed by no model.
FEATURIZER_ONLY_KEYS = ("pocket_feature", "gt_coords", "is_resolved", "frames_idx")


# ---------------------------------------------------------------------------
# A stand-in for the upstream model
# ---------------------------------------------------------------------------


class ChunkedBlock(torch.nn.Module):
    """A public ``chunk_size``, as upstream spells it; ours is ``_chunk_size``."""

    def __init__(self, chunk_size: int | None) -> None:
        super().__init__()
        self.chunk_size = chunk_size


class UpstreamStub(torch.nn.Module):
    """The port's ``fold`` contract, backed by one of our own tiny models.

    Records every call, because their real ``fold`` drops unknown keys silently
    and a translation bug is invisible from the outputs alone.
    """

    # Declared, or nn.Module.__getattr__ widens every read to Tensor | Module.
    wrapped: Any
    config: Any

    def __init__(self, tiny_model: Any) -> None:
        super().__init__()
        self.wrapped = tiny_model
        ours = tiny_model.config
        self.config = SimpleNamespace(
            structure_head=SimpleNamespace(
                noise_scale=ours.structure_head.noise_scale,
                step_scale=ours.structure_head.step_scale,
                # Upstream's spelling of our ``max_inference_sigma`` kwarg.
                inference_sigma_cap=256.0,
            ),
            # Real, so _lm_dropout_context drives the wrapped model.
            lm_encoder=ours.lm_encoder,
            chunk_size=64,
            esmc_config=SimpleNamespace(
                bos_token_id=0, eos_token_id=2, pad_token_id=1, mask_token_id=32
            ),
        )
        self.esmc = None
        # Upstream builds some of these with chunk_size=None; the adapter overrides all.
        self.chunked = torch.nn.ModuleList(
            [ChunkedBlock(64), ChunkedBlock(64), ChunkedBlock(None)]
        )
        self.fold_calls: list[dict] = []
        self.fold_returns: list[dict] = []
        self.trunk_calls: list[dict] = []

    @property
    def device(self) -> torch.device:
        return self.wrapped.device

    def fold(self, **kwargs):
        self.fold_calls.append(kwargs)
        head = self.config.structure_head
        features = {
            key: value
            for key, value in kwargs.items()
            if key not in ("num_loops", "num_diffusion_samples", "num_sampling_steps")
        }
        features["token_attention_mask"] = features.pop("attention_mask")
        output = self.wrapped(
            **features,
            num_loops=kwargs["num_loops"],
            num_diffusion_samples=kwargs["num_diffusion_samples"],
            num_sampling_steps=kwargs["num_sampling_steps"],
            noise_scale=head.noise_scale,
            step_scale=head.step_scale,
            max_inference_sigma=head.inference_sigma_cap,
        )
        upstream_shaped = {
            key: value
            for key, value in output.items()
            if key not in UPSTREAM_ABSENT_OUTPUT_KEYS
        }
        self.fold_returns.append(upstream_shaped)
        return upstream_shaped

    def forward(self, **kwargs):
        """Their ``forward``: trunk only. Recorded to prove nothing routes here."""
        self.trunk_calls.append(kwargs)
        return SimpleNamespace(distogram_logits=None)


@pytest.fixture(autouse=True)
def _force_reference_attention(monkeypatch):
    """flash-attn imports on a CPU box but needs a CUDA runtime to run."""
    if not torch.cuda.is_available():
        monkeypatch.setattr(_layers, "FLASH_ATTN_AVAILABLE", False)


@pytest.fixture
def adapter(tiny_esmfold2) -> EsmFold2HFAdapter:
    return EsmFold2HFAdapter(UpstreamStub(tiny_esmfold2)).eval()


@pytest.fixture(scope="session")
def builder(ccd_pickle) -> ESMFold2InputBuilder:
    """The real input builder; constructing it loads the CCD dictionary."""
    return ESMFold2InputBuilder()


def protein_input(sequence: str = TINY_SEQUENCE) -> StructurePredictionInput:
    return StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])


def tiny_features(model) -> dict:
    """Batched features plus the synthetic LM states, since the tiny model has no PLM."""
    features, lm_hidden_states = esmfold2_inputs(model, TINY_SEQUENCE)
    return {**features, "lm_hidden_states": lm_hidden_states}


# ---------------------------------------------------------------------------
# Feature translation
# ---------------------------------------------------------------------------


def test_translate_features_renames_only_the_attention_mask():
    """One rename in 23 keys, so pin the exact set."""
    features = prepare_protein_features(TINY_SEQUENCE)
    translated = translate_features(features)

    assert "token_attention_mask" not in translated
    assert translated["attention_mask"] is features["token_attention_mask"]
    assert set(translated) == (set(features) - {"token_attention_mask"}) | {
        "attention_mask"
    }


def test_translate_features_drops_the_featurizer_only_keys():
    """Their ``fold`` would swallow these, making a real typo indistinguishable."""
    features = prepare_protein_features(TINY_SEQUENCE)
    padded = {**features, **{key: torch.zeros(1) for key in FEATURIZER_ONLY_KEYS}}

    translated = translate_features(padded)

    assert not set(translated) & set(FEATURIZER_ONLY_KEYS)
    assert set(translated) == set(translate_features(features))


def test_translate_features_drops_an_empty_distogram_conditioning():
    features = prepare_protein_features(TINY_SEQUENCE)
    length = features["res_type"].shape[-1]
    translated = translate_features(
        {
            **features,
            "disto_cond": torch.zeros(1, length, length, 8),
            "disto_cond_mask": torch.zeros(1, length, length, dtype=torch.bool),
        }
    )
    assert "disto_cond" not in translated and "disto_cond_mask" not in translated


def test_translate_features_refuses_a_populated_distogram_conditioning():
    """Dropping the mask silently would fold unconditioned; ours raises instead."""
    features = prepare_protein_features(TINY_SEQUENCE)
    length = features["res_type"].shape[-1]
    mask = torch.zeros(1, length, length, dtype=torch.bool)
    mask[0, 0, 1] = True

    with pytest.raises(NotImplementedError, match="distogram conditioning"):
        translate_features({**features, "disto_cond_mask": mask})


# ---------------------------------------------------------------------------
# forward -> fold
# ---------------------------------------------------------------------------


def test_forward_dispatches_to_fold_not_to_their_forward(adapter):
    """Both sides have a ``forward``; theirs is the trunk alone."""
    adapter(**tiny_features(adapter.model.wrapped), **FAST_FOLD)

    assert len(adapter.model.fold_calls) == 1
    assert adapter.model.trunk_calls == []


def test_forward_sends_the_translated_key_set(adapter):
    """Asserted directly: upstream would accept the untranslated dict and ignore half."""
    adapter(**tiny_features(adapter.model.wrapped), **FAST_FOLD)
    sent = adapter.model.fold_calls[0]

    assert "attention_mask" in sent
    assert "token_attention_mask" not in sent
    assert not set(sent) & set(FEATURIZER_ONLY_KEYS)
    # The inference knobs are named arguments of ``fold``, not features.
    for key, value in FAST_FOLD.items():
        assert sent[key] == value
    # ``None`` would override a config default with nothing, so it is filtered.
    assert all(value is not None for value in sent.values())


def test_forward_rejects_an_unknown_keyword(adapter):
    """Theirs swallows a misspelled override; ours has to raise."""
    with pytest.raises(TypeError, match="noize_scale"):
        adapter(**tiny_features(adapter.model.wrapped), noize_scale=2.0, **FAST_FOLD)
    assert adapter.model.fold_calls == []


def test_an_unknown_keyword_to_the_constructor_raises(tiny_esmfold2):
    """The adapter wraps one model and takes nothing else; a keyword here would
    be a caller configuring something that does not exist."""
    with pytest.raises(TypeError, match="precision"):
        EsmFold2HFAdapter(UpstreamStub(tiny_esmfold2), precision="bf16")  # ty:ignore[unknown-argument]


def test_forward_accepts_the_featurizer_only_keys(adapter):
    """The full featurizer dict has to go straight in, exactly as with our model."""
    features = tiny_features(adapter.model.wrapped)
    extra = {key: torch.zeros(1) for key in FEATURIZER_ONLY_KEYS}

    output = adapter(**features, **extra, **FAST_FOLD)

    assert "sample_atom_coords" in output


def test_forward_reattaches_the_keys_decode_reads(adapter):
    """``EsmFold2Output`` carries none of the three, and ``decode`` reads all three."""
    features = tiny_features(adapter.model.wrapped)
    output = adapter(**features, **FAST_FOLD)

    assert not set(adapter.model.fold_returns[0]) & set(UPSTREAM_ABSENT_OUTPUT_KEYS)
    for key in UPSTREAM_ABSENT_OUTPUT_KEYS:
        assert key in output
    assert torch.equal(output["residue_index"], features["residue_index"])
    assert torch.equal(output["entity_id"], features["entity_id"])


def test_forward_output_matches_our_models_key_set(tiny_esmfold2, adapter):
    """Anything downstream reads off our output has to be there."""
    features = tiny_features(tiny_esmfold2)

    with torch.no_grad():
        ours = tiny_esmfold2(**features, **FAST_FOLD)
        theirs = adapter(**features, **FAST_FOLD)

    assert set(ours) == set(theirs)


# ---------------------------------------------------------------------------
# Sampler overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "config_field"),
    [
        ({"noise_scale": 2.0}, "noise_scale"),
        ({"step_scale": 2.5}, "step_scale"),
        ({"max_inference_sigma": 100.0}, "inference_sigma_cap"),
    ],
)
def test_sampler_overrides_reach_the_sampler(adapter, override, config_field):
    """Their ``fold`` takes no sampler kwargs, so the adapter scopes them onto the
    config the sampler reads at call time."""
    features = tiny_features(adapter.model.wrapped)
    settings = {**FAST_FOLD, "num_sampling_steps": SAMPLER_OVERRIDE_STEPS}
    seen: list[float] = []

    head = adapter.config.structure_head
    real_fold = adapter.model.fold

    def spy(**kwargs):
        seen.append(getattr(head, config_field))
        return real_fold(**kwargs)

    adapter.model.fold = spy
    torch.manual_seed(0)
    with torch.no_grad():
        baseline = adapter(**features, **settings)["sample_atom_coords"]
    torch.manual_seed(0)
    with torch.no_grad():
        overridden = adapter(**features, **settings, **override)["sample_atom_coords"]

    assert seen[1] == pytest.approx(next(iter(override.values())))
    assert not torch.allclose(baseline, overridden)


def test_sampler_overrides_are_restored_after_the_call(adapter):
    """A mutation of shared state has to be undone, including when the fold raises."""
    features = tiny_features(adapter.model.wrapped)
    head = adapter.config.structure_head
    before = copy.copy(vars(head))

    with torch.no_grad():
        adapter(**features, **FAST_FOLD, noise_scale=2.0, step_scale=2.5)
    assert vars(head) == before

    def boom(**_kwargs):
        raise RuntimeError("fold blew up")

    adapter.model.fold = boom
    with pytest.raises(RuntimeError, match="blew up"):
        adapter(**features, **FAST_FOLD, noise_scale=2.0)
    assert vars(head) == before


# ---------------------------------------------------------------------------
# The builder round trip: the reason the adapter exists
# ---------------------------------------------------------------------------


def test_builder_folds_decodes_and_renders(adapter, builder):
    """``fold`` -> ``decode`` -> ``to_pdb``, which the raw upstream model cannot do."""
    result = builder.fold(
        adapter, protein_input(), seed=0, lm_dropout=None, **FAST_FOLD
    )

    assert isinstance(result, MolecularComplexResult)
    assert "ATOM" in result.complex.to_protein_complex().to_pdb_string()
    assert result.complex.to_mmcif()


def test_builder_returns_one_result_per_diffusion_sample(adapter, builder):
    settings = {**FAST_FOLD, "num_diffusion_samples": 3}
    results = builder.fold(
        adapter, protein_input(), seed=0, lm_dropout=None, **settings
    )

    assert isinstance(results, list) and len(results) == 3


def test_builder_fold_drives_lm_dropout_through_their_config(adapter, builder):
    """``config.lm_encoder.{lm_dropout, per_loop_lm_dropout}`` survive the port."""
    lm_encoder = adapter.config.lm_encoder
    before = (lm_encoder.lm_dropout, lm_encoder.per_loop_lm_dropout)

    builder.fold(adapter, protein_input(), seed=0, lm_dropout=0.3, **FAST_FOLD)

    assert (lm_encoder.lm_dropout, lm_encoder.per_loop_lm_dropout) == before


def test_infer_protein_reattaches_the_pdb_feature_keys(adapter):
    output = adapter.infer_protein(TINY_SEQUENCE, **FAST_FOLD)

    for key in OUTPUT_TO_PDB_FEATURE_KEYS:
        assert key in output


def test_infer_protein_as_pdb_renders(adapter):
    pdb = adapter.infer_protein_as_pdb(TINY_SEQUENCE, **FAST_FOLD)

    assert "ATOM" in pdb


# ---------------------------------------------------------------------------
# Perf knobs and the surface the port does not have
# ---------------------------------------------------------------------------


def test_set_chunk_size_walks_every_module_that_holds_one(adapter):
    """Upstream reads ``config.chunk_size`` once, so setting it after load is inert."""
    adapter.set_chunk_size(4)

    assert adapter.config.chunk_size == 4
    holders = [m for m in adapter.model.modules() if hasattr(m, "chunk_size")]
    assert holders, "the stub should expose at least one chunk_size holder"
    assert all(m.chunk_size == 4 for m in holders)

    adapter.set_chunk_size(None)
    assert all(m.chunk_size is None for m in holders)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda a: type(a).from_pretrained("anything", load_esmc=False),
            "load_esmc=False",
        )
    ],
)
def test_the_unported_surface_refuses_loudly(adapter, call, match):
    """Capability the port lacks: refusing beats a silent no-op."""
    with pytest.raises(NotImplementedError, match=match):
        call(adapter)


def test_passthrough_attributes(adapter, tiny_esmfold2):
    assert adapter.config is adapter.model.config
    assert adapter.device == tiny_esmfold2.device


# ---------------------------------------------------------------------------
# The upstream dependency itself
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    upstream_available(), reason="upstream transformers ESMFold2 port is installed"
)
@pytest.mark.parametrize(
    "call",
    [
        lambda: EsmFold2HFAdapter.from_pretrained("biohub/ESMFold2"),
        lambda: EsmFold2HFAdapter(object()).trunk(),
    ],
)
def test_without_the_port_installed_the_error_says_why(call):
    """The module still imports; only the two entry points needing their classes fail."""
    with pytest.raises(ImportError, match="transformers 5.16"):
        call()


@pytest.mark.skipif(
    not upstream_available(),
    reason="needs transformers >= 5.16 (the ESMFold2 port); this repo pins 4.x",
)
def test_against_the_real_port(tiny_esmfold2_config):
    """The one thing the stub cannot cover: their classes construct and their
    ``fold`` accepts what the adapter sends. Weights are out of scope."""
    from transformers.models.esmfold2.modeling_esmfold2 import (
        EsmFold2Model as UpstreamEsmFold2Model,
    )

    adapter = EsmFold2HFAdapter(UpstreamEsmFold2Model(tiny_esmfold2_config).eval())
    features = prepare_protein_features(TINY_SEQUENCE)
    lm_hidden_states = torch.zeros(
        1,
        features["res_type"].shape[-1],
        adapter.config.esmc_config.num_hidden_layers + 1,
        adapter.config.esmc_config.hidden_size,
    )

    with torch.no_grad():
        output = adapter(**features, lm_hidden_states=lm_hidden_states, **FAST_FOLD)

    assert "sample_atom_coords" in output
    for key in UPSTREAM_ABSENT_OUTPUT_KEYS:
        assert key in output
