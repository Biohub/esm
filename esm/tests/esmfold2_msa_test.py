"""ESMFold2 MSA-path tests.

Every case runs on CPU against a tiny randomly-initialised model with
``msa_encoder.enabled=True`` and synthetic LM states, so no published MSA
checkpoint and no PLM backbone are needed. That is deliberate: until this file
existed, no test in the repo had ever run a forward pass with the MSA encoder
attached, and the first line inside the MSA branch read a config attribute that
does not exist on any config (see
``test_msa_encoder_overwrite_changes_the_pair_stream``).
"""

import pytest
import torch

from esm.models.esmfold2 import EsmFold2Config, EsmFold2Model
from esm.models.esmfold2 import layers as _layers
from esm.models.esmfold2 import model as _model
from esm.models.esmfold2.constants import MSA_GAP_TOKEN_ID
from esm.models.esmfold2.layers import NUM_RES_TYPES
from esm.models.esmfold2.prepare_input import prepare_esmfold2_input
from esm.models.esmfold2.types import MSA, ProteinInput, StructurePredictionInput
from esm.tests.conftest import ESMFOLD2_SEQUENCES, esmfold2_inputs

# A forward is O(L^2)
# and the default schedule is 68 ODE steps.
FAST_SAMPLER = dict(num_loops=1, num_sampling_steps=2, num_diffusion_samples=1)

TINY_SEQUENCE = ESMFOLD2_SEQUENCES["tiny"]

# The four blocks below are the smallest MSA encoder that still exercises every
# sub-module: the non-final block owns the pair-weighted averaging and the MSA
# transition, the final block owns only the outer product and the triangle
# updates, so two layers are the minimum that reaches both.
TINY_MSA_ENCODER = dict(
    enabled=True,
    hidden_size=32,
    outer_hidden_size=8,
    num_hidden_layers=2,
    num_attention_heads=2,
    head_width=8,
)

#: The row cap lives in the checkpoint config, so switching subsampling off is a
#: model-build override rather than a call keyword: a null cap means no draw.
#: ``msa_max_depth=None`` at the call site means "use the config's value".
_NO_ROW_SUBSAMPLING = dict(max_depth=None)


@pytest.fixture(autouse=True)
def _force_reference_attention(monkeypatch):
    """flash-attn imports on a CPU box but needs a CUDA runtime to run."""
    if not torch.cuda.is_available():
        monkeypatch.setattr(_layers, "FLASH_ATTN_AVAILABLE", False)


def msa_model(tiny_esmfold2_config, **msa_overrides) -> EsmFold2Model:
    """A tiny model with the MSA encoder attached.

    Parameters
    ----------
    tiny_esmfold2_config : EsmFold2Config
        The session-scoped tiny config; it is round-tripped through
        ``to_dict`` rather than mutated, because it is shared with every other
        test in the suite.
    **msa_overrides
        Fields of ``EsmFold2MsaEncoderConfig`` to override.

    Returns
    -------
    EsmFold2Model
        Seeded from ``torch.manual_seed(0)``, so two models built with
        structurally identical configs hold bit-identical weights.
    """
    raw = tiny_esmfold2_config.to_dict()
    raw.pop("model_type", None)
    raw["msa_encoder"] = {**TINY_MSA_ENCODER, **msa_overrides}
    torch.manual_seed(0)
    model = EsmFold2Model(EsmFold2Config(**raw)).eval()
    model.set_chunk_size(None)
    return model


def msa_inputs(model, depth: int, seq: str = TINY_SEQUENCE, seed: int = 0):
    """Single-chain features whose MSA has been widened to ``depth`` rows.

    ``prepare_protein_features`` always yields a depth-1 alignment (the query
    itself), so the deeper rows are drawn here rather than through the paired
    MSA builder: the depth has to be exact for the subsampling assertions, and
    row content only has to be distinct, not biological.

    Returns
    -------
    tuple[dict, torch.Tensor]
        Feature dict and synthetic LM hidden states.
    """
    features, lm_hidden_states = esmfold2_inputs(model, seq)
    length = features["res_type"].shape[-1]
    generator = torch.Generator().manual_seed(seed)
    # Skip token 0 (pad) and MSA_GAP_TOKEN_ID so a random row is never mistaken
    # for a gap row by an assertion below.
    msa = torch.randint(
        MSA_GAP_TOKEN_ID + 1, NUM_RES_TYPES, (1, depth, length), generator=generator
    )
    msa[0, 0] = features["res_type"][0]  # row 0 is the query, by convention
    features["msa"] = msa
    features["msa_attention_mask"] = torch.ones(1, depth, length, dtype=torch.bool)
    features["has_deletion"] = torch.zeros(1, depth, length, dtype=torch.bool)
    features["deletion_value"] = torch.zeros(1, depth, length)
    return features, lm_hidden_states


#: Reset before every forward: the pair state is initialised from a truncated
#: normal and the sampler draws fresh noise, so two runs only differ in what the
#: test changed if the RNG starts from the same place.
FORWARD_SEED = 1234


def run(model, features, lm_hidden_states, **kwargs) -> dict:
    """One seeded forward at the fast-tier sampler settings."""
    torch.manual_seed(FORWARD_SEED)
    with torch.no_grad():
        return model(
            **features, lm_hidden_states=lm_hidden_states, **{**FAST_SAMPLER, **kwargs}
        )


def record_encoder_calls(model) -> list[dict]:
    """Capture the keyword arguments of every ``MSAEncoder`` forward."""
    calls: list[dict] = []

    def hook(_module, _args, kwargs):
        calls.append(kwargs)
        return None

    model.msa_encoder.register_forward_pre_hook(hook, with_kwargs=True)
    return calls


def record_subsample_calls(monkeypatch) -> list[tuple]:
    """Capture ``(input_mask, subsampled_msa)`` for every row-subsample draw."""
    calls: list[tuple] = []
    real = _model.maybe_subsample_msa

    def spy(msa, msa_attention_mask, *args, **kwargs):
        out = real(msa, msa_attention_mask, *args, **kwargs)
        calls.append((msa_attention_mask, out[0]))
        return out

    monkeypatch.setattr(_model, "maybe_subsample_msa", spy)
    return calls


def record_column_mask_calls(monkeypatch) -> list:
    """Capture every column-masking call."""
    calls: list = []
    real = _model.maybe_apply_msa_column_masking

    def spy(msa_attention_mask, *args, **kwargs):
        calls.append(msa_attention_mask)
        return real(msa_attention_mask, *args, **kwargs)

    monkeypatch.setattr(_model, "maybe_apply_msa_column_masking", spy)
    return calls


# ---------------------------------------------------------------------------
# Dispatch — assert the module tree before comparing any tensor
# ---------------------------------------------------------------------------


def test_msa_encoder_is_built_only_when_enabled(tiny_esmfold2_config, tiny_esmfold2):
    """A disabled MSA encoder is absent, not an inert module that still runs."""
    assert tiny_esmfold2.msa_encoder is None

    model = msa_model(tiny_esmfold2_config)
    assert model.msa_encoder is not None
    assert len(model.msa_encoder.blocks) == TINY_MSA_ENCODER["num_hidden_layers"]
    # Only the last block drops the MSA-side updates; a stack that marked every
    # block final would still produce correctly-shaped output.
    assert [b.is_final_block for b in model.msa_encoder.blocks] == [False, True]


# ---------------------------------------------------------------------------
# Shapes at depth 1, 32 and above msa_max_depth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "depth,msa_max_depth,encoder_depth",
    [
        # A depth-1 alignment is what prepare_protein_features emits, so this is
        # the path every single-sequence fold takes.
        (1, 32, 1),
        (32, 32, 32),
        # Above the cap: subsampling keeps the query row plus msa_max_depth - 1
        # others, so the encoder still sees exactly msa_max_depth rows.
        (96, 32, 32),
    ],
    ids=["depth_1", "depth_32", "above_max_depth"],
)
def test_msa_shapes_across_depth_regimes(
    tiny_esmfold2_config, depth, msa_max_depth, encoder_depth
):
    """The MSA axis is transposed to ``[B, L, R, ...]`` before the encoder.

    The encoder input is asserted as an equality over the full shape rather than
    a rank check: swapping the row axis for the token axis is the bug this
    catches, and both are plausible-looking 3-D tensors.
    """
    model = msa_model(tiny_esmfold2_config)
    features, lm_hidden_states = msa_inputs(model, depth=depth)
    length = features["res_type"].shape[-1]
    calls = record_encoder_calls(model)

    out = run(
        model,
        features,
        lm_hidden_states,
        msa_max_depth=msa_max_depth,
        msa_column_mask_rate=0.0,
    )

    # num_loops=1 => total_steps = 2, so the encoder runs twice.
    assert len(calls) == FAST_SAMPLER["num_loops"] + 1
    for kwargs in calls:
        assert kwargs["msa_oh"].shape == (1, length, encoder_depth, NUM_RES_TYPES)
        assert kwargs["has_deletion"].shape == (1, length, encoder_depth)
        assert kwargs["deletion_value"].shape == (1, length, encoder_depth)
        assert kwargs["msa_attention_mask"].shape == (1, length, encoder_depth)
        assert kwargs["x_pair"].shape == (
            1,
            length,
            length,
            model.config.pairwise_hidden_size,
        )

    n_atoms = features["atom_attention_mask"].shape[-1]
    assert out["sample_atom_coords"].shape == (1, n_atoms, 3)
    assert out["plddt"].shape == (1, length)
    assert torch.isfinite(out["sample_atom_coords"]).all()


# ---------------------------------------------------------------------------
# Column masking once, row subsampling per loop
# ---------------------------------------------------------------------------


def test_column_masking_applies_once_and_subsampling_redraws_per_loop(
    tiny_esmfold2_config, monkeypatch
):
    """Column mask is shared across loops; the row subset is redrawn per loop.

    Both augmentations exist to diversify an ensemble of folds, but they sit on
    opposite sides of the loop: masking a *different* set of columns each loop
    would change the effective alignment mid-refinement, while reusing one row
    subset would collapse the diversity the deep alignment is there to provide.
    """
    num_loops = 3
    total_steps = num_loops + 1
    model = msa_model(tiny_esmfold2_config)
    features, lm_hidden_states = msa_inputs(model, depth=64)

    column_calls = record_column_mask_calls(monkeypatch)
    subsample_calls = record_subsample_calls(monkeypatch)

    run(
        model,
        features,
        lm_hidden_states,
        num_loops=num_loops,
        msa_max_depth=8,
        msa_column_mask_rate=0.5,
    )

    assert len(column_calls) == 1
    assert len(subsample_calls) == total_steps

    # Every loop subsamples the *same* masked alignment object, which is what
    # "applied once, before the loop" means operationally.
    masks = [mask for mask, _ in subsample_calls]
    assert all(mask is masks[0] for mask in masks)

    # Row content is drawn from randint over 31 token values at 12 positions, so
    # two different row subsets colliding on content has probability ~0; an
    # exact count therefore pins "redrawn every loop" rather than "redrawn
    # sometimes".
    row_sets = {tuple(msa.flatten().tolist()) for _, msa in subsample_calls}
    assert len(row_sets) == total_steps


#: Depth of the alignment the full-depth cases below run on.
_FULL_DEPTH = 64


@pytest.mark.parametrize(
    "config_overrides,call_overrides",
    [
        # A null cap in the checkpoint config: no draw is made at all. This is
        # the only route that switches subsampling off outright, now that
        # ``msa_max_depth=None`` at the call means "use the config's value".
        (_NO_ROW_SUBSAMPLING, {}),
        # A cap at the real depth: the draw is attempted and returns early.
        ({}, {"msa_max_depth": _FULL_DEPTH}),
    ],
    ids=["null_cap", "cap_at_depth"],
)
def test_full_depth_is_used_when_subsampling_is_off(
    tiny_esmfold2_config, config_overrides, call_overrides
):
    """Either route keeps all rows, and keeps them identical across loops."""
    depth = _FULL_DEPTH
    model = msa_model(tiny_esmfold2_config, **config_overrides)
    features, lm_hidden_states = msa_inputs(model, depth=depth)
    length = features["res_type"].shape[-1]
    calls = record_encoder_calls(model)

    run(
        model,
        features,
        lm_hidden_states,
        num_loops=2,
        msa_column_mask_rate=0.0,
        **call_overrides,
    )

    assert len(calls) == 3
    for kwargs in calls:
        assert kwargs["msa_oh"].shape == (1, length, depth, NUM_RES_TYPES)
    # No subsampling means no redraw: every loop sees the same rows.
    assert torch.equal(calls[0]["msa_oh"], calls[-1]["msa_oh"])


# ---------------------------------------------------------------------------
# Gap tokens and the one-hot zeroing at padded positions
# ---------------------------------------------------------------------------


def test_gap_token_fills_chains_without_an_alignment(ccd_pickle):
    """A chain with no MSA gets gaps everywhere except its own query row.

    ``MSA_GAP_TOKEN_ID`` has to be a real ``res_type`` index, because the model
    one-hots the alignment over ``NUM_RES_TYPES``; a sentinel outside that range
    would either throw or silently wrap into another residue's channel.
    """
    assert 0 <= MSA_GAP_TOKEN_ID < NUM_RES_TYPES

    with_msa = ESMFOLD2_SEQUENCES["tiny"]
    without_msa = "ACDEFGHI"
    homologue = "MQIFVKTLTGKS"
    features, _ = prepare_esmfold2_input(
        StructurePredictionInput(
            sequences=[
                ProteinInput(
                    id="A",
                    sequence=with_msa,
                    msa=MSA.from_sequences([with_msa, homologue]),
                ),
                ProteinInput(id="B", sequence=without_msa),
            ]
        )
    )

    msa = features["msa"]
    assert msa.shape[1] == len(with_msa) + len(without_msa)
    # Without a second row there is nothing for the gap fill to have written.
    assert msa.shape[0] >= 2
    unaligned = slice(len(with_msa), None)
    # Row 0 of the unaligned chain carries its own residues ...
    assert torch.equal(msa[0, unaligned], features["res_type"][unaligned])
    # ... and every other row is a gap.
    assert (msa[1:, unaligned] == MSA_GAP_TOKEN_ID).all()


def masked_msa_inputs(model, depth: int = 16):
    """Features whose alignment carries eight padded rows and three padded columns."""
    features, lm_hidden_states = msa_inputs(model, depth=depth)
    mask = features["msa_attention_mask"].clone()
    mask[0, depth // 2 :, :] = False  # padded rows
    mask[0, :, -3:] = False  # padded columns
    mask[0, 0, :] = True  # the query row is never padded
    features["msa_attention_mask"] = mask
    return features, lm_hidden_states, mask


# Both padded-position tests hold everything else fixed: the column mask and the
# row subsample are the only other sources of randomness on the MSA path.
_DETERMINISTIC_MSA = dict(msa_column_mask_rate=0.0)


def test_padded_msa_positions_cannot_reach_the_encoder(tiny_esmfold2_config):
    """Garbage residues under a false ``msa_attention_mask`` must not move the output.

    ``MSAEncoder.embed`` is bias-free, so zeroing the one-hot at masked
    positions is what makes a padded position contribute nothing. Overwriting
    the padded entries with a different residue rather than merely re-running is
    what separates a real mask from a no-op.
    """
    model = msa_model(tiny_esmfold2_config, **_NO_ROW_SUBSAMPLING)
    features, lm_hidden_states, mask = masked_msa_inputs(model)

    clean = run(model, features, lm_hidden_states, **_DETERMINISTIC_MSA)

    poisoned = dict(features)
    residues = features["msa"].clone()
    residues[~mask] = NUM_RES_TYPES - 1
    poisoned["msa"] = residues

    dirty = run(model, poisoned, lm_hidden_states, **_DETERMINISTIC_MSA)

    # Bit-equality, not a tolerance: the masked one-hot rows are multiplied by
    # zero, so no arithmetic difference can survive.
    torch.testing.assert_close(
        clean["sample_atom_coords"], dirty["sample_atom_coords"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        clean["distogram_logits"], dirty["distogram_logits"], atol=0, rtol=0
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "_run_one_loop zeroes only the 33 one-hot channels of the MSA encoder "
        "input; has_deletion and deletion_value are the other two channels of "
        "the same bias-free Linear and are passed through unmasked. A column "
        "that msa_column_mask_rate masks out therefore still contributes its "
        "deletion signal. Pinned rather than fixed here: the fix changes the "
        "numerics of every MSA checkpoint."
    ),
)
def test_padded_deletion_features_cannot_reach_the_encoder(tiny_esmfold2_config):
    """Deletion features at masked positions must be zeroed like the one-hot."""
    model = msa_model(tiny_esmfold2_config, **_NO_ROW_SUBSAMPLING)
    features, lm_hidden_states, mask = masked_msa_inputs(model)

    clean = run(model, features, lm_hidden_states, **_DETERMINISTIC_MSA)

    poisoned = dict(features)
    poisoned["has_deletion"] = ~mask
    poisoned["deletion_value"] = (~mask).float() * 7.0

    dirty = run(model, poisoned, lm_hidden_states, **_DETERMINISTIC_MSA)

    torch.testing.assert_close(
        clean["distogram_logits"], dirty["distogram_logits"], atol=0, rtol=0
    )


# ---------------------------------------------------------------------------
# ``config.msa_encoder_overwrite`` reaches the pair stream
# ---------------------------------------------------------------------------


def test_msa_encoder_overwrite_changes_the_pair_stream(tiny_esmfold2_config):
    """``overwrite`` selects replace-vs-add, and both settings must run.

    ``model.py`` read ``self.config.msa_encoder_overwrite``, a flat key that
    ``_LEGACY_PATHS`` remaps to ``msa_encoder.overwrite`` before the config
    object is built — so the flat attribute exists on no config, and this line
    raised ``AttributeError`` on the first loop of any MSA-enabled checkpoint.
    Nothing caught it because no test had ever run the MSA encoder.
    """
    for value in (True, False):
        config = EsmFold2Config(msa_encoder={"enabled": True, "overwrite": value})
        assert config.msa_encoder.overwrite is value
        assert not hasattr(config, "msa_encoder_overwrite")

    overwriting = msa_model(tiny_esmfold2_config, overwrite=True, **_NO_ROW_SUBSAMPLING)
    adding = msa_model(tiny_esmfold2_config, overwrite=False, **_NO_ROW_SUBSAMPLING)
    # Same seed, same architecture => the two models differ only in the flag.
    for left, right in zip(overwriting.parameters(), adding.parameters()):
        assert torch.equal(left, right)

    features, lm_hidden_states = msa_inputs(overwriting, depth=8)

    replaced = run(overwriting, features, lm_hidden_states, **_DETERMINISTIC_MSA)
    added = run(adding, features, lm_hidden_states, **_DETERMINISTIC_MSA)

    # overwrite=True feeds the encoder output to the trunk in place of z_init;
    # overwrite=False adds it. Identical outputs would mean the flag is dead.
    assert not torch.allclose(
        replaced["distogram_logits"], added["distogram_logits"], atol=1e-4
    )
