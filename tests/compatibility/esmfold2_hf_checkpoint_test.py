"""Loading an upstream HuggingFace-port ESMFold2 checkpoint into this package.

The remap is chosen from the tensor key names, so both layouts load through the
same ``from_pretrained``. Rule *correctness* is pinned against the port's own
``convert_esmfold2_checkpoint`` in :func:`test_rules_invert_their_converter`,
which needs upstream installed and skips here; everything else -- detection, the
structural repacking, the config filter -- runs anywhere.
"""

import json
from pathlib import Path

import pytest
import torch

from esm.models.esmfold2 import EsmFold2Config, EsmFold2Model
from esm.models.esmfold2.config import default_module_flags
from esm.models.esmfold2.hf_adapter import upstream_available
from esm.models.esmfold2.hf_checkpoint import (
    hf_state_dict_to_native,
    is_hf_layout,
    to_native_key,
)

# Real names from ``Rocketknight1/ESMFold2-merged-temp``, one per structural family.
HF_KEYS = {
    "folding_trunk.layers.0.pair_transition.mlp.gate_up_proj.weight": "folding_trunk.blocks.0.pair_transition.ffn.w12.weight",
    "folding_trunk.layers.0.tri_mul_in.proj_gate.weight": "folding_trunk.blocks.0.tri_mul_in._engine.proj_gate.weight",
    "input_embedder.pair_init_1.weight": "z_init_1.weight",
    "parcae.log_state_decay": "parcae_log_a",
    "language_model.layer_weights": "language_model.base_z_combine",
    "confidence_head.pae_layernorm.bias": "confidence_head.pae_ln.bias",
    "msa_encoder.layers.0.outer_product_mean.input_proj.weight": "msa_encoder.blocks.0.outer_product_mean.W.weight",
    "structure_head.coords_linear.weight": "structure_head.diffusion_module.atom_encoder.coords_linear.weight",
    "structure_head.token_transformer.layers.0.input_layernorm.gate_proj.bias": "structure_head.diffusion_module.token_transformer.attn_blocks.0.adaln.s_gate.bias",
}


def hf_tensors(keys, rows: int = 6, cols: int = 4) -> dict[str, torch.Tensor]:
    return {k: torch.randn(rows, cols) for k in keys}


# ---------------------------------------------------------------------------
# Which layout is this?
# ---------------------------------------------------------------------------


def test_our_own_layout_is_recognised(tiny_esmfold2):
    assert not is_hf_layout(tiny_esmfold2.state_dict())


def test_the_port_layout_is_recognised():
    assert is_hf_layout(HF_KEYS)


def test_an_unrecognised_layout_fails_loudly():
    """Guessing here would load a model of randomly initialised parameters."""
    with pytest.raises(ValueError, match="neither"):
        is_hf_layout({"totally.unrelated.tensor": None})


def test_our_layout_passes_through_untouched(tiny_esmfold2):
    """The path our own published checkpoints take must not move."""
    native = tiny_esmfold2.state_dict()
    assert set(EsmFold2Model._normalize_checkpoint_layout(native)) == set(native)


# ---------------------------------------------------------------------------
# The rules and the two structural ops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("theirs", "ours"), sorted(HF_KEYS.items()))
def test_each_rename_family(theirs, ours):
    assert to_native_key(theirs) == ours


def test_split_attention_projections_are_packed_again():
    """Upstream stores q/k/v separately; ours is one ``Wqkv``, in that row order."""
    base = "structure_head.atom_encoder.layers.0.self_attn."
    parts = {n: torch.randn(2, 3) for n in "qkv"}
    out = hf_state_dict_to_native(
        {f"{base}{n}_proj.weight": t for n, t in parts.items()}
    )

    (key,) = out
    assert key.endswith(".attn.Wqkv.weight")
    assert torch.equal(out[key], torch.cat([parts["q"], parts["k"], parts["v"]], dim=0))


def test_fused_gate_up_is_split_again():
    """The conditioning transitions are the only ones we keep unfused."""
    key = "structure_head.conditioning.pair_transition_0.mlp.gate_up_proj.weight"
    fused = torch.randn(4, 3)

    out = hf_state_dict_to_native({key: fused})

    prefix = "structure_head.diffusion_module.conditioning.z_transitions.0."
    assert torch.equal(out[prefix + "a_proj.weight"], fused[:2])
    assert torch.equal(out[prefix + "b_proj.weight"], fused[2:])


def test_a_tensor_that_finds_no_home_fails_loudly():
    """Two keys colliding on one destination is a silently halved model."""
    collide = {
        "input_embedder.rel_pos.embed.weight": torch.randn(2, 2),
        "rel_pos.embed.weight": torch.randn(2, 2),
    }
    with pytest.raises(ValueError, match="both map onto"):
        hf_state_dict_to_native(collide)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.fixture
def hf_config(tmp_path) -> str:
    """A re-exported port ``config.json``: HuggingFace's field names *and* ours.

    Upstream's ``AutoConfig`` absorbs unknown keys at every level, so one
    superset file is valid for both readers and neither side has to translate.
    """
    raw = {
        "hidden_size": 32,
        "pairwise_hidden_size": 16,
        "num_relative_residx_bins": 8,
        "n_relative_residx_bins": 8,
        "model_type": "esmfold2",
        "atom_encoder": {"hidden_size": 64, "head_dim": 16, "output_dim": 48},
        "msa_encoder": {"enabled": True, "head_dim": 16, "head_width": 16},
        "lm_encoder": {"enabled": True, "num_hidden_layers": 1},
        "parcae": {"enabled": True},
        "structure_head": {
            "num_distogram_bins": 39,
            "distogram_bins": 39,
            "inference_exponent": 8.0,
            "inference_p": 8.0,
            "diffusion_module": {
                "hidden_size": 96,
                "token_hidden_size": 96,
                "head_dim": 48,
            },
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    return str(tmp_path)


def test_an_overspecified_config_is_read_silently(hf_config, recwarn):
    """Upstream derives ``head_dim`` / ``intermediate_size`` and we recompute
    them, so its half of the superset is discarded. A ``config.json`` is not the
    caller's to fix and there is nothing to act on, so nothing is said."""
    config = EsmFold2Config.from_pretrained(hf_config)

    assert not hasattr(config.atom_encoder, "head_dim")
    assert config.atom_encoder.hidden_size == 64
    assert config.structure_head.distogram_bins == 39
    assert config.structure_head.diffusion_module.token_hidden_size == 96
    assert [w for w in recwarn if "config.json" in str(w.message)] == []


def test_an_underspecified_config_warns_and_defaults(hf_config):
    """A section named without ``enabled`` is a module the checkpoint has weights
    for, so it defaults on. Warned, not refused: the caller cannot edit the file,
    and the strict key accounting catches the reading if it is wrong."""
    path = Path(hf_config) / "config.json"
    raw = json.loads(path.read_text())
    del raw["msa_encoder"]["enabled"]
    path.write_text(json.dumps(raw))

    with pytest.warns(UserWarning, match="msa_encoder.enabled"):
        defaults = default_module_flags(hf_config)

    assert EsmFold2Config.from_pretrained(hf_config, **defaults).msa_encoder.enabled


def test_a_section_the_config_omits_entirely_still_defaults(hf_config, recwarn):
    """Nothing to warn about: a model with no MSA encoder is not underspecified."""
    path = Path(hf_config) / "config.json"
    raw = json.loads(path.read_text())
    del raw["msa_encoder"]
    path.write_text(json.dumps(raw))

    assert default_module_flags(hf_config) == {}
    assert not EsmFold2Config.from_pretrained(hf_config).msa_encoder.enabled
    assert [w for w in recwarn if "config.json" in str(w.message)] == []


# ---------------------------------------------------------------------------
# Keywords we own
# ---------------------------------------------------------------------------


def test_an_unknown_keyword_to_the_config_reader_raises(hf_config):
    """The mirror image of the rule above: an extra *key* is the file's business
    and ignored, but an extra *keyword* is the caller's own code doing nothing."""
    with pytest.raises(TypeError, match="hidden_sizee"):
        EsmFold2Config.from_pretrained(hf_config, hidden_sizee=32)


def test_an_unknown_keyword_to_the_model_reader_raises(hf_config):
    with pytest.raises(TypeError, match="some_knob"):
        EsmFold2Model.from_pretrained(hf_config, some_knob=3)


# ---------------------------------------------------------------------------
# Against the port itself
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not upstream_available(),
    reason="needs transformers >= 5.16 (the ESMFold2 port); this repo pins 4.x",
)
def test_rules_invert_their_converter(tiny_esmfold2_config):
    """The rules have to reproduce their converter key for key.

    Their ``rename_trunk_keys`` is the only source of truth for the mapping, so
    it generates the expectation here rather than a table being maintained by
    hand. When upstream moves a name, this fails naming the key that moved.
    """
    from transformers.models.esmfold2.convert_esmfold2_checkpoint import (
        rename_trunk_keys,
    )

    tiny_esmfold2_config.msa_encoder.enabled = True
    with torch.device("meta"):
        ours = set(EsmFold2Model(tiny_esmfold2_config).state_dict())
        theirs = rename_trunk_keys(
            dict(EsmFold2Model(tiny_esmfold2_config).state_dict())
        )

    got = set()
    for key in theirs:
        native = to_native_key(key)
        got.add(native)
        if native.endswith(".a_proj.weight"):
            got.add(native.replace(".a_proj.", ".b_proj."))
    assert got == ours


@pytest.mark.manual
def test_a_re_exported_port_checkpoint_loads(hf_port_checkpoint_dir):
    """End to end on the real artifact: ~26 GB, so not part of any suite.

    Its ``config.json`` states no module flags, so this also exercises the
    warn-and-default path against a checkpoint that really does carry the
    weights those defaults assume. ``_load_pretrained`` is strict, so reaching
    this line at all means every tensor found a parameter and every parameter
    was filled.
    """
    model = EsmFold2Model.from_pretrained(hf_port_checkpoint_dir, esmc_precision="fp32")

    assert model.esmc is not None
    assert json.loads(json.dumps(model.config.to_dict()))["hidden_size"] > 0
