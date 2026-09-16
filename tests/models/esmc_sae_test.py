"""ESMC SAE tests, focused on which activation each SAE is fed.

An SAE handed the wrong activation still returns exactly ``k`` non-zeros with
the right shape and raises nothing - it just reconstructs badly. So every test
here asserts on the signed values of the tensor the SAE actually receives, not
on its shape or sparsity, which are identical either way.
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

from esm.models.esmc import EsmcForMaskedLM, EsmcModel, EsmcSaeModel, EsmcTokenizer
from esm.models.esmc.sae import CONFIG_NAME, EsmcSaeConfig, EsmcSaeLayer, EsmcSaeParams
from tests.conftest import ESMC_300M_REPO, TINY_ESMC, TINY_SEQUENCES

# Big enough that top-k is a real selection, small enough to build instantly.
TINY_CODEBOOK_DIM = 16
TINY_K = 2

FLAG = "use_residual_update_instead_of_states"

# The published ESMC-300M SAEs for the layer the sweep targeted.
PUBLISHED_LAYER = 23
RESIDUAL_UPDATE_REPO = "biohub/ESMC-300M-sae-mlp-k64-codebook131072"
STATES_REPO = "biohub/ESMC-300M-sae-layer23-k64-codebook131072"

#: A published SAE ``config.json`` as it exists today: no flag in any spelling.
LEGACY_SAE_CONFIG = {
    "d_model": 960,
    "codebook_dim": 131072,
    "k": 64,
    "available_layers": [PUBLISHED_LAYER],
    "model_type": "esmc_sae",
}


def tiny_sae_layer(layer: int, use_residual_update: bool) -> EsmcSaeLayer:
    """A randomly-initialised SAE sized for the tiny ESMC fixtures."""
    torch.manual_seed(layer)
    params = EsmcSaeParams(
        d_model=TINY_ESMC["hidden_size"],
        codebook_dim=TINY_CODEBOOK_DIM,
        k=TINY_K,
        layer=layer,
        use_residual_update_instead_of_states=use_residual_update,
    )
    sae = EsmcSaeLayer(params)
    nn.init.normal_(sae.W_enc, std=0.02)
    nn.init.normal_(sae.W_dec, std=0.02)
    return sae.eval()


def sae_layer_of(model: EsmcSaeModel, layer: int) -> EsmcSaeLayer:
    """``model.layers`` is an ``nn.ModuleDict``, so it is typed ``Module``."""
    loaded = model.layers[str(layer)]
    assert isinstance(loaded, EsmcSaeLayer)
    return loaded


def write_sae_config(directory, raw: dict) -> None:
    (directory / CONFIG_NAME).write_text(json.dumps(raw))


def hidden_states_of(model: EsmcModel, tokenizer) -> torch.Tensor:
    enc = tokenizer(TINY_SEQUENCES, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, compute_sae=False)
    return out.hidden_states


def run_with_saes(model: EsmcModel, tokenizer) -> None:
    enc = tokenizer(TINY_SEQUENCES, return_tensors="pt", padding=True)
    with torch.no_grad():
        model(**enc)


@pytest.fixture
def captured_sae_inputs(monkeypatch):
    """Record, per layer, the tensor each SAE is handed."""
    captured: dict[int, torch.Tensor] = {}
    original = EsmcSaeLayer.get_sae_output

    def spy(self, layer_states, token_mask):
        captured[self.layer] = layer_states.detach().clone()
        return original(self, layer_states, token_mask)

    monkeypatch.setattr(EsmcSaeLayer, "get_sae_output", spy)
    return captured


# ---------------------------------------------------------------------------
# Declaring the flag
# ---------------------------------------------------------------------------


def test_the_flag_defaults_to_states():
    """The 97 repos published without it rely on this default."""
    assert EsmcSaeConfig().use_residual_update_instead_of_states is False
    assert EsmcSaeParams().use_residual_update_instead_of_states is False


def test_a_config_without_the_flag_reads_as_states(tmp_path):
    write_sae_config(tmp_path, LEGACY_SAE_CONFIG)
    config = EsmcSaeConfig.from_pretrained(tmp_path)
    assert config.use_residual_update_instead_of_states is False


@pytest.mark.parametrize("flag", [False, True])
def test_the_flag_round_trips_through_the_config(tmp_path, flag):
    EsmcSaeConfig(
        d_model=8,
        codebook_dim=16,
        k=2,
        available_layers=[1],
        use_residual_update_instead_of_states=flag,
    ).save_pretrained(tmp_path)
    assert json.loads((tmp_path / CONFIG_NAME).read_text())[FLAG] is flag
    assert (
        EsmcSaeConfig.from_pretrained(tmp_path).use_residual_update_instead_of_states
        is flag
    )


@pytest.mark.parametrize("flag", [False, True])
def test_the_flag_reaches_the_loaded_layer(tmp_path, flag):
    """``_get_sae_outputs`` reads it off the layer, not off the container."""
    config = EsmcSaeConfig(
        d_model=TINY_ESMC["hidden_size"],
        codebook_dim=TINY_CODEBOOK_DIM,
        k=TINY_K,
        available_layers=[1],
        use_residual_update_instead_of_states=flag,
    )
    model = EsmcSaeModel(config)
    model.layers["1"] = tiny_sae_layer(1, flag)
    model.save_pretrained(tmp_path)

    reloaded = EsmcSaeModel.from_pretrained(tmp_path)
    reloaded.initialize_layers([1])
    layer = sae_layer_of(reloaded, 1)
    assert layer.params.use_residual_update_instead_of_states is flag


# ---------------------------------------------------------------------------
# What the backbone feeds each SAE
# ---------------------------------------------------------------------------


def test_a_states_sae_is_fed_the_plain_state(
    tiny_esmc, esmc_tokenizer, captured_sae_inputs
):
    reference = hidden_states_of(tiny_esmc, esmc_tokenizer)
    tiny_esmc.add_sae_models([tiny_sae_layer(1, False)])
    run_with_saes(tiny_esmc, esmc_tokenizer)
    torch.testing.assert_close(captured_sae_inputs[1], reference[1])


def test_a_residual_update_sae_is_fed_the_difference(
    tiny_esmc, esmc_tokenizer, captured_sae_inputs
):
    """The bug: this SAE used to be handed ``h[1]`` instead of ``h[1] - h[0]``."""
    reference = hidden_states_of(tiny_esmc, esmc_tokenizer)
    tiny_esmc.add_sae_models([tiny_sae_layer(1, True)])
    run_with_saes(tiny_esmc, esmc_tokenizer)
    fed = captured_sae_inputs[1]
    torch.testing.assert_close(fed, reference[1] - reference[0])
    assert not torch.allclose(fed, reference[1])


def test_the_difference_runs_in_the_training_direction(
    tiny_esmc, esmc_tokenizer, captured_sae_inputs
):
    """Layer 2, so ``N-2`` is a valid index and an off-by-one is observable."""
    reference = hidden_states_of(tiny_esmc, esmc_tokenizer)
    tiny_esmc.add_sae_models([tiny_sae_layer(2, True)])
    run_with_saes(tiny_esmc, esmc_tokenizer)
    fed = captured_sae_inputs[2]
    torch.testing.assert_close(fed, reference[2] - reference[1])
    # Reversed subtraction, an off-by-one predecessor, and no subtraction at
    # all all produce a correctly-shaped tensor with the same sparsity.
    assert not torch.allclose(fed, reference[1] - reference[2])
    assert not torch.allclose(fed, reference[2] - reference[0])
    assert not torch.allclose(fed, reference[2])


def test_a_residual_update_sae_at_layer_zero_is_fed_the_raw_state(
    tiny_esmc, esmc_tokenizer, captured_sae_inputs
):
    """``include_first_state=True`` in training: index 0 is not a difference."""
    reference = hidden_states_of(tiny_esmc, esmc_tokenizer)
    tiny_esmc.add_sae_models([tiny_sae_layer(0, True)])
    run_with_saes(tiny_esmc, esmc_tokenizer)
    fed = captured_sae_inputs[0]
    torch.testing.assert_close(fed, reference[0])
    # Rules out the other plausible layer-0 conventions: a zero tensor, or a
    # state differenced against itself.
    assert fed.abs().max() > 0


def test_each_sae_gets_its_own_tensor(tiny_esmc, esmc_tokenizer, captured_sae_inputs):
    """Mixed flags in one model: neither SAE may see the other's activation."""
    reference = hidden_states_of(tiny_esmc, esmc_tokenizer)
    tiny_esmc.add_sae_models([tiny_sae_layer(1, False), tiny_sae_layer(2, True)])
    enc = esmc_tokenizer(TINY_SEQUENCES, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = tiny_esmc(**enc)
    torch.testing.assert_close(captured_sae_inputs[1], reference[1])
    torch.testing.assert_close(captured_sae_inputs[2], reference[2] - reference[1])
    assert set(out.sae_outputs) == {"layer1", "layer2"}


def test_each_residual_update_sae_gets_its_own_difference(
    tiny_esmc, esmc_tokenizer, captured_sae_inputs
):
    """Guards a difference computed once and reused across SAEs."""
    reference = hidden_states_of(tiny_esmc, esmc_tokenizer)
    tiny_esmc.add_sae_models([tiny_sae_layer(1, True), tiny_sae_layer(2, True)])
    run_with_saes(tiny_esmc, esmc_tokenizer)
    torch.testing.assert_close(captured_sae_inputs[1], reference[1] - reference[0])
    torch.testing.assert_close(captured_sae_inputs[2], reference[2] - reference[1])
    assert not torch.allclose(captured_sae_inputs[1], captured_sae_inputs[2])


def test_two_saes_cannot_share_a_backbone_layer(tiny_esmc):
    """So one layer can never carry two disagreeing flags."""
    with pytest.raises(ValueError, match="already registered"):
        tiny_esmc.add_sae_models([tiny_sae_layer(1, True), tiny_sae_layer(1, False)])


# ---------------------------------------------------------------------------
# Collecting the predecessor
# ---------------------------------------------------------------------------


def test_a_residual_update_sae_pulls_in_its_predecessor_layer(tiny_esmc):
    tiny_esmc.add_sae_models([tiny_sae_layer(2, True)])
    assert tiny_esmc._sae_layers_to_collect() == [1, 2]


def test_a_states_sae_collects_only_its_own_layer(tiny_esmc):
    tiny_esmc.add_sae_models([tiny_sae_layer(2, False)])
    assert tiny_esmc._sae_layers_to_collect() == [2]


def test_layer_zero_needs_no_predecessor(tiny_esmc):
    tiny_esmc.add_sae_models([tiny_sae_layer(0, True)])
    assert tiny_esmc._sae_layers_to_collect() == [0]


def test_a_missing_predecessor_raises_instead_of_reconstructing_the_state(tiny_esmc):
    """Rather than quietly falling back to ``h[N]``, which is the original bug."""
    tiny_esmc.add_sae_models([tiny_sae_layer(1, True)])
    states = torch.randn(1, 2, 4, TINY_ESMC["hidden_size"])
    token_mask = torch.ones(2, 4, dtype=torch.bool)
    with pytest.raises(KeyError, match="residual updates"):
        tiny_esmc._get_sae_outputs(states, [1], token_mask)


def test_forward_raises_when_collection_drops_the_predecessor(
    tiny_esmc, esmc_tokenizer, monkeypatch
):
    """Pins the two halves together: collection and differencing must agree."""
    tiny_esmc.add_sae_models([tiny_sae_layer(2, True)])
    monkeypatch.setattr(tiny_esmc, "_sae_layers_to_collect", lambda: [2])
    enc = esmc_tokenizer(TINY_SEQUENCES, return_tensors="pt", padding=True)
    with pytest.raises(KeyError, match="residual updates"), torch.no_grad():
        tiny_esmc(**enc)


# ---------------------------------------------------------------------------
# Published weights
# ---------------------------------------------------------------------------

# From the README quickstart; the sequence the numbers below were measured on.
GFP = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMK"
    "QHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQK"
    "NGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)


def fraction_of_variance_unexplained(
    sae: EsmcSaeLayer, activation: torch.Tensor
) -> float:
    """Reconstruction error of ``sae`` on ``activation``, in units of its variance."""
    normalized = sae._zscore_normalize_representation(activation)
    out = sae(activation)
    assert out.reconstruction_loss is not None
    return float(out.reconstruction_loss.mean() / normalized.var(dim=-1).mean())


def published_sae_with_flag(tmp_path, repo: str, flag: bool) -> EsmcSaeModel:
    """One layer of a published repo, re-declared with ``flag``.

    Stands in for the re-export: same weights, ``config.json`` rewritten
    locally so no Hub write is needed.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    shard = f"layer_{PUBLISHED_LAYER}.safetensors"
    snapshot = Path(snapshot_download(repo, allow_patterns=[CONFIG_NAME, shard]))
    (tmp_path / shard).symlink_to(snapshot / shard)
    raw = json.loads((snapshot / CONFIG_NAME).read_text())
    raw["available_layers"] = [PUBLISHED_LAYER]
    raw[FLAG] = flag
    write_sae_config(tmp_path, raw)
    return EsmcSaeModel.from_pretrained(tmp_path)


@pytest.mark.manual
def test_the_flag_fixes_reconstruction_on_published_weights(tmp_path):
    """The measurement behind this change, on ~2 GB of published weights.

    ``manual`` because it downloads an ESMC-300M checkpoint plus one 1 GB SAE
    shard from each of two repos.
    """
    backbone = EsmcModel.from_pretrained(ESMC_300M_REPO, device="cpu").eval()
    enc = EsmcTokenizer()(GFP, return_tensors="pt")
    with torch.no_grad():
        states = backbone(**enc, output_hidden_states=True).hidden_states
    state = states[PUBLISHED_LAYER]
    update = state - states[PUBLISHED_LAYER - 1]

    residual = sae_layer_of(
        published_sae_with_flag(tmp_path / "residual", RESIDUAL_UPDATE_REPO, True),
        PUBLISHED_LAYER,
    )
    control = sae_layer_of(
        published_sae_with_flag(tmp_path / "states", STATES_REPO, False),
        PUBLISHED_LAYER,
    )
    with torch.no_grad():
        on_update = fraction_of_variance_unexplained(residual, update)
        on_state = fraction_of_variance_unexplained(residual, state)
        on_control = fraction_of_variance_unexplained(control, state)

    assert on_update < on_state
    assert on_update == pytest.approx(0.1787, abs=0.02)
    assert on_state == pytest.approx(0.2258, abs=0.02)
    assert on_control == pytest.approx(0.0360, abs=0.02)


@pytest.mark.manual
def test_a_published_residual_update_repo_is_fed_the_difference(tmp_path):
    """End to end on real weights, through ``add_sae_models``."""
    model = EsmcForMaskedLM.from_pretrained(ESMC_300M_REPO, device="cpu").eval()
    enc = EsmcTokenizer()(GFP, return_tensors="pt")
    with torch.no_grad():
        states = model(**enc, output_hidden_states=True).hidden_states

    sae = published_sae_with_flag(tmp_path, RESIDUAL_UPDATE_REPO, True)
    assert sae.config.use_residual_update_instead_of_states is True
    captured: dict[int, torch.Tensor] = {}
    original = EsmcSaeLayer.get_sae_output

    def spy(self, layer_states, token_mask):
        captured[self.layer] = layer_states.detach().clone()
        return original(self, layer_states, token_mask)

    EsmcSaeLayer.get_sae_output = spy  # ty:ignore[invalid-assignment]
    try:
        model.add_sae_models([sae_layer_of(sae, PUBLISHED_LAYER)])
        with torch.no_grad():
            model(**enc)
    finally:
        EsmcSaeLayer.get_sae_output = original

    expected = states[PUBLISHED_LAYER] - states[PUBLISHED_LAYER - 1]
    torch.testing.assert_close(captured[PUBLISHED_LAYER], expected)
