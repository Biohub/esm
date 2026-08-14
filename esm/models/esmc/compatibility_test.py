"""Pins the deprecated ``ESMC`` surface so the compatibility wrapper can't rot."""

import warnings

import pytest
import torch

from esm.models.esmc import ESMC, ESMCOutput
from esm.models.esmc.compatibility import _legacy_name_to_repo
from esm.models.esmc.model import EsmcForMaskedLM
from esm.utils.constants.models import (
    ESMC_6B,
    ESMC_300M,
    ESMC_600M,
)

D_MODEL, N_HEADS, N_LAYERS = 32, 4, 2
SEQUENCES = ["AAAA", "MKV"]


def make_legacy_model() -> ESMC:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ESMC(
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, use_flash_attn=False
        ).eval()


def test_legacy_constructor_warns_and_wraps_native_model():
    with pytest.warns(DeprecationWarning):
        model = ESMC(
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, use_flash_attn=False
        )
    assert isinstance(model.model, EsmcForMaskedLM)


def test_legacy_forward_returns_old_field_names():
    model = make_legacy_model()
    tokens = model._tokenize(SEQUENCES)

    with torch.no_grad():
        out = model(sequence_tokens=tokens)

    assert isinstance(out, ESMCOutput)
    b, length = tokens.shape
    assert out.sequence_logits.shape == (b, length, 64)
    assert out.embeddings is not None
    assert out.embeddings.shape == (b, length, D_MODEL)
    # One entry per block output, as before the port.
    assert out.hidden_states is not None
    assert out.hidden_states.shape == (N_LAYERS, b, length, D_MODEL)


def test_tokenize_detokenize_round_trip():
    model = make_legacy_model()
    assert model._detokenize(model._tokenize(SEQUENCES)) == SEQUENCES


def test_boolean_sequence_id_is_treated_as_padding_mask():
    """The old ``sequence_id`` was a bool mask; that is ``attention_mask`` now.

    Passing it straight through as the native ``sequence_id`` would silently
    mark every position valid, since ``False >= 0`` is true.
    """
    model = make_legacy_model()
    tokens = model._tokenize(SEQUENCES)
    mask = tokens != model.tokenizer.pad_token_id

    with torch.no_grad():
        via_legacy = model(sequence_tokens=tokens, sequence_id=mask)
        via_native = model.model(
            input_ids=tokens, attention_mask=mask, output_hidden_states=True
        )

    assert via_legacy.embeddings is not None
    torch.testing.assert_close(via_legacy.embeddings, via_native.last_hidden_state)


def test_output_attentions_is_forwarded():
    model = make_legacy_model()
    tokens = model._tokenize(SEQUENCES)

    with torch.no_grad():
        assert model(sequence_tokens=tokens).attentions is None
        attns = model(sequence_tokens=tokens, output_attentions=True).attentions

    assert attns is not None
    assert len(attns) == N_LAYERS


def test_state_dict_keys_match_the_unwrapped_model():
    model = make_legacy_model()
    assert list(model.state_dict()) == list(model.model.state_dict())
    model.load_state_dict(model.state_dict())


def test_module_plumbing_reaches_the_wrapped_model():
    model = make_legacy_model()
    assert sum(p.numel() for p in model.parameters()) > 0
    assert model.device == torch.device("cpu")
    assert model.raw_model is model


@pytest.mark.parametrize("name", [ESMC_300M, ESMC_600M, ESMC_6B])
def test_legacy_model_names_map_to_hub_repos(name: str):
    assert _legacy_name_to_repo(name).startswith("biohub/ESMC-")


def test_unknown_name_passes_through_so_repo_ids_still_work():
    assert _legacy_name_to_repo("biohub/ESMC-300M") == "biohub/ESMC-300M"


def test_legacy_hidden_states_match_the_old_block_output_stack():
    """``main``'s stack held one entry per block *output*, before the final norm.

    The native stack follows the ``transformers`` convention instead - one entry
    per block *input* plus the normalised final state - so the wrapper has to
    realign it or every ``hidden_states[i]`` silently returns a different layer.
    """
    model = make_legacy_model()
    tokens = model._tokenize(SEQUENCES)
    with torch.no_grad():
        legacy = model(sequence_tokens=tokens)
        native = model.model(
            input_ids=tokens, output_hidden_states=True, return_dict=True
        )

    assert legacy.hidden_states is not None and native.hidden_states is not None
    assert legacy.hidden_states.shape[0] == N_LAYERS
    assert native.hidden_states.shape[0] == N_LAYERS + 1
    # Index alignment: block i's output is native[i + 1].
    for i in range(N_LAYERS - 1):
        assert torch.equal(legacy.hidden_states[i], native.hidden_states[i + 1])
    # The last entry is pre-LayerNorm, so it must differ from the normed output.
    assert not torch.allclose(legacy.hidden_states[-1], legacy.embeddings)


def test_legacy_output_supports_native_access_patterns():
    model = make_legacy_model()
    tokens = model._tokenize(SEQUENCES)
    with torch.no_grad():
        out = model(sequence_tokens=tokens)
    assert torch.equal(out.logits, out.sequence_logits)
    assert torch.equal(out["logits"], out.sequence_logits)


def test_legacy_forward_accepts_input_ids():
    """So callers need not branch on the model class to pick a calling style."""
    model = make_legacy_model()
    tokens = model._tokenize(SEQUENCES)
    with torch.no_grad():
        by_alias = model(input_ids=tokens)
        by_legacy_name = model(sequence_tokens=tokens)
    assert torch.equal(by_alias["logits"], by_legacy_name.sequence_logits)
    with pytest.raises(ValueError, match="not both"):
        model(sequence_tokens=tokens, input_ids=tokens)


def test_retired_date_stamped_factories_are_still_importable():
    """They shipped publicly as ``esm.pretrained.ESMC_*_202412``."""
    from esm import pretrained

    for name in ("ESMC_300M_202412", "ESMC_600M_202412", "ESMC_6B_202412"):
        assert callable(getattr(pretrained, name))
    for model_name in (ESMC_300M, ESMC_600M, ESMC_6B):
        assert model_name in pretrained.LOCAL_MODEL_REGISTRY
