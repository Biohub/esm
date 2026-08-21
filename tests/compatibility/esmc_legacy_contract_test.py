"""Pins the ESM 3.x public contract for ``ESMC``, as shipped at tag ``v3.2.3``.

Every assertion comes from ``v3.2.3`` (``23b084b``) of the public ``esm`` repo,
not from the current shim: a failure means a program written against 3.x broke,
which is a stricter question than whether the shim changed.
"""

import sys
import warnings

import attr
import pytest
import torch
import torch.nn as nn

from esm.models.esmc import ESMC, ESMCOutput
from esm.models.esmc.compatibility import _legacy_name_to_repo
from esm.models.esmc.config import EsmcConfig
from esm.models.esmc.model import EsmcForMaskedLM
from esm.sdk.api import ESMCInferenceClient, ESMProtein, LogitsConfig, LogitsOutput
from esm.utils.constants.models import ESMC_6B, ESMC_300M, ESMC_600M

# Deep enough that the hidden-state stack has a middle as well as two ends.
D_MODEL, N_HEADS, N_LAYERS = 32, 4, 4
SEQUENCES = ["AAAA", "MKV"]

V3_ESMCOUTPUT_FIELDS = ("sequence_logits", "embeddings", "hidden_states")

# Verbatim from the v3.2.3 README; do not reword, it is the thing under test.
V3_README_QUICKSTART = """\
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

protein = ESMProtein(sequence="AAAAA")
client = ESMC.from_pretrained("esmc_300m").to("cuda") # or "cpu"
protein_tensor = client.encode(protein)
logits_output = client.logits(
   protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
)
print(logits_output.logits, logits_output.embeddings)
"""


def tiny_native_model(**config_kwargs) -> EsmcForMaskedLM:
    torch.manual_seed(0)
    return EsmcForMaskedLM(
        EsmcConfig(
            hidden_size=D_MODEL,
            num_attention_heads=N_HEADS,
            num_hidden_layers=N_LAYERS,
            **config_kwargs,
        )
    ).eval()


def legacy_model(**kwargs) -> ESMC:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ESMC(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            use_flash_attn=False,
            **kwargs,
        ).eval()


def wrap(native: EsmcForMaskedLM) -> ESMC:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ESMC(model=native).eval()


def _zero_output() -> ESMCOutput:
    """Positional construction, as in v3; the field layout needs no forward pass."""
    return ESMCOutput(
        torch.zeros(1, 3, 64),
        torch.zeros(1, 3, D_MODEL),
        torch.zeros(N_LAYERS, 1, 3, D_MODEL),
    )


# --- README ---


@pytest.fixture
def published_esm_namespace(monkeypatch):
    """Expose this package as ``esm``, the name the sync publishes it under.

    In the monorepo ``esm`` is the unrelated ``fair-esm``, so the released
    layout is reproduced rather than rewriting the snippets under test.
    """
    import esm.models.esmc as _models_esmc
    import esm.sdk.api as _sdk_api

    monkeypatch.setitem(sys.modules, "esm.models.esmc", _models_esmc)
    monkeypatch.setitem(sys.modules, "esm.sdk.api", _sdk_api)


def test_v3_readme_quickstart_runs_verbatim_against_a_stand_in_checkpoint(
    monkeypatch, published_esm_namespace
):
    """As-is, bar two substitutions the README sanctions: cuda -> cpu, tiny model."""
    monkeypatch.setattr(
        ESMC,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: wrap(tiny_native_model())),
    )
    namespace: dict = {}
    exec(V3_README_QUICKSTART.replace('"cuda"', '"cpu"'), namespace)  # noqa: S102

    out = namespace["logits_output"]
    assert isinstance(out, LogitsOutput)
    assert out.logits is not None and out.logits.sequence is not None
    assert out.embeddings is not None
    # "AAAAA" plus BOS/EOS.
    assert out.logits.sequence.shape[:2] == (1, 7)
    assert out.embeddings.shape == (1, 7, D_MODEL)


@pytest.mark.nightly
def test_v3_readme_quickstart_runs_verbatim_on_the_published_esmc_300m(
    published_esm_namespace,
):
    """The same block with nothing substituted, against the real checkpoint."""
    namespace: dict = {}
    source = V3_README_QUICKSTART
    if not torch.cuda.is_available():
        source = source.replace('"cuda"', '"cpu"')
    exec(source, namespace)  # noqa: S102

    out = namespace["logits_output"]
    assert out.logits is not None and out.logits.sequence is not None
    assert out.embeddings is not None
    assert out.logits.sequence.shape[:2] == (1, 7)


# --- Type contracts ---


def test_esmc_is_still_an_esmc_inference_client():
    """3.x code guards on this before calling ``encode`` / ``logits``."""
    model = legacy_model()
    assert isinstance(model, ESMCInferenceClient)
    assert isinstance(model, nn.Module)


def test_esmc_output_is_an_attrs_class_so_asdict_and_evolve_work():
    out = _zero_output()
    assert attr.has(ESMCOutput)
    assert torch.equal(attr.asdict(out)["sequence_logits"], out.sequence_logits)
    assert attr.evolve(out, hidden_states=None).hidden_states is None


def test_esmc_output_field_names_and_order_match_v3():
    names = tuple(f.name for f in attr.fields(ESMCOutput))
    assert names[: len(V3_ESMCOUTPUT_FIELDS)] == V3_ESMCOUTPUT_FIELDS
    # Anything added since must be optional, or positional construction breaks.
    for field in attr.fields(ESMCOutput)[len(V3_ESMCOUTPUT_FIELDS) :]:
        assert field.default is not attr.NOTHING, f"{field.name} needs a default"
    # Not kw_only in v3: the three fields are passed positionally.
    assert isinstance(_zero_output(), ESMCOutput)


def test_esmc_output_hidden_states_is_still_assignable():
    """3.x ``ESMC.logits`` mutates ``output.hidden_states`` in place."""
    out = _zero_output()
    assert out.hidden_states is not None
    out.hidden_states = out.hidden_states[0:1]
    assert out.hidden_states.shape[0] == 1


# --- Attribute surface ---


def test_v3_attribute_surface_matches_v3():
    model = legacy_model()

    assert isinstance(model.embed, nn.Embedding)
    # 3.x: nn.Embedding(64, d_model).
    assert model.embed.num_embeddings == 64
    assert model.embed.embedding_dim == D_MODEL

    # 3.x indexed the block list directly.
    assert len(model.transformer.blocks) == N_LAYERS
    assert isinstance(model.transformer.blocks[0], nn.Module)

    assert isinstance(model.sequence_head, nn.Module)
    assert model.tokenizer.pad_token_id is not None


def test_assigning_a_forwarded_submodule_does_not_shadow_the_real_one():
    """Without the ``__setattr__`` override the write lands in ``ESMC._modules``,
    so reads keep the old module and every parameter is counted twice."""
    model = legacy_model()
    before = sum(p.numel() for p in model.parameters())
    replacement = nn.Embedding(64, D_MODEL)

    model.embed = replacement

    assert model.embed is replacement
    assert model.model.esmc.embed is replacement
    assert sum(p.numel() for p in model.parameters()) == before


# --- The documented SDK journey: encode -> logits -> decode ---


def test_logits_honours_ith_hidden_layer_as_in_the_cookbook_snippet():
    model = legacy_model()
    protein_tensor = model.encode(ESMProtein(sequence="AAAAA"))

    output = model.logits(
        protein_tensor, LogitsConfig(return_hidden_states=True, ith_hidden_layer=1)
    )
    assert output.hidden_states is not None
    assert output.hidden_states.shape[0] == 1

    every = model.logits(protein_tensor, LogitsConfig(return_hidden_states=True))
    assert every.hidden_states is not None
    assert torch.equal(output.hidden_states[0], every.hidden_states[1])


def test_logits_defaults_return_only_sequence_logits():
    model = legacy_model()
    output = model.logits(model.encode(ESMProtein(sequence="AAAAA")))
    assert output.embeddings is None
    assert output.hidden_states is None


def test_raw_forward_shapes_from_the_cookbook_snippet():
    """``cookbook/snippets/esmc.py`` at v3.2.3, ``raw_forward()``."""
    model = legacy_model()
    input_ids = model._tokenize([s for s in ("AAAAA", "AAAAA")])
    with torch.no_grad():
        output = model(input_ids)
    logits, embeddings, hiddens = (
        output.sequence_logits,
        output.embeddings,
        output.hidden_states,
    )
    assert logits.shape == (2, 7, 64)
    assert embeddings.shape == (2, 7, D_MODEL)
    assert hiddens.shape == (N_LAYERS, 2, 7, D_MODEL)


def test_forward_takes_sequence_tokens_positionally_as_in_v3():
    model = legacy_model()
    tokens = model._tokenize(SEQUENCES)
    with torch.no_grad():
        positional = model(tokens)
        by_keyword = model(sequence_tokens=tokens)
    assert torch.equal(positional.sequence_logits, by_keyword.sequence_logits)


# --- Numerical parity between the legacy path and the native one ---


def test_legacy_path_is_bit_exact_with_the_native_model():
    """The shim must be a renaming, not a second implementation."""
    native = tiny_native_model()
    model = wrap(native)
    tokens = model._tokenize(SEQUENCES)

    with torch.no_grad():
        legacy = model(sequence_tokens=tokens)
        modern = native(input_ids=tokens, output_hidden_states=True, return_dict=True)

    assert torch.equal(legacy.sequence_logits, modern.logits), "logits drifted"
    assert torch.equal(legacy.embeddings, modern.last_hidden_state), (
        "embeddings drifted"
    )

    # Legacy stack is per block output pre-LayerNorm; native is per block input.
    assert legacy.hidden_states.shape == (N_LAYERS, *tokens.shape, D_MODEL)
    assert modern.hidden_states.shape == (N_LAYERS + 1, *tokens.shape, D_MODEL)
    for i in range(N_LAYERS - 1):
        assert torch.equal(legacy.hidden_states[i], modern.hidden_states[i + 1]), (
            f"hidden state {i} drifted"
        )
    assert torch.equal(legacy.hidden_states[-1], modern.last_hidden_state_prenorm)


def test_legacy_logits_endpoint_is_bit_exact_with_the_native_model():
    """Also walks ``cookbook/snippets/esmc.py`` at v3.2.3, ``main()``."""
    native = tiny_native_model()
    model = wrap(native)
    protein_tensor = model.encode(ESMProtein(sequence="AAAAA"))

    output = model.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True),
    )
    assert protein_tensor.sequence is not None
    with torch.no_grad():
        modern = native(
            input_ids=protein_tensor.sequence[None],
            output_hidden_states=True,
            return_dict=True,
        )

    assert output.logits is not None and output.logits.sequence is not None
    assert output.embeddings is not None
    assert torch.equal(output.logits.sequence, modern.logits)
    assert torch.equal(output.embeddings, modern.last_hidden_state)
    assert output.hidden_states is not None
    assert output.hidden_states.shape[0] == N_LAYERS
    assert model.decode(protein_tensor).sequence == "AAAAA"


# --- Checkpoint resolution ---


@pytest.mark.parametrize(
    ("name", "repo"),
    [
        (ESMC_300M, "biohub/ESMC-300M"),
        (ESMC_600M, "biohub/ESMC-600M"),
        (ESMC_6B, "biohub/ESMC-6B"),
    ],
)
def test_v3_model_names_resolve_to_the_published_repos(name: str, repo: str):
    """``esmc_300m`` and friends are the strings 3.x users hard-coded."""
    assert _legacy_name_to_repo(name) == repo


def test_from_pretrained_defaults_to_esmc_600m_as_in_v3():
    import inspect

    default = inspect.signature(ESMC.from_pretrained).parameters["model_name"].default
    assert default == ESMC_600M


# --- Flash attention ---


def test_v3_default_use_flash_attn_true_degrades_quietly_without_flash_attn(
    monkeypatch,
):
    """``use_flash_attn=True`` is the 3.x default, so it must degrade quietly."""
    monkeypatch.setattr("esm.models.esmc.model.FLASH_ATTN_INSTALLED", False)
    monkeypatch.setattr("esm.models.esmc.layers.FLASH_ATTN_INSTALLED", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = ESMC(d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS).eval()

    assert model._use_flash_attn is False
    # A forwarding property, not a copy of the constructor argument.
    assert model._use_flash_attn == model.model.esmc._use_flash_attn
    tokens = model._tokenize(SEQUENCES)
    with torch.no_grad():
        out = model(sequence_tokens=tokens)
    assert out.sequence_logits.shape == (*tokens.shape, 64)
