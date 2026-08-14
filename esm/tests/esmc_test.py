"""ESMC test suite.

Structural tests run on CPU against tiny randomly-initialised models. Tests that
need published weights use ESMC-300M and skip when the Hub is unreachable. GPU
tests are marked ``gpu``.
"""

import functools
import gzip
import json
import math
import pickle
import warnings
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from esm.models.esmc import (
    ESMC,
    EsmcConfig,
    EsmcForMaskedLM,
    EsmcForSequenceClassification,
    EsmcMaskedLMOutput,
    EsmcModel,
    EsmcOutput,
    EsmcTokenizer,
)
from esm.models.esmc import model as model_module
from esm.models.esmc.checkpoint_layout import (
    native_to_published,
    published_to_native,
)
from esm.models.esmc.kernels import (
    FLASH_ATTN_INSTALLED,
    TE_INSTALLED,
)
from esm.models.esmc.layers import (
    EsmcFlashMultiHeadAttention,
    EsmcLayerNormLinear,
    EsmcLayerNormMLP,
    EsmcRotaryEmbedding,
)
from esm.models.hub import CONFIG_NAME, read_safetensors_dir
from esm.tests.conftest import (
    LENGTH_NAMES,
    MEDIUM_SEQUENCE,
    REFERENCE_DIR,
    SEQUENCES,
    SHORT_SEQUENCE,
    TINY_ESMC,
    TINY_SEQUENCES,
    random_amino_acid_sequences,
)

REFERENCE_FILE = REFERENCE_DIR / "esmc_300m_cpu_fp32.pkl.gz"

# Each masked position costs one forward, so an uncapped pseudo-perplexity is
# O(L^2) tokens. 16 evenly spaced positions hold every length to a few seconds.
PPL_MAX_POSITIONS = 16

# Which layers ``build_esmc`` puts in the model.
FUSED_LAYERS = "fused"
REFERENCE_LAYERS = "reference"


def _tokens(tokenizer, sequences):
    return tokenizer(sequences, return_tensors="pt", padding=True)


@functools.lru_cache(maxsize=1)
def reference_values() -> dict:
    # A missing file is a broken checkout, not a reason to skip.
    assert REFERENCE_FILE.exists(), (
        f"{REFERENCE_FILE} is missing. It is checked in; regenerate with "
        f"`python -m esm.tests.regenerate_reference`."
    )
    with gzip.open(REFERENCE_FILE, "rb") as f:
        return pickle.load(f)


class PseudoPerplexity(NamedTuple):
    perplexity: float
    #: Fraction of masked residues the model's argmax recovers.
    recovery: float


def masked_token_positions(
    n_residues: int, limit: int = PPL_MAX_POSITIONS
) -> list[int]:
    """Token indices to mask: every residue, or ``limit`` evenly spaced ones."""
    # Token 0 is <cls> and the last is <eos>, so residue i is token i + 1.
    if n_residues <= limit:
        return list(range(1, n_residues + 1))
    return [1 + (i * n_residues) // limit for i in range(limit)]


def pseudo_perplexity(
    model,
    tokenizer,
    sequence: str,
    *,
    limit: int = PPL_MAX_POSITIONS,
    batch_size: int = 8,
) -> PseudoPerplexity:
    """Masked-LM pseudo-perplexity and top-1 recovery over the masked positions.

    Row ``i`` of a batch masks one residue, so one forward scores ``batch_size``
    positions. Rows are equal-length and never attend to each other, so batching
    cannot change a row's result beyond reduction order.
    """
    enc = tokenizer(sequence, return_tensors="pt")
    ids = enc["input_ids"][0]
    positions = masked_token_positions(len(sequence), limit)

    total_logprob = 0.0
    recovered = 0
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        batch = ids.unsqueeze(0).repeat(len(chunk), 1).to(model.device)
        for row, position in enumerate(chunk):
            batch[row, position] = model.config.mask_token_id

        with torch.no_grad():
            logits = model(input_ids=batch).logits

        for row, position in enumerate(chunk):
            row_logits = logits[row, position].float().cpu()
            target = int(ids[position])
            total_logprob += float(torch.log_softmax(row_logits, dim=-1)[target])
            recovered += int(int(row_logits.argmax()) == target)

    return PseudoPerplexity(
        perplexity=math.exp(-total_logprob / len(positions)),
        recovery=recovered / len(positions),
    )


def build_esmc(
    directory,
    *,
    device: str,
    dtype: torch.dtype | None = None,
    attn: str = "sdpa",
    layers: str = FUSED_LAYERS,
    model_class=EsmcModel,
):
    """Load ESMC with an explicit choice of fused vs pure-PyTorch layers.

    ``EsmcModel.__init__`` reads ``torch.get_default_device()`` to pick its
    kernels, so building for the CPU and then moving is the only way to get the
    reference layers onto CUDA. That also disables the flash-attention block, so
    ``layers="reference"`` is only meaningful with ``attn="sdpa"`` / ``"eager"``.
    """
    build_device = "cpu" if layers == REFERENCE_LAYERS else device
    model = model_class.from_pretrained(
        directory, device=build_device, attn_implementation=attn
    )
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
    return model.eval()


def module_output_shapes(model: nn.Module, **inputs) -> dict[str, tuple[int, ...]]:
    """Shape of every module's (first) tensor output during one forward.

    Modules returning a dataclass are skipped; the caller asserts their fields.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    handles = []

    def record(name: str):
        def hook(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                shapes[name] = tuple(tensor.shape)

        return hook

    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(record(name)))
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return shapes


def argmax_mismatches(logits: torch.Tensor, expected: list[int]) -> int:
    """How many positions predict a different token than the reference."""
    predicted = logits.float().argmax(-1)[0].tolist()
    assert len(predicted) == len(expected)
    return sum(a != b for a, b in zip(predicted, expected))


def drop_keys(directory: Path, predicate) -> list[str]:
    """Rewrite a saved checkpoint without the keys matching ``predicate``."""
    from safetensors import safe_open

    with safe_open(str(directory / "model.safetensors"), framework="pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    dropped = [k for k in tensors if predicate(k)]
    assert dropped, f"predicate matched nothing among {sorted(tensors)[:5]}"
    save_file(
        {k: v.contiguous() for k, v in tensors.items() if k not in set(dropped)},
        str(directory / "model.safetensors"),
        metadata={"format": "pt"},
    )
    return dropped


def safetensors_keys(directory) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(directory / "model.safetensors"), framework="pt") as f:
        return list(f.keys())


def read_config(directory: Path) -> dict:
    return json.loads((directory / CONFIG_NAME).read_text())


def write_config(directory: Path, raw: dict) -> None:
    (directory / CONFIG_NAME).write_text(json.dumps(raw))


# ---------------------------------------------------------------------------
# Construction and the forward shape contract
# ---------------------------------------------------------------------------


def test_construction(tiny_esmc, tiny_esmc_mlm, tiny_esmc_config):
    config = tiny_esmc_config
    assert len(tiny_esmc.transformer.blocks) == config.num_hidden_layers
    assert tiny_esmc.embed.weight.shape == (config.vocab_size, config.hidden_size)
    for block in tiny_esmc.transformer.blocks:
        assert block.attn.n_heads == config.num_attention_heads

    assert isinstance(tiny_esmc_mlm.esmc, EsmcModel)
    assert tiny_esmc_mlm.lm_head[-1].out_features == config.vocab_size


def test_forward_shape_contract(tiny_esmc, tiny_esmc_mlm, esmc_tokenizer):
    enc = _tokens(esmc_tokenizer, TINY_SEQUENCES)
    b, length = enc["input_ids"].shape

    with torch.no_grad():
        out = tiny_esmc(**enc, output_hidden_states=True, output_attentions=True)
        mlm = tiny_esmc_mlm(**enc, output_hidden_states=True)

    assert isinstance(out, EsmcOutput)
    assert out.last_hidden_state.shape == (b, length, TINY_ESMC["hidden_size"])
    # One entry per block input plus the final post-LayerNorm output.
    assert out.hidden_states.shape == (
        TINY_ESMC["num_hidden_layers"] + 1,
        b,
        length,
        TINY_ESMC["hidden_size"],
    )
    assert len(out.attentions) == TINY_ESMC["num_hidden_layers"]

    assert isinstance(mlm, EsmcMaskedLMOutput)
    assert mlm.logits.shape == (b, length, TINY_ESMC["vocab_size"])
    assert mlm.last_hidden_state.shape == (b, length, TINY_ESMC["hidden_size"])


@pytest.mark.parametrize("name", LENGTH_NAMES)
def test_every_module_output_has_the_expected_shape(
    tiny_esmc_mlm, esmc_tokenizer, tiny_esmc_config, name
):
    """Pin the width of every intermediate, not just the model's output.

    A wrong QKV fan-out or FFN width still produces a correctly-shaped final
    tensor.
    """
    config = tiny_esmc_config
    hidden = config.hidden_size
    heads = config.num_attention_heads
    enc = esmc_tokenizer(SEQUENCES[name], return_tensors="pt")
    b, length = enc["input_ids"].shape

    expected: dict[str, tuple[int, ...]] = {
        "esmc.embed": (b, length, hidden),
        "esmc.transformer": (b, length, hidden),
        "esmc.transformer.norm": (b, length, hidden),
        # Linear -> GELU -> LayerNorm -> Linear, so only the last widens.
        "lm_head": (b, length, config.vocab_size),
        "lm_head.0": (b, length, hidden),
        "lm_head.1": (b, length, hidden),
        "lm_head.2": (b, length, hidden),
        "lm_head.3": (b, length, config.vocab_size),
    }
    for layer in range(config.num_hidden_layers):
        block = f"esmc.transformer.blocks.{layer}"
        expected.update(
            {
                block: (b, length, hidden),
                f"{block}.attn": (b, length, hidden),
                # One fused projection for Q, K and V.
                f"{block}.attn.layernorm_qkv": (b, length, 3 * hidden),
                f"{block}.attn.q_ln": (b, length, hidden),
                f"{block}.attn.k_ln": (b, length, hidden),
                # RoPE runs on the unflattened heads.
                f"{block}.attn.rotary": (b, length, heads, config.head_dim),
                f"{block}.attn.out_proj": (b, length, hidden),
                f"{block}.ffn": (b, length, hidden),
            }
        )

    # Equality, not containment: a new sub-module has to be added here too.
    assert module_output_shapes(tiny_esmc_mlm, **enc) == expected

    # EsmcLayerNormMLP has no child modules, so the SwiGLU widths are not
    # observable through a hook. fc1 is doubled for the gate and value halves.
    ffn = tiny_esmc_mlm.esmc.transformer.blocks[0].ffn
    assert ffn.fc1_weight.shape == (2 * config.intermediate_size, hidden)
    assert ffn.fc2_weight.shape == (hidden, config.intermediate_size)

    with torch.no_grad():
        out = tiny_esmc_mlm(**enc, output_attentions=True)
    for attention in out.attentions:
        assert attention.shape == (b, heads, length, length)


def test_gradients_reach_every_parameter(tiny_esmc_config, esmc_tokenizer):
    """Fine-tuning is a supported use, so the backward pass has to work."""
    torch.manual_seed(0)
    model = EsmcForMaskedLM(tiny_esmc_config).train()
    batch = _tokens(esmc_tokenizer, TINY_SEQUENCES)

    model(**batch).logits.square().mean().backward()

    gradients = {name: p.grad for name, p in model.named_parameters()}
    assert not [name for name, grad in gradients.items() if grad is None]
    # A dead parameter has a gradient of exactly zero everywhere.
    assert not [
        name
        for name, grad in gradients.items()
        if grad is not None and (not torch.isfinite(grad).all() or grad.eq(0).all())
    ]


def test_sequence_classification_head(tiny_esmc_config, esmc_tokenizer):
    """The head reads ``<cls>``, so it must emit one row per sequence."""
    torch.manual_seed(0)
    config = replace(tiny_esmc_config, num_labels=3)
    model = EsmcForSequenceClassification(config).eval()
    batch = _tokens(esmc_tokenizer, TINY_SEQUENCES)
    labels = torch.tensor([0, 2])

    with torch.no_grad():
        out = model(**batch, labels=labels)

    assert out.logits.shape == (len(TINY_SEQUENCES), 3)
    assert out.loss is not None and torch.isfinite(out.loss)
    # A label-independent head would score both rows identically.
    assert not torch.allclose(out.logits[0], out.logits[1])


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


def test_config_round_trip(tmp_path, tiny_esmc_config):
    """Every architectural field survives ``save_pretrained`` / ``from_pretrained``.

    Goes through the real ``save_pretrained``: ``to_dict`` deliberately drops the
    run-scoped fields, so writing the dataclass verbatim would assert they
    persist when in fact they must not.
    """
    tiny_esmc_config.save_pretrained(tmp_path)
    reloaded = EsmcConfig.from_pretrained(tmp_path)

    for key in tiny_esmc_config.to_dict():
        if key == "model_type":
            continue
        assert getattr(reloaded, key) == getattr(tiny_esmc_config, key), key


def test_attn_implementation_is_not_persisted(tmp_path, tiny_esmc_config):
    """It describes the run, not the weights, so it must not survive a save."""
    replace(tiny_esmc_config, attn_implementation="eager").save_pretrained(tmp_path)

    written = read_config(tmp_path)
    assert "attn_implementation" not in written
    default = EsmcConfig(hidden_size=1, num_attention_heads=1).attn_implementation
    assert EsmcConfig.from_pretrained(tmp_path).attn_implementation == default


def test_config_tolerates_unknown_fields(tmp_path):
    """A newer published config.json must not break an older reader.

    ``EsmcConfig.from_pretrained`` reads known fields by name, so unknown ones
    are ignored rather than retained as attributes.
    """
    write_config(tmp_path, {**TINY_ESMC, "some_future_field": 123})
    cfg = EsmcConfig.from_pretrained(tmp_path)
    assert cfg.hidden_size == TINY_ESMC["hidden_size"]
    assert not hasattr(cfg, "some_future_field")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenizer_round_trips_the_whole_alphabet(esmc_tokenizer):
    canonical = "ACDEFGHIKLMNPQRSTVWY"
    # X/B/U/Z/O and the gap and insertion tokens are real vocabulary entries.
    extended = canonical + "XBUZO.-"

    ids = esmc_tokenizer(extended)["input_ids"]
    assert ids == esmc_tokenizer(extended)["input_ids"]
    assert esmc_tokenizer.unk_token_id not in ids
    assert len(ids) == len(extended) + 2
    assert esmc_tokenizer.decode(torch.tensor(ids[1:-1])).replace(" ", "") == extended

    # Lowercase is not in the vocabulary and silently becomes <unk>.
    assert set(esmc_tokenizer(canonical.lower())["input_ids"][1:-1]) == {
        esmc_tokenizer.unk_token_id
    }


def test_tokenizer_pads_to_longest(esmc_tokenizer):
    enc = _tokens(esmc_tokenizer, TINY_SEQUENCES)
    longest = max(len(s) for s in TINY_SEQUENCES) + 2

    assert enc["input_ids"].shape == (len(TINY_SEQUENCES), longest)
    for row, sequence in enumerate(TINY_SEQUENCES):
        keep = len(sequence) + 2
        assert enc["attention_mask"][row].tolist() == [1] * keep + [0] * (
            longest - keep
        )
        assert (enc["input_ids"][row, keep:] == esmc_tokenizer.pad_token_id).all()


def test_the_chain_break_token_survives_tokenisation(esmc_tokenizer):
    """``|`` separates chains within one sequence, so it has to reach the model."""
    ids = esmc_tokenizer("AAAA|MKV", return_tensors="pt")["input_ids"][0]
    break_id = esmc_tokenizer.convert_tokens_to_ids("|")

    assert break_id is not None
    assert (ids == break_id).sum() == 1
    # Position 0 is <cls>, so the break lands after the first chain's 4 residues.
    assert int((ids == break_id).nonzero()[0, 0]) == 5


# ---------------------------------------------------------------------------
# Determinism and the RoPE cache, the only length-dependent state in the model
# ---------------------------------------------------------------------------


def test_eval_forward_is_deterministic(tiny_esmc, esmc_tokenizer):
    enc = _tokens(esmc_tokenizer, TINY_SEQUENCES)
    with torch.no_grad():
        a = tiny_esmc(**enc).last_hidden_state
        b = tiny_esmc(**enc).last_hidden_state
    assert torch.equal(a, b)


def _fill_rope_cache(rope: EsmcRotaryEmbedding, length: int):
    q = torch.zeros(1, length, 1, rope.dim)
    rope(q, q)
    assert rope._cos_cached is not None and rope._sin_cached is not None
    return rope._cos_cached.clone(), rope._sin_cached.clone()


def test_rope_cache_grows_to_the_longest_sequence_seen(tiny_esmc_config):
    """Growing the table must extend it, not re-derive it on a new grid.

    ``_update_cos_sin_cache`` rebuilds only when asked for more positions, so the
    rows it already holds have to stay correct for every shorter sequence.
    """
    rope = EsmcRotaryEmbedding(tiny_esmc_config.head_dim)
    assert rope._seq_len_cached == 0

    short_cos, short_sin = _fill_rope_cache(rope, 12)
    assert rope._seq_len_cached == 12

    long_cos, long_sin = _fill_rope_cache(rope, 412)
    assert rope._seq_len_cached == 412
    assert torch.equal(long_cos[:12], short_cos)
    assert torch.equal(long_sin[:12], short_sin)

    # A shorter sequence reuses the long table rather than shrinking it.
    _fill_rope_cache(rope, 12)
    assert rope._seq_len_cached == 412


@pytest.mark.parametrize(
    "warmup,measured", [("short", "long"), ("long", "short"), ("long", "medium")]
)
def test_a_forward_does_not_depend_on_the_length_run_before_it(
    tiny_esmc, tiny_esmc_config, esmc_tokenizer, warmup, measured
):
    """Compared against a freshly-built model, not against itself, so a stale or
    wrongly-extended RoPE cache cannot hide by being wrong both times.
    """
    enc = esmc_tokenizer(SEQUENCES[measured], return_tensors="pt")
    with torch.no_grad():
        tiny_esmc(**esmc_tokenizer(SEQUENCES[warmup], return_tensors="pt"))
        warm = tiny_esmc(**enc).last_hidden_state

    # Same seed as the ``tiny_esmc`` fixture, so the weights are identical.
    torch.manual_seed(0)
    cold = EsmcModel(tiny_esmc_config).eval()
    with torch.no_grad():
        expected = cold(**enc).last_hidden_state
    assert torch.equal(warm, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name", LENGTH_NAMES)
def test_cpu_dtypes(tiny_esmc_config, esmc_tokenizer, dtype, name):
    torch.manual_seed(0)
    model = EsmcModel(tiny_esmc_config).eval().to(dtype)
    # Batched against a short sequence so every length also carries padding.
    enc = _tokens(esmc_tokenizer, [SEQUENCES[name], SHORT_SEQUENCE])
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    assert out.dtype == dtype
    assert torch.isfinite(out.float()).all()


# ---------------------------------------------------------------------------
# Published-weight loading and the key policy
# ---------------------------------------------------------------------------


def _checkpoint_state(directory) -> dict[str, torch.Tensor]:
    """The published checkpoint, in the model's own (fused) tensor layout."""
    raw = published_to_native(read_safetensors_dir(directory))
    return {k: v for k, v in raw.items() if not k.endswith("_extra_state")}


@pytest.mark.parametrize("model_class", [EsmcForMaskedLM, EsmcModel])
def test_every_state_dict_entry_comes_from_the_checkpoint(esmc_300m_dir, model_class):
    """No parameter may keep the value ``init_empty_weights`` left behind.

    Bit-equality against the file on disk is the strongest form of this: a key the
    checkpoint never covered holds whatever ``to_empty`` found in memory.
    ``EsmcModel`` drops ``lm_head.*`` and strips the ``esmc.`` prefix, which is
    what the key-set comparison encodes.
    """
    model = model_class.from_pretrained(esmc_300m_dir, device="cpu")
    checkpoint = _checkpoint_state(esmc_300m_dir)
    if model_class is EsmcModel:
        checkpoint = {
            k.removeprefix("esmc."): v
            for k, v in checkpoint.items()
            if k.startswith("esmc.")
        }

    state = model.state_dict()
    assert not [k for k, t in state.items() if t.is_meta]
    assert set(state) == set(checkpoint)
    unpopulated = [k for k, t in state.items() if not torch.equal(t, checkpoint[k])]
    assert not unpopulated, unpopulated


def test_rope_buffers_are_materialised_after_loading(esmc_300m):
    """The one thing no checkpoint can cover.

    ``inv_freq`` is registered non-persistently, so it is absent from both the
    checkpoint and ``state_dict``; it still has to come off the meta device.
    """
    rotaries = [m for m in esmc_300m.modules() if isinstance(m, EsmcRotaryEmbedding)]
    assert len(rotaries) == esmc_300m.config.num_hidden_layers
    for rope in rotaries:
        assert not rope.inv_freq.is_meta
        assert torch.isfinite(rope.inv_freq).all()


def test_hidden_state_and_attention_shapes_on_real_weights(esmc_300m, esmc_tokenizer):
    """The per-layer contract at the published width, not just the tiny one.

    Short sequence only: the attention tensor is
    ``n_layers x batch x heads x L x L``.
    """
    config = esmc_300m.config
    enc = esmc_tokenizer(SHORT_SEQUENCE, return_tensors="pt")
    length = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = esmc_300m(**enc, output_hidden_states=True, output_attentions=True)

    assert out.logits.shape == (1, length, config.vocab_size)
    assert out.last_hidden_state.shape == (1, length, config.hidden_size)
    assert out.last_hidden_state_prenorm.shape == (1, length, config.hidden_size)
    assert out.hidden_states.shape == (
        config.num_hidden_layers + 1,
        1,
        length,
        config.hidden_size,
    )
    assert len(out.attentions) == config.num_hidden_layers
    for attention in out.attentions:
        assert attention.shape == (1, config.num_attention_heads, length, length)
        # Every row is a distribution over the keys it is allowed to see.
        torch.testing.assert_close(
            attention.sum(-1), torch.ones_like(attention.sum(-1)), atol=1e-5, rtol=0
        )


# ---------------------------------------------------------------------------
# Config reading rules and the published tensor layout
# ---------------------------------------------------------------------------


def test_reads_pre_alignment_config(tmp_path):
    """Checkpoints published before the field-name alignment must still resolve."""
    write_config(
        tmp_path,
        {
            "vocab_size": 64,
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 3,
            "pad_token_id": 1,
            "mask_token_id": 32,
        },
    )
    with pytest.warns(FutureWarning, match="pre-alignment"):
        cfg = EsmcConfig.from_pretrained(tmp_path)
    assert (cfg.hidden_size, cfg.num_attention_heads, cfg.num_hidden_layers) == (
        64,
        4,
        3,
    )


@pytest.mark.parametrize(
    "missing", ["vocab_size", "hidden_size", "num_attention_heads", "num_hidden_layers"]
)
def test_missing_architecture_field_raises(tmp_path, missing):
    """Defaulting an architecture field would build the wrong model silently."""
    raw = {**TINY_ESMC}
    del raw[missing]
    write_config(tmp_path, raw)
    with pytest.raises(KeyError, match=missing):
        EsmcConfig.from_pretrained(tmp_path)


def test_rejects_inconsistent_intermediate_size():
    with pytest.raises(ValueError, match="intermediate_size"):
        EsmcConfig(hidden_size=32, num_attention_heads=4, intermediate_size=999)


def test_honours_attn_implementation_from_config(tmp_path, tiny_esmc_config):
    """It gates the flash-attention block, so a config that sets it must be read."""
    EsmcForMaskedLM(tiny_esmc_config).save_pretrained(tmp_path)
    raw = read_config(tmp_path)
    raw["attn_implementation"] = "eager"
    write_config(tmp_path, raw)

    assert EsmcConfig.from_pretrained(tmp_path).attn_implementation == "eager"
    loaded = EsmcForMaskedLM.from_pretrained(tmp_path, device="cpu")
    explicit = EsmcForMaskedLM.from_pretrained(
        tmp_path, device="cpu", attn_implementation="sdpa"
    )
    assert loaded.config.attn_implementation == "eager"
    # An explicit argument still wins.
    assert explicit.config.attn_implementation == "sdpa"


def test_published_layout_round_trip_is_bit_identical(tiny_esmc_config):
    """Fusing on load and splitting on save must compose to the identity.

    A fused weight is the row-wise concatenation of its parts, so any drift means
    a checkpoint written by one implementation is read wrongly by the other.
    """
    model = EsmcForMaskedLM(tiny_esmc_config)
    native = model.state_dict()
    published = native_to_published(native)

    # The published layout is one tensor per projection.
    for suffix in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"):
        assert any(suffix in k for k in published), suffix
    assert not any("layernorm_qkv" in k or "fc1_weight" in k for k in published)
    assert any(k.startswith("esmc.layers.0.") for k in published)

    back = published_to_native(published)
    assert set(back) == set(native)
    assert all(torch.equal(back[k], native[k]) for k in native)


def test_save_pretrained_round_trips_through_published_layout(
    tmp_path, tiny_esmc_config, esmc_tokenizer
):
    """A full save -> load cycle must not move the logits at all."""
    model = EsmcForMaskedLM(tiny_esmc_config).eval()
    batch = _tokens(esmc_tokenizer, TINY_SEQUENCES)
    with torch.no_grad():
        before = model(**batch).logits

    model.save_pretrained(tmp_path)
    saved = set(safetensors_keys(tmp_path))
    assert any("self_attn.q_proj" in k for k in saved)
    assert not any("layernorm_qkv" in k for k in saved)

    reloaded = EsmcForMaskedLM.from_pretrained(tmp_path, device="cpu")
    with torch.no_grad():
        after = reloaded(**batch).logits
    assert torch.equal(before, after)


def test_an_incomplete_checkpoint_is_refused(tmp_path, tiny_esmc_config):
    """Strict loading is what turns an uncovered parameter into an error.

    Drop one block and the load must refuse rather than silently keep whatever
    ``to_empty`` left in memory.
    """
    EsmcForMaskedLM(tiny_esmc_config).save_pretrained(tmp_path)
    drop_keys(tmp_path, lambda k: ".1." in k and ("layers" in k or "blocks" in k))

    with pytest.raises(RuntimeError, match="missing keys"):
        EsmcForMaskedLM.from_pretrained(tmp_path, device="cpu")


def test_a_fresh_classification_head_is_the_only_excused_gap(
    tmp_path, tiny_esmc_config
):
    """``_keys_to_ignore_on_load_missing`` must excuse the head and nothing else.

    Loading a backbone checkpoint into a classification model is the intended
    fine-tuning entry point, so an absent ``classifier.*`` cannot raise. Both
    sides of that hole are pinned: the head may be missing, a missing backbone
    tensor must still refuse.
    """
    EsmcForMaskedLM(tiny_esmc_config).save_pretrained(tmp_path)

    model = EsmcForSequenceClassification.from_pretrained(tmp_path, device="cpu")
    assert not any(p.is_meta for p in model.classifier.parameters())
    assert all(torch.isfinite(p).all() for p in model.classifier.parameters())

    drop_keys(tmp_path, lambda k: k.endswith("embed_tokens.weight"))
    with pytest.raises(RuntimeError, match="missing keys"):
        EsmcForSequenceClassification.from_pretrained(tmp_path, device="cpu")


# ---------------------------------------------------------------------------
# Numerical reference values on ESMC-300M (CPU, fp32)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", LENGTH_NAMES)
def test_reference_logits(esmc_300m_cpu_reference, name):
    case = reference_values()["cases"][name]
    logits = esmc_300m_cpu_reference[name]["logits"]
    hidden = esmc_300m_cpu_reference[name]["last_hidden_state"]

    assert list(logits.shape) == case["logits_shape"]
    # Bit-identical across reruns on one host; the tolerance leaves room for a
    # different CPU BLAS.
    torch.testing.assert_close(
        logits[0, 1, : len(case["first_residue_logits"])],
        torch.tensor(case["first_residue_logits"]),
        atol=1e-3,
        rtol=1e-3,
    )
    assert logits.argmax(-1)[0].tolist() == case["argmax"]
    # Per-position L2 norms of the post-LayerNorm output: O(1) in magnitude, so
    # one tolerance is meaningful at every dtype, and position-sensitive in a way
    # the argmax over a 64-token vocabulary is not.
    torch.testing.assert_close(
        hidden[0].norm(dim=-1),
        torch.tensor(case["last_hidden_state_norms"]),
        atol=1e-3,
        rtol=1e-4,
    )


@pytest.mark.parametrize("name", LENGTH_NAMES)
def test_reference_pseudo_perplexity(esmc_300m, esmc_tokenizer, name):
    """The model's own loss: the cheapest number that moves when anything in the
    stack changes numerically *and* has a scale worth reading.
    """
    case = reference_values()["cases"][name]
    measured = pseudo_perplexity(esmc_300m, esmc_tokenizer, SEQUENCES[name])
    assert measured.perplexity == pytest.approx(case["pseudo_perplexity"], rel=1e-3)
    # Integer count over a fixed position set, so this is exact.
    assert measured.recovery == case["top1_recovery"]

    if name == "long":
        # The long fixture is a tandem repeat of two domains, so every masked
        # residue has an exact copy ~205 positions away and full recovery is only
        # reachable if attention spans the whole sequence. Asserted as a value,
        # not against the stored reference: attention confined to a local window
        # would still produce a stable reference.
        assert measured.recovery == 1.0
        assert measured.perplexity < 1.1


@pytest.mark.parametrize("name", ["medium", "long"])
def test_real_sequences_score_better_than_random_ones(esmc_300m, esmc_tokenizer, name):
    """The check a stored reference cannot make: the numbers have to be *good*.

    A model with its weights loaded into the wrong slots still produces stable
    logits and a reproducible perplexity - it just cannot tell a real protein
    from a uniformly random one. The short sequence is excluded: 10 residues
    carry too little context for the gap to mean anything.
    """
    sequence = SEQUENCES[name]
    scrambled = random_amino_acid_sequences([len(sequence)], seed=7)[0]
    real = pseudo_perplexity(esmc_300m, esmc_tokenizer, sequence, limit=8)
    noise = pseudo_perplexity(esmc_300m, esmc_tokenizer, scrambled, limit=8)

    # Observed on ESMC-300M over 8 masked positions: real ppl 2.26 / recovery
    # 0.75 vs random 32.6 / 0.00 (medium), 1.02 / 1.00 vs 24.0 / 0.00 (long).
    # The thresholds sit an order of magnitude inside those margins.
    assert real.perplexity < 0.5 * noise.perplexity
    assert real.recovery >= 0.5
    assert noise.recovery <= 0.25


@pytest.mark.parametrize("name", LENGTH_NAMES)
def test_cpu_bf16_reaches_the_fp32_reference(esmc_300m_cpu_bf16, esmc_tokenizer, name):
    """bf16 cannot reproduce the reference logits, so pin what it must preserve:
    the decision the model makes and its loss, not the logits themselves.
    """
    case = reference_values()["cases"][name]
    enc = esmc_tokenizer(SEQUENCES[name], return_tensors="pt")
    with torch.no_grad():
        out = esmc_300m_cpu_bf16(**enc)

    # bf16 keeps 8 mantissa bits, so a near-tie between two residues can flip;
    # exact equality is not available. Observed 0 / 2 / 0 mismatches.
    assert argmax_mismatches(out.logits, case["argmax"]) <= 3
    # Observed max|Δ| on the hidden-state norms (~1.1-1.5): under 0.009.
    torch.testing.assert_close(
        out.last_hidden_state.float()[0].norm(dim=-1),
        torch.tensor(case["last_hidden_state_norms"]),
        atol=0.03,
        rtol=0,
    )
    measured = pseudo_perplexity(esmc_300m_cpu_bf16, esmc_tokenizer, SEQUENCES[name])
    # Observed relative shift vs the fp32 reference: under 6e-3.
    assert measured.perplexity == pytest.approx(case["pseudo_perplexity"], rel=2e-2)


# ---------------------------------------------------------------------------
# Masking schemes: padded, unpadded, multi-chain, and -1 as padding
# ---------------------------------------------------------------------------


def test_pad_tokens_cannot_influence_real_positions(tiny_esmc, esmc_tokenizer):
    """Overwriting the padding must not move a single valid output."""
    batch = _tokens(esmc_tokenizer, ["MQIFVKTLTGKT", "MKV"])
    keep = batch["attention_mask"].bool()

    with torch.no_grad():
        before = tiny_esmc(**batch).last_hidden_state

    scrambled = batch["input_ids"].clone()
    scrambled[~keep] = 7  # any residue token, where padding used to be
    with torch.no_grad():
        after = tiny_esmc(
            input_ids=scrambled, attention_mask=batch["attention_mask"]
        ).last_hidden_state

    for row in range(keep.shape[0]):
        length = int(keep[row].sum())
        torch.testing.assert_close(
            before[row, :length], after[row, :length], atol=0, rtol=0
        )


def test_padding_attends_to_the_real_tokens(tiny_esmc, esmc_tokenizer):
    """The one assertion that separates a key-axis mask from a query==key one.

    Padding hides a key from every query, so a pad *query* still attends to the
    real tokens and its output moves when they change.
    """
    batch = _tokens(esmc_tokenizer, ["MQIFVKTLTGKT", "MKV"])
    keep = batch["attention_mask"].bool()
    padded_row = next(
        row for row in range(keep.shape[0]) if int(keep[row].sum()) < keep.shape[1]
    )
    tail = slice(int(keep[padded_row].sum()), None)

    changed = batch["input_ids"].clone()
    changed[padded_row, 1] = 7 if changed[padded_row, 1] != 7 else 8

    with torch.no_grad():
        before = tiny_esmc(**batch).last_hidden_state[padded_row, tail]
        after = tiny_esmc(
            input_ids=changed, attention_mask=batch["attention_mask"]
        ).last_hidden_state[padded_row, tail]

    assert not torch.allclose(before, after, atol=1e-6)


@pytest.mark.parametrize("name", LENGTH_NAMES)
def test_padded_batch_matches_solo_runs_at_every_length(
    tiny_esmc, esmc_tokenizer, name
):
    """Padding a batch must not perturb the unpadded rows.

    The most direct check that attention masking is wired correctly, held at
    every length regime: the padded tail here is hundreds of tokens long.
    """
    sequences = [SEQUENCES[name], SHORT_SEQUENCE, MEDIUM_SEQUENCE[:7]]
    batch = _tokens(esmc_tokenizer, sequences)
    keep = batch["attention_mask"].bool()

    with torch.no_grad():
        padded = tiny_esmc(**batch).last_hidden_state
        for row, sequence in enumerate(sequences):
            alone = tiny_esmc(
                **esmc_tokenizer(sequence, return_tensors="pt")
            ).last_hidden_state[0]
            length = int(keep[row].sum())
            # Not exact: the padded batch takes SDPA's masked path and the solo
            # run its unmasked one, so the reduction orders differ.
            torch.testing.assert_close(
                padded[row, :length], alone[:length], atol=1e-5, rtol=1e-4
            )


def test_padded_batch_matches_solo_runs_across_random_lengths(
    tiny_esmc, esmc_tokenizer, random_sequences
):
    """Same invariant over ten seeded lengths, including a single residue.

    A one-residue sequence in a 410-wide batch is the extreme case: 411 of its
    412 key positions are padding.
    """
    batch = _tokens(esmc_tokenizer, random_sequences)
    keep = batch["attention_mask"].bool()

    with torch.no_grad():
        padded = tiny_esmc(**batch).last_hidden_state
        for row, sequence in enumerate(random_sequences):
            alone = tiny_esmc(
                **esmc_tokenizer(sequence, return_tensors="pt")
            ).last_hidden_state[0]
            length = int(keep[row].sum())
            torch.testing.assert_close(
                padded[row, :length], alone[:length], atol=1e-5, rtol=1e-4
            )


@pytest.mark.parametrize("name", LENGTH_NAMES)
def test_the_unmasked_fast_path_agrees_with_the_masked_one(
    tiny_esmc, esmc_tokenizer, name
):
    """Equal-length rows carry no padding, so ``forward`` drops the mask entirely
    and the fused kernels can take over. That fast path has to agree with the
    masked one, reached here by handing the same batch a single-chain
    ``sequence_id`` - every position attends everywhere either way.
    """
    sequence = SEQUENCES[name]
    batch = esmc_tokenizer([sequence, sequence[::-1]], return_tensors="pt")
    assert batch["attention_mask"].bool().all()
    one_chain = torch.zeros_like(batch["input_ids"])

    with torch.no_grad():
        unmasked = tiny_esmc(**batch).last_hidden_state
        masked = tiny_esmc(
            input_ids=batch["input_ids"], sequence_id=one_chain
        ).last_hidden_state

    # Observed exactly equal, but which kernel SDPA picks is not contractual.
    torch.testing.assert_close(unmasked, masked, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("name", ["medium", "long"])
def test_multi_chain_sequence_id_isolates_chains(tiny_esmc, esmc_tokenizer, name):
    """Chains must not see each other, at lengths where each chain is real.

    Not compared against running each chain alone: RoPE positions are absolute,
    so a chain's representation legitimately depends on where it sits in the
    concatenated input. What must hold is that perturbing one chain leaves the
    others exactly where they were.
    """
    ids = esmc_tokenizer(SEQUENCES[name], return_tensors="pt")["input_ids"]
    length = ids.shape[1]
    bounds = [0, length // 3, 2 * length // 3, length]
    sequence_id = torch.zeros(1, length, dtype=torch.long)
    for chain, start in enumerate(bounds[:-1]):
        sequence_id[:, start : bounds[chain + 1]] = chain

    middle = slice(bounds[1], bounds[2])
    changed = ids.clone()
    changed[:, middle] = 7  # any residue token, uniformly

    with torch.no_grad():
        before = tiny_esmc(input_ids=ids, sequence_id=sequence_id).last_hidden_state
        after = tiny_esmc(input_ids=changed, sequence_id=sequence_id).last_hidden_state

    outside = torch.ones(length, dtype=torch.bool)
    outside[middle] = False
    # Bit-identical: masked attention weights are exactly zero, so the middle
    # chain contributes 0 * v to every other chain's context.
    assert torch.equal(before[0, outside], after[0, outside])
    # And the perturbation has to do something inside its own chain, or this
    # would pass with attention switched off entirely.
    assert not torch.allclose(before[0, middle], after[0, middle])


def test_chain_break_and_chain_ids_describe_the_same_split(tiny_esmc, esmc_tokenizer):
    """Ties the two ways of expressing multiple chains together: the ``|`` token
    marks the boundary, the ``sequence_id`` enforces it.
    """
    ids = esmc_tokenizer("AAAA|MKV", return_tensors="pt")["input_ids"]
    break_at = int(
        (ids[0] == esmc_tokenizer.convert_tokens_to_ids("|")).nonzero()[0, 0]
    )
    sequence_id = torch.zeros_like(ids)
    sequence_id[:, break_at + 1 :] = 1

    with torch.no_grad():
        isolated = tiny_esmc(
            input_ids=ids, sequence_id=sequence_id
        ).last_hidden_state.clone()
        # Perturbing the second chain must not move the first.
        perturbed = ids.clone()
        perturbed[0, -2] = 7 if perturbed[0, -2] != 7 else 8
        after = tiny_esmc(
            input_ids=perturbed, sequence_id=sequence_id
        ).last_hidden_state

    torch.testing.assert_close(
        isolated[:, : break_at + 1], after[:, : break_at + 1], atol=0, rtol=0
    )
    assert not torch.allclose(isolated[:, break_at + 1 :], after[:, break_at + 1 :])


@pytest.mark.gpu
def test_multi_chain_under_flash_attention_2_raises(tiny_esmc_config, esmc_tokenizer):
    """The varlen flash path cannot express a block-diagonal chain mask, and the
    model documents a ValueError for it; a silent wrong answer would be worse.
    """
    if not FLASH_ATTN_INSTALLED:
        pytest.skip("flash-attn is not installed")  # ty:ignore[too-many-positional-arguments]

    with torch.device("cuda"):
        model = EsmcForMaskedLM(
            replace(tiny_esmc_config, attn_implementation="flash_attention_2")
        ).eval()
    assert model.esmc._use_flash_attn, "flash attention did not actually dispatch"

    ids = esmc_tokenizer("MQIFVKTLTGKT", return_tensors="pt")["input_ids"].cuda()
    sequence_id = torch.zeros_like(ids)
    sequence_id[:, ids.shape[1] // 2 :] = 1

    with pytest.raises(ValueError, match="Multi-chain"):
        model(input_ids=ids, sequence_id=sequence_id)


@pytest.mark.parametrize("name", LENGTH_NAMES)
def test_sequence_id_of_minus_one_masks_padding_like_attention_mask(
    tiny_esmc, esmc_tokenizer, name
):
    """``sequence_id == -1`` and ``attention_mask == 0`` hide the same keys.

    They do not build the same mask: ``sequence_id`` compares query against key,
    so a *pad* query ends up attending only to other padding, while an
    ``attention_mask`` hides padding from every query. Only the valid positions
    can be required to agree - which is the half that feeds anything downstream.
    """
    batch = _tokens(esmc_tokenizer, [SEQUENCES[name], SHORT_SEQUENCE])
    keep = batch["attention_mask"].bool()
    sequence_id = torch.where(keep, 0, -1)

    with torch.no_grad():
        via_mask = tiny_esmc(**batch).last_hidden_state
        via_ids = tiny_esmc(
            input_ids=batch["input_ids"], sequence_id=sequence_id
        ).last_hidden_state

    torch.testing.assert_close(via_mask[keep], via_ids[keep], atol=1e-5, rtol=1e-4)


def test_attention_mask_folds_into_a_multi_chain_sequence_id(tiny_esmc, esmc_tokenizer):
    """Both arguments at once: the mask is folded in, not dropped.

    A caller with chains *and* ragged lengths supplies a ``sequence_id`` that
    describes only the chains, leaving padding to ``attention_mask``. Folding it
    in has to equal marking the padding ``-1`` by hand.
    """
    batch = _tokens(esmc_tokenizer, [MEDIUM_SEQUENCE, SHORT_SEQUENCE])
    keep = batch["attention_mask"].bool()
    assert not keep.all(), "fixture must actually pad for this to test anything"

    length = batch["input_ids"].shape[1]
    chains = torch.zeros_like(batch["input_ids"])
    chains[:, length // 2 :] = 1
    folded_by_hand = torch.where(keep, chains, -1)

    with torch.no_grad():
        folded = tiny_esmc(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            sequence_id=chains,
        ).last_hidden_state
        by_hand = tiny_esmc(
            input_ids=batch["input_ids"], sequence_id=folded_by_hand
        ).last_hidden_state

    torch.testing.assert_close(folded[keep], by_hand[keep], atol=0, rtol=0)

    # And the fold must not have cost the chain isolation it was folded into.
    changed = batch["input_ids"].clone()
    tail = length // 2
    changed[:, tail] = 7 if int(changed[0, tail]) != 7 else 8
    with torch.no_grad():
        after = tiny_esmc(
            input_ids=changed,
            attention_mask=batch["attention_mask"],
            sequence_id=chains,
        ).last_hidden_state
    head = keep.clone()
    head[:, tail:] = False
    torch.testing.assert_close(folded[head], after[head], atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Kernel selection follows the target device
# ---------------------------------------------------------------------------


def _pretend_installed(monkeypatch) -> None:
    """Patch where the flags are read, not where they are defined."""
    monkeypatch.setattr(model_module, "TE_INSTALLED", True)
    monkeypatch.setattr(model_module, "FLASH_ATTN_INSTALLED", True)


@pytest.mark.parametrize("attn", ["sdpa", "flash_attention_2"])
def test_cpu_builds_never_get_cuda_only_kernels(
    monkeypatch, tiny_esmc_config, esmc_tokenizer, attn
):
    """Every direct constructor has to reach CPU-safe layers *and* return usable
    numbers, with the CUDA-only kernels pretending to be importable.
    """
    _pretend_installed(monkeypatch)
    config = replace(tiny_esmc_config, attn_implementation=attn)
    batch = _tokens(esmc_tokenizer, TINY_SEQUENCES)

    bare = EsmcModel(config).eval()
    mlm = EsmcForMaskedLM(config).eval()
    for encoder in (bare, mlm.esmc):
        block = encoder.transformer.blocks[0]
        assert not encoder._use_flash_attn
        assert not isinstance(block.attn, EsmcFlashMultiHeadAttention)
        assert isinstance(block.attn.layernorm_qkv, EsmcLayerNormLinear)  # ty:ignore[unresolved-attribute]
        assert isinstance(block.ffn, EsmcLayerNormMLP)

    with torch.no_grad():
        assert torch.isfinite(bare(**batch).last_hidden_state).all()
        assert torch.isfinite(mlm(**batch).logits).all()


def test_the_deprecated_constructor_runs_on_cpu(monkeypatch, esmc_tokenizer):
    _pretend_installed(monkeypatch)
    batch = _tokens(esmc_tokenizer, TINY_SEQUENCES)

    with pytest.warns(DeprecationWarning):
        legacy = ESMC(d_model=32, n_heads=4, n_layers=2).eval()
    with torch.no_grad():
        out = legacy(sequence_tokens=batch["input_ids"])
    assert torch.isfinite(out.sequence_logits).all()


def test_eager_and_sdpa_build_the_same_attention(tiny_esmc_config):
    """``attn_implementation`` only selects the flash block, so "sdpa" and "eager"
    produce an identical module tree and comparing their outputs proves nothing.
    Pinned structurally instead, so a future divergence has to be deliberate.
    """
    builds = [
        EsmcModel(replace(tiny_esmc_config, attn_implementation=attn))
        for attn in ("sdpa", "eager")
    ]
    first, second = (model.transformer.blocks[0].attn for model in builds)
    assert type(first) is type(second)
    assert not builds[0]._use_flash_attn and not builds[1]._use_flash_attn


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_state_dict_round_trips_under_transformer_engine(tiny_esmc_config):
    """TE adds ``_extra_state`` to ``state_dict`` and a hook strips it, so the
    loader has to be told not to expect it back."""
    if not TE_INSTALLED:
        pytest.skip("TransformerEngine unavailable")  # ty:ignore[too-many-positional-arguments]
    with torch.device("cuda"):
        model = EsmcForMaskedLM(tiny_esmc_config).eval()

    assert not [k for k in model.state_dict() if k.endswith("_extra_state")]
    model.load_state_dict(model.state_dict())


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("name", LENGTH_NAMES)
def test_gpu_dtypes(tiny_esmc_config, esmc_tokenizer, dtype, name):
    torch.manual_seed(0)
    model = EsmcModel(tiny_esmc_config).eval().to("cuda", dtype)
    # Batched against a short sequence so every length also carries padding.
    enc = {
        k: v.cuda()
        for k, v in _tokens(esmc_tokenizer, [SEQUENCES[name], SHORT_SEQUENCE]).items()
    }
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    assert out.dtype == dtype
    assert torch.isfinite(out.float()).all()


@pytest.mark.gpu
def test_fp8_backbone(esmc_300m_dir):
    """ESMC is commonly run fp8 as a folding backbone."""
    if not TE_INSTALLED:
        pytest.skip("TransformerEngine unavailable")  # ty:ignore[too-many-positional-arguments]
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("fp8 needs compute capability >= 9.0 (Hopper)")  # ty:ignore[too-many-positional-arguments]

    from esm.models.esmfold2.model import (
        _convert_te_modules_to_fp8_inplace,
    )

    model = EsmcModel.from_pretrained(
        esmc_300m_dir, device="cuda", dtype=torch.bfloat16
    )
    with torch.no_grad():
        _convert_te_modules_to_fp8_inplace(model)
    tokenizer = EsmcTokenizer()
    enc = {
        k: v.cuda() for k, v in tokenizer(SHORT_SEQUENCE, return_tensors="pt").items()
    }
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    assert torch.isfinite(out.float()).all()


@pytest.mark.gpu
def test_flash_attention_2_dispatches_and_agrees_with_sdpa(
    esmc_300m_dir, esmc_tokenizer
):
    """The only backend that is a distinct build, so the only one worth comparing.

    Dispatch is asserted first: ``from_pretrained`` silently falls back to sdpa
    when flash-attn is missing, so without this the test could pass having
    compared sdpa against itself.
    """
    if not FLASH_ATTN_INSTALLED:
        pytest.skip("flash-attn is not installed")  # ty:ignore[too-many-positional-arguments]

    reference = EsmcModel.from_pretrained(
        esmc_300m_dir, device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    candidate = EsmcModel.from_pretrained(
        esmc_300m_dir,
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    assert candidate._use_flash_attn, "flash_attention_2 silently fell back"
    assert isinstance(candidate.transformer.blocks[0].attn, EsmcFlashMultiHeadAttention)
    assert not isinstance(
        reference.transformer.blocks[0].attn, EsmcFlashMultiHeadAttention
    )

    for sequence in SEQUENCES.values():
        enc = {
            k: v.cuda()
            for k, v in esmc_tokenizer(sequence, return_tensors="pt").items()
        }
        with torch.no_grad():
            expected = reference(**enc).last_hidden_state.float()
            actual = candidate(**enc).last_hidden_state.float()
        # The flash path packs the batch and runs flash_attn_varlen where sdpa
        # runs xformers, so the reduction orders differ and the gap compounds
        # over 30 blocks. Held to a few bf16 ULP at activation scale.
        torch.testing.assert_close(
            actual, expected, atol=5 * torch.finfo(torch.bfloat16).eps, rtol=1e-2
        )


@pytest.mark.gpu
def test_cpu_inference_works_on_a_cuda_machine(esmc_300m_dir, esmc_tokenizer):
    """Transformer Engine is selected whenever it imports and CUDA exists, but its
    modules reject CPU inputs, so a CPU-targeted load has to opt out of them.
    """
    model = EsmcModel.from_pretrained(esmc_300m_dir, device="cpu")
    enc = esmc_tokenizer(SHORT_SEQUENCE, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    assert out.device.type == "cpu"
    assert torch.isfinite(out).all()


@pytest.mark.gpu
def test_cpu_and_gpu_agree(esmc_300m_dir, esmc_tokenizer):
    """Same weights, same input, both devices, fp32 on each side.

    Also a cross-implementation check: the CPU model uses the reference layers
    while the CUDA model uses the fused Transformer Engine ones, which is where
    almost all of the difference below comes from.
    """
    on_cpu = EsmcModel.from_pretrained(esmc_300m_dir, device="cpu")
    on_gpu = EsmcModel.from_pretrained(esmc_300m_dir, device="cuda")

    for sequence in SEQUENCES.values():
        enc = esmc_tokenizer(sequence, return_tensors="pt")
        with torch.no_grad():
            expected = on_cpu(**enc).last_hidden_state.float()
            actual = on_gpu(**{k: v.cuda() for k, v in enc.items()})
            actual = actual.last_hidden_state.float().cpu()

        # Observed max|Δ| at fp32, activations ~O(1): 2.4e-4 short to 1.3e-3
        # long. It grows with length because the difference is a LayerNorm
        # reduction repeated per block over a lengthening residual stream.
        torch.testing.assert_close(actual, expected, atol=3e-3, rtol=1e-4)


# Every build reachable on a GPU: dtype x attention backend x fused/reference
# layers. flash_attention_2 is only paired with fp16 / bf16 because the kernel
# rejects fp32, and only with the fused layers because a CPU-built model never
# gets the flash block at all (see ``build_esmc``).
GPU_BUILDS = [
    (torch.float32, "sdpa", FUSED_LAYERS),
    (torch.float32, "sdpa", REFERENCE_LAYERS),
    (torch.float32, "eager", FUSED_LAYERS),
    (torch.bfloat16, "sdpa", FUSED_LAYERS),
    (torch.bfloat16, "sdpa", REFERENCE_LAYERS),
    (torch.bfloat16, "flash_attention_2", FUSED_LAYERS),
    (torch.float16, "sdpa", FUSED_LAYERS),
    (torch.float16, "flash_attention_2", FUSED_LAYERS),
]

# What each dtype can be held to against the CPU / fp32 reference: max|Δ| on the
# per-position hidden-state norms (~1.1-1.5) and how many positions' argmax may
# differ. Worst observed across every build and length in GPU_BUILDS:
#              max|Δ| norms   argmax mismatches (of 12 / 131 / 412)
#   fp32       4.5e-4         0 / 0 / 0
#   bf16       1.46e-2        1 / 3 / 3
#   fp16       1.62e-3        0 / 1 / 0
# fp32 with the reference layers reproduces the CPU reference to 1.7e-6; the fp32
# bound has to accommodate Transformer Engine.
REFERENCE_NORM_TOLERANCE = {
    torch.float32: 1e-3,
    torch.bfloat16: 0.03,
    torch.float16: 5e-3,
}
REFERENCE_ARGMAX_MISMATCHES = {torch.float32: 0, torch.bfloat16: 5, torch.float16: 2}


@pytest.mark.gpu
@pytest.mark.parametrize("dtype,attn,layers", GPU_BUILDS)
def test_every_gpu_build_reaches_the_reference(
    esmc_300m_dir, esmc_tokenizer, dtype, attn, layers
):
    """Correct, not merely self-consistent: every GPU build vs the CPU reference.

    No GPU build can reach it bit-for-bit, so what is asserted is the decision
    path (argmax over the vocabulary) plus a bound on the hidden-state norms.
    Without Transformer Engine or flash-attn installed the builds that would use
    them degenerate to the reference ones rather than failing, so this stays a
    superset of whatever a given runner can reach.
    """
    try:
        model = build_esmc(
            esmc_300m_dir,
            device="cuda",
            dtype=dtype,
            attn=attn,
            layers=layers,
            model_class=EsmcForMaskedLM,
        )
    # Only a genuinely absent package is a reason to skip. `from_pretrained`
    # signals a strict-key failure with RuntimeError, so catching that here would
    # turn a checkpoint regression into a green skip across every build.
    except ImportError as exc:
        pytest.skip(f"build {dtype}/{attn}/{layers} unavailable: {exc}")  # ty:ignore[too-many-positional-arguments]

    for name, sequence in SEQUENCES.items():
        case = reference_values()["cases"][name]
        enc = {
            k: v.cuda()
            for k, v in esmc_tokenizer(sequence, return_tensors="pt").items()
        }
        with torch.no_grad():
            out = model(**enc)
        mismatches = argmax_mismatches(out.logits, case["argmax"])
        assert mismatches <= REFERENCE_ARGMAX_MISMATCHES[dtype], (name, mismatches)
        torch.testing.assert_close(
            out.last_hidden_state.float()[0].norm(dim=-1).cpu(),
            torch.tensor(case["last_hidden_state_norms"]),
            atol=REFERENCE_NORM_TOLERANCE[dtype],
            rtol=0,
            msg=lambda text: f"{name}: {text}",
        )


# ---------------------------------------------------------------------------
# The retired date-stamped factories still do real work
# ---------------------------------------------------------------------------


def test_date_stamped_factory_loads_real_weights(esmc_300m_dir, esmc_300m):
    """``esm.pretrained.ESMC_300M_202412`` shipped publicly, so it has to keep
    loading a usable model and agree with the model loaded directly.
    """
    from esm import pretrained

    with pytest.warns(DeprecationWarning):
        legacy = pretrained.ESMC_300M_202412(use_flash_attn=False)

    assert isinstance(legacy, ESMC)
    # The date-stamped factories loaded raw fp32 weights on every device; only
    # ESMC.from_pretrained ever cast to bf16.
    assert next(legacy.parameters()).dtype is torch.float32
    assert not legacy.training

    tokens = legacy._tokenize(["MQIFVKTLTGKTITLEVEPS"])
    with torch.no_grad():
        legacy_out = legacy(sequence_tokens=tokens)
        direct_out = esmc_300m(input_ids=tokens, return_dict=True)

    # Same weights, so the logits must be identical, not merely close.
    assert torch.equal(legacy_out.sequence_logits, direct_out.logits)
    # 300M has 30 blocks, and the old stack held one entry per block output.
    assert legacy_out.hidden_states.shape[0] == 30
    assert legacy_out.embeddings.shape == direct_out.last_hidden_state.shape

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        via_registry = pretrained.load_local_model("esmc_300m", use_flash_attn=False)
    assert isinstance(via_registry, ESMC)
