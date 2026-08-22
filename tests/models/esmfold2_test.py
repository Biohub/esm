"""ESMFold2 test suite.

Structural tests run on CPU against a tiny randomly-initialised model with
synthetic LM states, so no 6B PLM backbone is needed. Published-weight tests use
biohub/ESMFold2 and skip when the Hub is unreachable. GPU tests are marked
``gpu``.
"""

import json
import math
import warnings
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from esm.models.esmfold2 import (
    ESMFOLD2_EXPERIMENTAL_HF_REPO,
    ESMFOLD2_HF_REPO,
    EsmFold2Config,
    EsmFold2ExperimentalModel,
    EsmFold2Model,
)

# flash-attn imports on CPU boxes but needs a CUDA runtime, so the reference
# path is selected explicitly for the CPU tests below.
from esm.models.esmfold2 import layers as _layers
from esm.models.esmfold2.config import _LEGACY_PATHS, _prune_empty
from esm.models.esmfold2.protein_utils import prepare_protein_features
from esm.models.hub import CONFIG_NAME
from tests.conftest import (
    ATOM_ALIGNED_SEQUENCE,
    ATOM_PADDED_SEQUENCE,
    ESMFOLD2_LENGTH_NAMES,
    ESMFOLD2_SEQUENCES,
    _skip_if_unreachable,
    esmfold2_inputs,
)

#: The atom axis is padded to a multiple of this in ``prepare_protein_features``.
ATOM_BLOCK = 32


@pytest.fixture(autouse=True)
def _force_reference_attention(monkeypatch):
    if not torch.cuda.is_available():
        monkeypatch.setattr(_layers, "FLASH_ATTN_AVAILABLE", False)


def _forward(model, seq="MQIFVKTLTGKT", **kwargs):
    features, lm_hidden_states = esmfold2_inputs(model, seq)
    with torch.no_grad():
        return model(
            **features,
            lm_hidden_states=lm_hidden_states,
            num_loops=1,
            num_diffusion_samples=1,
            num_sampling_steps=2,
            **kwargs,
        ), features


# ---------------------------------------------------------------------------
# Construction and the forward shape contract
# ---------------------------------------------------------------------------


def test_both_architectures_construct(tiny_esmfold2_config):
    """The release and experimental models are separate architectures."""
    release = EsmFold2Model(tiny_esmfold2_config).eval()
    experimental = EsmFold2ExperimentalModel(
        EsmFold2Config(**{**asdict_safe(tiny_esmfold2_config), "type": "experimental"})
    ).eval()
    assert len(release.state_dict()) != len(experimental.state_dict())


def asdict_safe(config) -> dict:
    """Config attributes as a plain dict, nested dataclasses included."""
    out = {}
    for key, value in vars(config).items():
        out[key] = asdict(value) if hasattr(value, "__dataclass_fields__") else value
    return out


@pytest.mark.parametrize("name", ESMFOLD2_LENGTH_NAMES)
def test_forward_shape_contract(tiny_esmfold2, name):
    """The output contract at every length regime, not just the 12-mer.

    ``tiny`` reproduces the single length this test used before; the rest sit on
    the sliding-window, chunk and relpos-saturation boundaries listed in
    ``conftest.ESMFOLD2_LENGTHS``, where the token and atom axes can diverge.
    """
    out, features = _forward(tiny_esmfold2, seq=ESMFOLD2_SEQUENCES[name])
    n_tokens = int(features["token_attention_mask"].sum())
    n_atoms = features["atom_attention_mask"].shape[-1]

    assert out["sample_atom_coords"].shape == (1, n_atoms, 3)
    assert out["plddt"].shape == (1, n_tokens)
    assert out["plddt_per_atom"].shape == (1, n_atoms)
    for key in ("pae", "pde"):
        assert out[key].shape == (1, n_tokens, n_tokens)
    for key in ("ptm", "iptm", "complex_plddt"):
        assert out[key].shape == (1,)
    assert torch.isfinite(out["sample_atom_coords"]).all()


# ---------------------------------------------------------------------------
# The per-module shape map, asserted as equality
# ---------------------------------------------------------------------------


def module_output_shapes(model: nn.Module, **inputs) -> dict[str, tuple[int, ...]]:
    """Shape of every module's (first) tensor output during one forward.

    Modules that return a dict record nothing -- ``structure_head``,
    ``structure_head.diffusion_module`` and ``confidence_head`` all do, and their
    fields are what ``test_forward_shape_contract`` asserts. Modules called more
    than once (every diffusion step re-enters the atom encoder) keep the last
    call's shape, which is the same shape each time.
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


def swiglu_ffn_width(d_model: int, expansion_ratio: int) -> int:
    """Hidden width of ``SwiGLUFFN``, which rounds up to a multiple of 256.

    The rounding is why the width cannot simply be read as ``ratio * d_model``:
    at the tiny widths used here every ratio lands on the same 256.
    """
    return ((expansion_ratio * (d_model // 3) * 2) + 255) // 256 * 256


def swa_atom_transformer_shapes(
    prefix: str, n_atoms: int, d_atom: int, n_blocks: int, expansion_ratio: int
) -> dict[str, tuple[int, ...]]:
    """Shapes inside a ``SWAAtomTransformer``: every tensor is on the atom axis."""
    hidden = swiglu_ffn_width(d_atom, expansion_ratio)
    atom = (1, n_atoms, d_atom)
    shapes: dict[str, tuple[int, ...]] = {prefix: atom}
    for index in range(n_blocks):
        block = f"{prefix}.blocks.{index}"
        shapes.update(
            {
                f"{block}.adaln_modulation.0": atom,
                # adaLN-Zero emits shift/scale/gate for the attention and the FFN.
                f"{block}.adaln_modulation.1": (1, n_atoms, 6 * d_atom),
                f"{block}.adaln_modulation": (1, n_atoms, 6 * d_atom),
                # One fused projection for Q, K and V.
                f"{block}.attn.Wqkv": (1, n_atoms, 3 * d_atom),
                f"{block}.attn.gate_proj": atom,
                f"{block}.attn.out_proj": atom,
                f"{block}.attn": atom,
                # SwiGLU packs the gate and the value half into one projection.
                f"{block}.ffn.w_up": (1, n_atoms, 2 * hidden),
                f"{block}.ffn.w_down": atom,
                f"{block}.ffn": atom,
                block: atom,
            }
        )
    return shapes


def atom_encoder_shapes(
    prefix: str,
    n_tokens: int,
    n_atoms: int,
    d_atom: int,
    out_dim: int,
    n_blocks: int,
    expansion_ratio: int,
    structure_prediction: bool,
) -> dict[str, tuple[int, ...]]:
    """Shapes inside an ``EsmFold2AtomEncoder``.

    The whole stack runs on the atom axis and only the module's own output is
    per-token -- which is the rank change a swapped axis would show up in.
    """
    shapes: dict[str, tuple[int, ...]] = {
        f"{prefix}.atom_linear": (1, n_atoms, d_atom),
        f"{prefix}.atom_norm": (1, n_atoms, d_atom),
        f"{prefix}.atom_to_token_linear": (1, n_atoms, out_dim),
        prefix: (1, n_tokens, out_dim),
    }
    if structure_prediction:
        shapes[f"{prefix}.coords_linear"] = (1, n_atoms, d_atom)
    shapes.update(
        swa_atom_transformer_shapes(
            f"{prefix}.atom_transformer", n_atoms, d_atom, n_blocks, expansion_ratio
        )
    )
    return shapes


def folding_trunk_shapes(
    prefix: str, n_tokens: int, d_pair: int, n_layers: int, expansion_ratio: int = 4
) -> dict[str, tuple[int, ...]]:
    """Shapes inside a ``FoldingTrunk``: everything is a ``[B, L, L, c_z]`` pair."""
    pair = (1, n_tokens, n_tokens, d_pair)
    shapes: dict[str, tuple[int, ...]] = {prefix: pair}
    for index in range(n_layers):
        block = f"{prefix}.blocks.{index}"
        for triangle in ("tri_mul_out", "tri_mul_in"):
            engine = f"{block}.{triangle}._engine"
            shapes.update(
                {
                    f"{engine}.norm_start": pair,
                    # Both triangle operands and both their gates in one matmul.
                    f"{engine}.proj_bundle": (1, n_tokens, n_tokens, 4 * d_pair),
                    f"{engine}.norm_mix": pair,
                    f"{engine}.proj_emit": pair,
                    f"{engine}.proj_gate": pair,
                    engine: pair,
                    f"{block}.{triangle}": pair,
                }
            )
        shapes.update(
            {
                f"{block}.row_drop": pair,
                f"{block}.pair_transition.norm": pair,
                f"{block}.pair_transition.ffn.w12": (
                    1,
                    n_tokens,
                    n_tokens,
                    2 * expansion_ratio * d_pair,
                ),
                f"{block}.pair_transition.ffn.w3": pair,
                f"{block}.pair_transition.ffn": pair,
                f"{block}.pair_transition": pair,
                block: pair,
            }
        )
    return shapes


def transition_shapes(
    prefix: str, leading: tuple[int, ...], dim: int, multiplier: int
) -> dict[str, tuple[int, ...]]:
    """Shapes inside a ``TransitionLayer``, on whichever axes ``leading`` names."""
    return {
        f"{prefix}.norm": (*leading, dim),
        f"{prefix}.a_proj": (*leading, multiplier * dim),
        f"{prefix}.b_proj": (*leading, multiplier * dim),
        f"{prefix}.out_proj": (*leading, dim),
        prefix: (*leading, dim),
    }


def expected_module_shapes(
    config, n_tokens: int, n_atoms: int
) -> dict[str, tuple[int, ...]]:
    """Every intermediate width in the release architecture, fully enumerated.

    Derived from the config rather than hardcoded, so the map says what each
    width *is* -- ``c_z`` versus ``d_single`` versus ``d_atom`` -- instead of
    restating a number the model happens to produce.
    """
    d_single = config.hidden_size
    d_pair = config.pairwise_hidden_size
    d_inputs = config.single_inputs_size
    swa = config.atom_encoder
    diffusion = config.structure_head.diffusion_module
    c_token = diffusion.token_hidden_size
    c_z = diffusion.c_z
    c_atom = diffusion.atom_encoder.hidden_size
    multiplier = diffusion.transition_multiplier
    confidence = config.confidence_head

    pair = (1, n_tokens, n_tokens, d_pair)
    lm_pair = (1, n_tokens, n_tokens, d_pair)
    token_c_z = (1, n_tokens, n_tokens, c_z)

    shapes: dict[str, tuple[int, ...]] = {
        # Inputs and the trunk's pair initialisation.
        "inputs_embedder": (1, n_tokens, d_inputs),
        "z_init_1": (1, n_tokens, d_pair),
        "z_init_2": (1, n_tokens, d_pair),
        "rel_pos.embed": pair,
        "rel_pos": pair,
        "token_bonds": pair,
        # The LM shim: per-layer projection first, then the outer product to pair.
        "language_model.base_z_linear.0": (
            1,
            n_tokens,
            config.lm_num_layers + 1,
            config.lm_d_model,
        ),
        "language_model.base_z_linear.1": (
            1,
            n_tokens,
            config.lm_num_layers + 1,
            d_pair,
        ),
        "language_model.base_z_linear": (1, n_tokens, config.lm_num_layers + 1, d_pair),
        "language_model.base_z_mlp.0.downproject": (1, n_tokens, d_pair),
        "language_model.base_z_mlp.0.output_mlp.0": lm_pair,
        "language_model.base_z_mlp.0.output_mlp.1": lm_pair,
        "language_model.base_z_mlp.0.output_mlp.2": lm_pair,
        "language_model.base_z_mlp.0.output_mlp": lm_pair,
        "language_model.base_z_mlp.0": lm_pair,
        "language_model.base_z_mlp.1": lm_pair,
        "language_model.base_z_mlp": lm_pair,
        "language_model": lm_pair,
        # The parcae recurrence around the trunk.
        "parcae_input_norm": pair,
        "parcae_readout": pair,
        "distogram_head": (1, n_tokens, n_tokens, config.structure_head.distogram_bins),
        # Diffusion conditioning: pair on [B, L, L, *], single on [B, L, *],
        # noise on [B, *] alone.
        "structure_head.diffusion_module.conditioning.z_input_norm": (
            1,
            n_tokens,
            n_tokens,
            2 * c_z,
        ),
        "structure_head.diffusion_module.conditioning.z_proj": token_c_z,
        "structure_head.diffusion_module.conditioning.s_input_norm": (
            1,
            n_tokens,
            d_inputs,
        ),
        "structure_head.diffusion_module.conditioning.s_proj": (1, n_tokens, c_token),
        "structure_head.diffusion_module.conditioning.fourier": (
            1,
            diffusion.fourier_dim,
        ),
        "structure_head.diffusion_module.conditioning.noise_norm": (
            1,
            diffusion.fourier_dim,
        ),
        "structure_head.diffusion_module.conditioning.noise_proj": (1, c_token),
        "structure_head.diffusion_module.conditioning": (1, n_tokens, c_token),
        "structure_head.diffusion_module.s_step_norm": (1, n_tokens, c_token),
        "structure_head.diffusion_module.s_to_token": (1, n_tokens, c_token),
        "structure_head.diffusion_module.token_norm": (1, n_tokens, c_token),
        # Confidence head.
        "confidence_head.s_inputs_norm": (1, n_tokens, d_inputs),
        "confidence_head.z_norm": pair,
        "confidence_head.s_to_z": (1, n_tokens, d_pair),
        "confidence_head.s_to_z_transpose": (1, n_tokens, d_pair),
        "confidence_head.s_to_z_prod_in1": (1, n_tokens, d_pair),
        "confidence_head.s_to_z_prod_in2": (1, n_tokens, d_pair),
        "confidence_head.s_to_z_prod_out": pair,
        "confidence_head.dist_bin_pairwise_embed": pair,
        "confidence_head.row_attention_pooling.attn_proj": (1, n_tokens, n_tokens, 1),
        "confidence_head.row_attention_pooling.out_proj": (1, n_tokens, d_single),
        "confidence_head.row_attention_pooling": (1, n_tokens, d_single),
        # pLDDT and resolved are per-atom; PAE and PDE are per pair.
        "confidence_head.plddt_ln": (1, n_atoms, d_single),
        "confidence_head.resolved_ln": (1, n_atoms, d_single),
        "confidence_head.pae_ln": pair,
        "confidence_head.pae_head": (1, n_tokens, n_tokens, confidence.num_pae_bins),
        "confidence_head.pde_ln": pair,
        "confidence_head.pde_head": (1, n_tokens, n_tokens, confidence.num_pde_bins),
    }

    shapes.update(
        atom_encoder_shapes(
            "inputs_embedder.atom_attention_encoder",
            n_tokens,
            n_atoms,
            d_atom=swa.hidden_size,
            # The inputs embedder takes half the token width and concatenates
            # aatype / profile / deletion_mean onto it to reach d_inputs.
            out_dim=swa.output_dim // 2,
            n_blocks=swa.num_hidden_layers,
            expansion_ratio=swa.expansion_ratio,
            structure_prediction=False,
        )
    )
    shapes.update(
        atom_encoder_shapes(
            "structure_head.diffusion_module.atom_encoder",
            n_tokens,
            n_atoms,
            d_atom=c_atom,
            out_dim=c_token,
            n_blocks=diffusion.atom_encoder.num_hidden_layers,
            # DiffusionModule hardcodes 2 rather than reading the config.
            expansion_ratio=2,
            structure_prediction=True,
        )
    )
    # The decoder mirrors the encoder but ends on 3 coordinates per atom.
    decoder = "structure_head.diffusion_module.atom_decoder"
    shapes.update(
        swa_atom_transformer_shapes(
            f"{decoder}.atom_transformer",
            n_atoms,
            c_atom,
            diffusion.atom_encoder.num_hidden_layers,
            expansion_ratio=2,
        )
    )
    shapes.update(
        {
            f"{decoder}.token_to_atom_linear": (1, n_tokens, c_atom),
            f"{decoder}.norm": (1, n_atoms, c_atom),
            f"{decoder}.output_linear": (1, n_atoms, 3),
            decoder: (1, n_atoms, 3),
        }
    )

    for stack, layers in (
        ("lm_encoder", config.lm_encoder.num_hidden_layers),
        ("folding_trunk", config.folding_trunk_num_hidden_layers),
        ("parcae_coda", config.parcae_num_coda_layers),
        ("confidence_head.folding_trunk", confidence.num_hidden_layers),
    ):
        shapes.update(folding_trunk_shapes(stack, n_tokens, d_pair, layers))

    for index in range(2):
        shapes.update(
            transition_shapes(
                f"structure_head.diffusion_module.conditioning.z_transitions.{index}",
                (1, n_tokens, n_tokens),
                c_z,
                multiplier,
            )
        )
        shapes.update(
            transition_shapes(
                f"structure_head.diffusion_module.conditioning.s_transitions.{index}",
                (1, n_tokens),
                c_token,
                multiplier,
            )
        )

    transformer = "structure_head.diffusion_module.token_transformer"
    shapes[transformer] = (1, n_tokens, c_token)
    for index in range(diffusion.token_num_blocks):
        attn = f"{transformer}.attn_blocks.{index}"
        shapes.update(
            {
                f"{attn}.adaln.s_gate": (1, n_tokens, c_token),
                f"{attn}.adaln.s_shift": (1, n_tokens, c_token),
                f"{attn}.adaln": (1, n_tokens, c_token),
                f"{attn}.q_proj": (1, n_tokens, c_token),
                # K and V in one projection.
                f"{attn}.kv_proj": (1, n_tokens, 2 * c_token),
                f"{attn}.g_proj": (1, n_tokens, c_token),
                f"{attn}.pair_norm": token_c_z,
                # One bias scalar per head, on the pair axes.
                f"{attn}.pair_bias_proj": (
                    1,
                    n_tokens,
                    n_tokens,
                    diffusion.token_num_heads,
                ),
                f"{attn}.out_proj": (1, n_tokens, c_token),
                f"{attn}.out_gate": (1, n_tokens, c_token),
                attn: (1, n_tokens, c_token),
            }
        )
        block = f"{transformer}.transition_blocks.{index}"
        shapes.update(
            {
                f"{block}.adaln.s_gate": (1, n_tokens, c_token),
                f"{block}.adaln.s_shift": (1, n_tokens, c_token),
                f"{block}.adaln": (1, n_tokens, c_token),
                # SwiGLU again: the gate and the value half share one projection.
                f"{block}.lin_swish": (1, n_tokens, 2 * multiplier * c_token),
                f"{block}.lin_out": (1, n_tokens, c_token),
                f"{block}.output_gate": (1, n_tokens, c_token),
                block: (1, n_tokens, c_token),
            }
        )
    return shapes


@pytest.mark.parametrize("name", ESMFOLD2_LENGTH_NAMES)
def test_every_module_output_has_the_expected_shape(
    tiny_esmfold2, tiny_esmfold2_config, name
):
    """Pin the width of every intermediate, not just the model's output.

    ESMFold2 carries three tensor ranks at once -- token ``[B, L, d]``, pair
    ``[B, L, L, c_z]`` and atom ``[B, A, d_atom]`` -- and a bug that swaps a
    pair axis for a token axis still produces correctly-shaped coordinates.
    Asserted as equality, so a new submodule fails here rather than slipping in
    unnoticed.
    """
    features, lm_hidden_states = esmfold2_inputs(
        tiny_esmfold2, ESMFOLD2_SEQUENCES[name]
    )
    n_tokens = int(features["token_attention_mask"].shape[-1])
    n_atoms = int(features["atom_attention_mask"].shape[-1])

    measured = module_output_shapes(
        tiny_esmfold2,
        **features,
        lm_hidden_states=lm_hidden_states,
        num_loops=1,
        num_diffusion_samples=1,
        num_sampling_steps=2,
    )
    assert measured == expected_module_shapes(tiny_esmfold2_config, n_tokens, n_atoms)


def test_widths_no_hook_can_see_are_pinned_too(tiny_esmfold2, tiny_esmfold2_config):
    """The pLDDT / resolved / LM-mixing weights are used through einsum and softmax.

    No module wraps them, so ``module_output_shapes`` is blind to them, exactly
    as the ESMC suite's SwiGLU widths are. ``max_atoms_per_token`` is hardcoded
    at 23 in ``ConfidenceHead``: it is the widest residue's heavy-atom count and
    a checkpoint cannot change it, so it is asserted as the literal.
    """
    config = tiny_esmfold2_config
    confidence = tiny_esmfold2.confidence_head
    assert confidence.plddt_weight.shape == (
        23,
        config.hidden_size,
        config.confidence_head.num_plddt_bins,
    )
    # 2 = [unresolved, resolved].
    assert confidence.resolved_weight.shape == (23, config.hidden_size, 2)
    # One mixing coefficient per LM layer, plus the embedding output.
    assert tiny_esmfold2.language_model.base_z_combine.shape == (
        config.lm_num_layers + 1,
    )


# ---------------------------------------------------------------------------
# The atom axis and its 32-wide padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sequence",
    [*ESMFOLD2_SEQUENCES.values(), ATOM_ALIGNED_SEQUENCE, ATOM_PADDED_SEQUENCE],
    ids=[*ESMFOLD2_LENGTH_NAMES, "atom_aligned", "atom_padded"],
)
def test_atom_axis_is_padded_to_a_multiple_of_32(sequence):
    """Every feature on the atom axis is padded to the same 32-wide block.

    The atom count is residue-dependent, so the padding is recomputed here from
    the heavy-atom tables rather than compared against a stored number: what has
    to hold is that the tensors agree with each other and that the mask marks
    exactly the real atoms, with no partially-filled block left behind.
    """
    features = prepare_protein_features(sequence)
    mask = features["atom_attention_mask"][0]
    n_atoms = int(mask.shape[0])
    n_real = int(mask.sum())

    assert n_atoms % ATOM_BLOCK == 0
    assert n_atoms == math.ceil(n_real / ATOM_BLOCK) * ATOM_BLOCK
    # Real atoms come first and padding is a contiguous tail, which is what lets
    # every atom-axis feature share one mask.
    assert torch.equal(mask, torch.arange(n_atoms) < n_real)
    for key in (
        "ref_pos",
        "ref_element",
        "ref_charge",
        "ref_space_uid",
        "atom_to_token",
    ):
        assert features[key].shape[1] == n_atoms, key
    assert features["ref_atom_name_chars"].shape[1] == n_atoms
    # Padded rows must be zero: the atom-encoder Linears are bias-free, so a
    # non-zero pad row would leak into the token aggregation.
    assert not features["ref_pos"][0, n_real:].any()
    assert not features["ref_element"][0, n_real:].any()


@pytest.mark.parametrize(
    "sequence,n_real,n_atoms",
    [(ATOM_ALIGNED_SEQUENCE, 32, 32), (ATOM_PADDED_SEQUENCE, 37, 64)],
    ids=["exactly_one_block", "one_residue_past_a_block"],
)
def test_the_atom_padding_edges_are_where_they_are_claimed_to_be(
    tiny_esmfold2, sequence, n_real, n_atoms
):
    """The two cases the padding branch splits on, and a forward through each.

    Poly-glycine is the only sequence family whose atom count is predictable
    without the residue tables (4 heavy atoms each), so 8 glycines is exactly one
    block and one alanine past that forces 27 rows of padding. Asserted as
    literals here because these two counts are what makes the fixtures edges at
    all -- if the tables change, this test is where it must be noticed.
    """
    features = prepare_protein_features(sequence)
    mask = features["atom_attention_mask"][0]
    assert int(mask.sum()) == n_real
    assert int(mask.shape[0]) == n_atoms

    out, _ = _forward(tiny_esmfold2, seq=sequence)
    assert out["sample_atom_coords"].shape == (1, n_atoms, 3)
    assert out["plddt_per_atom"].shape == (1, n_atoms)
    # The padded rows are still predicted, but only the real ones are finite
    # inputs to anything downstream, so that is what is required to be usable.
    assert torch.isfinite(out["sample_atom_coords"][0, :n_real]).all()
    assert torch.isfinite(out["plddt_per_atom"]).all()


# ---------------------------------------------------------------------------
# Config round-trip, including the nested dataclasses
# ---------------------------------------------------------------------------


def test_config_round_trip(tmp_path, tiny_esmfold2_config):
    tiny_esmfold2_config.save_pretrained(tmp_path)
    reloaded = EsmFold2Config.from_pretrained(tmp_path)

    assert reloaded.type == tiny_esmfold2_config.type
    assert reloaded.hidden_size == tiny_esmfold2_config.hidden_size
    # Nested dataclasses must survive the dict -> json -> dict trip.
    assert (
        reloaded.structure_head.diffusion_module.token_hidden_size
        == tiny_esmfold2_config.structure_head.diffusion_module.token_hidden_size
    )
    assert (
        reloaded.structure_head.diffusion_module.atom_encoder.hidden_size
        == tiny_esmfold2_config.structure_head.diffusion_module.atom_encoder.hidden_size
    )
    assert (
        reloaded.confidence_head.num_hidden_layers
        == tiny_esmfold2_config.confidence_head.num_hidden_layers
    )


def test_config_keeps_unknown_fields(tmp_path):
    """A newer published config.json must not lose fields this reader ignores."""
    (tmp_path / "config.json").write_text(json.dumps({"some_future_field": 7}))
    cfg = EsmFold2Config.from_pretrained(tmp_path)
    assert cfg.some_future_field == 7  # ty:ignore[unresolved-attribute]


def test_rejects_unknown_type():
    with pytest.raises(ValueError, match="release.*experimental"):
        EsmFold2Config(type="nonsense")


# ---------------------------------------------------------------------------
# Reading pre-alignment configs
# ---------------------------------------------------------------------------


def test_reads_pre_alignment_config(tmp_path, tiny_esmfold2_config):
    """Checkpoints published before the alignment must still resolve, loudly."""
    legacy = {
        "type": "release",
        "d_single": 48,
        "d_pair": 32,
        "num_loops": 3,
        "inputs": {"d_inputs": 451, "atom_encoder": {"d_atom": 96, "n_heads": 3}},
        "folding_trunk": {"n_layers": 5, "n_heads": 2},
        "parcae": {"coda_n_layers": 1},
        "confidence_head": {"folding_trunk": {"n_layers": 2}},
    }
    (tmp_path / "config.json").write_text(json.dumps(legacy))

    with pytest.warns(FutureWarning, match="pre-alignment"):
        cfg = EsmFold2Config.from_pretrained(tmp_path)

    assert cfg.hidden_size == 48
    assert cfg.pairwise_hidden_size == 32
    assert cfg.single_inputs_size == 451
    assert cfg.atom_encoder.hidden_size == 96
    assert cfg.atom_encoder.num_attention_heads == 3
    assert cfg.folding_trunk_num_hidden_layers == 5
    assert cfg.parcae_num_coda_layers == 1
    assert cfg.confidence_head.num_hidden_layers == 2
    # The emptied wrappers must not survive as stray attributes.
    assert not hasattr(cfg, "inputs")
    assert not hasattr(cfg, "folding_trunk")


def test_rejects_num_recycles(tmp_path):
    """``num_recycles`` is the old name for ``num_loops``.

    Reading past it would leave the trunk on its default loop count, which is not
    what such a checkpoint was trained for, so it has to fail loudly.
    """
    (tmp_path / "config.json").write_text(json.dumps({"num_recycles": 3}))
    with pytest.raises(ValueError, match="num_loops"):
        EsmFold2Config.from_pretrained(tmp_path)


# ---------------------------------------------------------------------------
# Every legacy path, one at a time
# ---------------------------------------------------------------------------

# A value no default in the config equals, so "the remap worked" cannot be
# confused with "the destination already held this". Deliberately not a
# plausible setting for any of the fields: nothing is constructed from it.
_LEGACY_SENTINEL = 7


def nested_path(tree, path: str):
    """Follow a dotted path through nested dicts / dataclasses / configs."""
    node = tree
    for part in path.split("."):
        node = node[part] if isinstance(node, dict) else getattr(node, part)
    return node


@pytest.mark.parametrize("legacy", sorted(_LEGACY_PATHS), ids=sorted(_LEGACY_PATHS))
def test_every_legacy_path_remaps_and_warns_once(tmp_path, legacy):
    """Each pre-alignment field lands on its current name, loudly and exactly once.

    Parametrized one path at a time rather than asserted on a whole legacy
    config, because that is the only shape in which a *silently dropped* entry
    fails: in a bulk config the surviving 27 remaps still produce a warning and
    a plausible model.
    """
    destination = _LEGACY_PATHS[legacy]
    nested: dict = {}
    node = nested
    parts = legacy.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = _LEGACY_SENTINEL
    (tmp_path / CONFIG_NAME).write_text(json.dumps({"type": "release", **nested}))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = EsmFold2Config.from_pretrained(tmp_path)

    future = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert len(future) == 1, [str(w.message) for w in future]
    assert "pre-alignment" in str(future[0].message)
    assert nested_path(config, destination) == _LEGACY_SENTINEL
    # The old spelling must not survive alongside the new one, or a reader could
    # pick up whichever of the two it happened to know about.
    with pytest.raises((AttributeError, KeyError, TypeError)):
        nested_path(config, legacy)


def test_a_current_config_warns_about_nothing():
    """The negative half: no FutureWarning when nothing legacy is present.

    Without this the parametrized test above would pass on an implementation
    that warned unconditionally.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        EsmFold2Config(hidden_size=32, pairwise_hidden_size=16)
    assert not [w for w in caught if issubclass(w.category, FutureWarning)]


def test_prune_empty_drops_wrappers_emptied_by_remapping():
    """``inputs`` and ``folding_trunk`` hold nothing once their leaves move out.

    Innermost first, so a wrapper whose only child was itself emptied goes too --
    ``confidence_head.folding_trunk`` is exactly that shape. A wrapper that still
    holds a leaf, and a falsy leaf that is not a dict, must both survive.
    """
    tree = {
        "inputs": {},
        "confidence_head": {"folding_trunk": {}},
        "structure_head": {"diffusion_module": {}, "distogram_bins": 8},
        "num_loops": 0,
        "parcae": {"enabled": False},
    }
    _prune_empty(tree)
    assert tree == {
        "structure_head": {"distogram_bins": 8},
        "num_loops": 0,
        "parcae": {"enabled": False},
    }


def test_a_config_missing_an_architecture_field_cannot_load_a_checkpoint(
    tmp_path, tiny_esmfold2_config
):
    """``EsmFold2Config`` defaults every architectural field, so the refusal is
    strict loading's, not the config's.

    Unlike ``EsmcConfig``, which raises ``KeyError`` on a missing width, dropping
    ``hidden_size`` here silently yields the 384-wide default. What must not
    happen is that such a config quietly builds a model and accepts the
    checkpoint anyway, so this is pinned one layer down.
    """
    directory = save_tiny_checkpoint(tmp_path, tiny_esmfold2_config)
    raw = json.loads((directory / CONFIG_NAME).read_text())
    assert raw.pop("hidden_size") != EsmFold2Config().hidden_size
    (directory / CONFIG_NAME).write_text(json.dumps(raw))

    assert EsmFold2Config.from_pretrained(directory).hidden_size == 384
    with pytest.raises(RuntimeError, match="size mismatch"):
        EsmFold2Model.from_pretrained(directory, load_esmc=False)


# ---------------------------------------------------------------------------
# Strict loading, from the negative side
# ---------------------------------------------------------------------------


def save_tiny_checkpoint(
    directory: Path, config, model_class=EsmFold2Model, extra: dict | None = None
) -> Path:
    """Write ``config.json`` + ``model.safetensors`` for a freshly-built model.

    Neither ESMFold2 class has ``save_pretrained``, and the trunk's tensor layout
    is already the published one -- ``_normalize_checkpoint_layout`` only
    rewrites the bundled ``esmc.`` subtree -- so the state dict is written
    verbatim.
    """
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    state = dict(model_class(config).state_dict())
    state.update(extra or {})
    config.save_pretrained(directory)
    save_file(
        {k: v.contiguous() for k, v in state.items()},
        str(directory / "model.safetensors"),
        metadata={"format": "pt"},
    )
    return directory


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


@pytest.mark.parametrize(
    "model_class",
    [EsmFold2Model, EsmFold2ExperimentalModel],
    ids=["release", "experimental"],
)
def test_an_incomplete_checkpoint_is_refused(
    tmp_path, tiny_esmfold2_config, model_class
):
    """A parameter the checkpoint never covered holds whatever memory it landed on.

    ``_load_pretrained`` calls ``load_state_dict(strict=False)`` and does its own
    accounting afterwards, so this is the only thing standing between a truncated
    checkpoint and a model full of ``to_empty`` garbage. ESMFold2 had no negative
    strict-load test at all before this one.
    """
    config = architecture_config(tiny_esmfold2_config, model_class)
    directory = save_tiny_checkpoint(tmp_path, config, model_class)
    drop_keys(directory, lambda k: k.startswith("folding_trunk.blocks.0."))

    with pytest.raises(RuntimeError, match=r"missing keys \(\d+\)"):
        model_class.from_pretrained(directory, load_esmc=False)


def test_an_unexpected_key_is_refused(tmp_path, tiny_esmfold2_config):
    """A checkpoint entry that reaches nothing is a load that lost weights.

    The symmetric failure to a missing key: the model builds fine and every
    parameter is populated, but part of the checkpoint was silently discarded --
    which is what a rename landing nowhere looks like.
    """
    directory = save_tiny_checkpoint(
        tmp_path,
        tiny_esmfold2_config,
        extra={"folding_trunk.blocks.0.not_a_real_parameter": torch.zeros(4)},
    )
    with pytest.raises(RuntimeError, match=r"unexpected keys \(\d+\)"):
        EsmFold2Model.from_pretrained(directory, load_esmc=False)


@pytest.mark.parametrize(
    "model_class",
    [EsmFold2Model, EsmFold2ExperimentalModel],
    ids=["release", "experimental"],
)
def test_only_transformer_engine_extra_state_is_excused(
    tmp_path, tiny_esmfold2_config, model_class
):
    """The ignore list must stay exactly one entry wide.

    ``_extra_state`` is Transformer Engine's non-tensor bookkeeping, which no
    module tree has a home for. Every other unexpected key has to raise, so the
    list is pinned as a literal *and* exercised: an added ``_extra_state`` entry
    loads, and the near-miss spelling next to it does not.
    """
    assert model_class._keys_to_ignore_on_load_unexpected == [r"\._extra_state$"]

    config = architecture_config(tiny_esmfold2_config, model_class)
    directory = save_tiny_checkpoint(
        tmp_path,
        config,
        model_class,
        extra={"folding_trunk.blocks.0._extra_state": torch.zeros(1)},
    )
    model = model_class.from_pretrained(directory, load_esmc=False)
    assert not [k for k, t in model.state_dict().items() if t.is_meta]

    # Not anchored at a ``.`` boundary, so the regex must not match it.
    save_tiny_checkpoint(
        tmp_path / "near_miss",
        config,
        model_class,
        extra={"folding_trunk.blocks.0.fp8_extra_state": torch.zeros(1)},
    )
    with pytest.raises(RuntimeError, match=r"unexpected keys \(\d+\)"):
        model_class.from_pretrained(tmp_path / "near_miss", load_esmc=False)


# ---------------------------------------------------------------------------
# ``from_pretrained`` dispatches on ``config.type``
# ---------------------------------------------------------------------------


def architecture_config(config, model_class):
    """``config`` retyped for whichever of the two architectures is being built."""
    kind = "release" if model_class is EsmFold2Model else "experimental"
    return EsmFold2Config(**{**asdict_safe(config), "type": kind})


def test_from_pretrained_dispatches_on_config_type(tmp_path, tiny_esmfold2_config):
    """An experimental checkpoint must come back as the experimental class.

    ``EsmFold2Model`` is the entry point users are given, and the two
    architectures have different state dicts, so a dispatch that failed would
    surface as a key error rather than a wrong model -- but only because loading
    is strict. Run against a locally saved tiny config rather than the
    experimental Hub repo so it stays in the fast tier.
    """
    experimental = architecture_config(tiny_esmfold2_config, EsmFold2ExperimentalModel)
    directory = save_tiny_checkpoint(
        tmp_path / "experimental", experimental, EsmFold2ExperimentalModel
    )
    loaded = EsmFold2Model.from_pretrained(directory, load_esmc=False)
    assert isinstance(loaded, EsmFold2ExperimentalModel)
    assert loaded.config.type == "experimental"

    release = save_tiny_checkpoint(tmp_path / "release", tiny_esmfold2_config)
    assert (
        type(EsmFold2Model.from_pretrained(release, load_esmc=False)) is EsmFold2Model
    )


def test_the_experimental_class_does_not_dispatch_back_to_release(
    tmp_path, tiny_esmfold2_config
):
    """The converse: dispatch is one-way, and the mismatch is refused, not silent.

    ``from_pretrained`` only redirects when ``cls is EsmFold2Model``, so asking
    the experimental class for a release checkpoint builds the experimental
    architecture. What must not happen is that it loads anyway.
    """
    release = save_tiny_checkpoint(tmp_path, tiny_esmfold2_config)
    with pytest.raises(RuntimeError, match=r"(missing|unexpected) keys"):
        EsmFold2ExperimentalModel.from_pretrained(release, load_esmc=False)


# ---------------------------------------------------------------------------
# A backbone bundled into the same checkpoint
# ---------------------------------------------------------------------------


def test_bundled_backbone_is_built_and_loaded_in_one_pass(
    tmp_path, tiny_esmfold2_config, tiny_esmc_config
):
    """``esmc_config`` means the backbone ships with the trunk.

    It must be built in ``__init__`` and filled by the same ``from_pretrained``
    pass -- no second download, nothing left on meta -- and the encoder arrives in
    the published split-projection layout, so it also exercises the fuse-on-load
    translation through a host model.
    """
    from safetensors.torch import save_file

    from esm.models.esmc.checkpoint_layout import native_to_published
    from esm.models.esmc.model import EsmcModel

    unbundled = EsmFold2Model(tiny_esmfold2_config)
    assert unbundled.esmc is None
    trunk_keys = set(unbundled.state_dict())

    bundled_config = EsmFold2Config(
        **{
            k: v
            for k, v in tiny_esmfold2_config.to_dict().items()
            if k not in ("model_type", "esmc_config")
        },
        esmc_config=tiny_esmc_config,
    )
    bundled = EsmFold2Model(bundled_config)
    backbone = EsmcModel(tiny_esmc_config)
    assert bundled.esmc is not None
    esmc_keys = {k for k in bundled.state_dict() if k.startswith("esmc.")}
    assert esmc_keys and set(bundled.state_dict()) == trunk_keys | esmc_keys

    state = dict(bundled.state_dict())
    state.update({f"esmc.{k}": v for k, v in backbone.state_dict().items()})
    published = native_to_published(state)
    assert any("self_attn.q_proj" in k for k in published)

    bundled_config.save_pretrained(tmp_path)
    save_file(
        {k: v.contiguous() for k, v in published.items()},
        str(tmp_path / "model.safetensors"),
        metadata={"format": "pt"},
    )

    loaded = EsmFold2Model.from_pretrained(
        tmp_path, load_esmc=False, esmc_precision="fp32"
    )
    assert not [k for k, t in loaded.state_dict().items() if t.is_meta]
    got, want = loaded.esmc.state_dict(), backbone.state_dict()
    assert set(got) == set(want)
    assert all(torch.equal(got[k], want[k]) for k in want)


def test_bundled_backbone_is_frozen_and_cast(
    tmp_path, tiny_esmfold2_config, tiny_esmc_config
):
    """The bundled path must land on the same dtype/grad state as ``load_esmc``."""
    from safetensors.torch import save_file

    from esm.models.esmc.checkpoint_layout import native_to_published

    bundled_config = EsmFold2Config(
        **{
            k: v
            for k, v in tiny_esmfold2_config.to_dict().items()
            if k not in ("model_type", "esmc_config")
        },
        esmc_config=tiny_esmc_config,
    )
    bundled = EsmFold2Model(bundled_config)
    bundled_config.save_pretrained(tmp_path)
    save_file(
        {
            k: v.contiguous()
            for k, v in native_to_published(bundled.state_dict()).items()
        },
        str(tmp_path / "model.safetensors"),
        metadata={"format": "pt"},
    )

    loaded = EsmFold2Model.from_pretrained(tmp_path, load_esmc=False)
    assert {t.dtype for t in loaded.esmc.state_dict().values()} == {torch.bfloat16}
    assert not any(p.requires_grad for p in loaded.esmc.parameters())


# ---------------------------------------------------------------------------
# Invariances
# ---------------------------------------------------------------------------


def test_eval_forward_is_deterministic(tiny_esmfold2):
    """Determinism holds per-seed, and only per-seed.

    Both reseeds are explicit: ``esmfold2_inputs`` also seeds, to build the
    synthetic LM states, so leaving them implicit would make this pass for a
    reason it does not state.
    """
    torch.manual_seed(0)
    first, _ = _forward(tiny_esmfold2)
    torch.manual_seed(0)
    second, _ = _forward(tiny_esmfold2)
    torch.testing.assert_close(
        first["sample_atom_coords"], second["sample_atom_coords"]
    )


# ---------------------------------------------------------------------------
# PDB rendering
# ---------------------------------------------------------------------------


def test_output_to_pdb_is_well_formed(tiny_esmfold2):
    """Guards the ProteinChain rendering path that replaced openfold's to_pdb."""
    from esm.models.esmfold2.protein_utils import OUTPUT_TO_PDB_FEATURE_KEYS

    out, features = _forward(tiny_esmfold2)
    for key in OUTPUT_TO_PDB_FEATURE_KEYS:
        out[key] = features[key]
    pdb = tiny_esmfold2.output_to_pdb(out)

    atoms = [ln for ln in pdb.splitlines() if ln.startswith("ATOM")]
    assert atoms, "no ATOM records emitted"

    for line in atoms:
        assert len(line) >= 66, line
        # Columns 61-66 are the b-factor, which carries per-atom pLDDT.
        float(line[60:66])
        # Coordinates must parse and be finite.
        for start in (30, 38, 46):
            assert not (v := line[start : start + 8]).strip() == "" and float(
                v
            ) == float(v)

    # Residue numbering starts at 1 and never decreases.
    numbers = [int(ln[22:26]) for ln in atoms]
    assert numbers[0] == 1
    assert numbers == sorted(numbers)


# ---------------------------------------------------------------------------
# Dtypes reachable on CPU
# ---------------------------------------------------------------------------


def test_cpu_fp32(tiny_esmfold2):
    """fp32 is the only precision ESMFold2 supports on CPU.

    The trunk mixes fp32 features with model weights under
    ``torch.autocast(device_type="cuda")``, which is a no-op off CUDA, so a
    bf16-weight model on CPU raises a dtype mismatch in the first Linear.
    bf16 / fp16 are therefore covered by the GPU test below.
    """
    out, _ = _forward(tiny_esmfold2)
    assert out["sample_atom_coords"].dtype == torch.float32
    assert torch.isfinite(out["sample_atom_coords"]).all()


# TODO: test more precisions. Casting the trunk to bf16/fp16 raises
# "mat1 and mat2 must have the same dtype" on GPU as well as CPU - the model
# runs fp32 weights under CUDA autocast, so a whole-model cast is not a
# supported path.


# ---------------------------------------------------------------------------
# Published weights load completely
# ---------------------------------------------------------------------------


# Resolves the repo itself rather than taking a weights fixture, so the
# conftest auto-marking does not reach it.
@pytest.mark.merge_only
@pytest.mark.parametrize(
    "repo,expected",
    [
        (ESMFOLD2_HF_REPO, EsmFold2Model),
        (ESMFOLD2_EXPERIMENTAL_HF_REPO, EsmFold2ExperimentalModel),
    ],
)
def test_loads_from_hub_with_full_key_coverage(repo, expected):
    """Strict load: nothing dropped, nothing left on meta, right class chosen.

    ``EsmFold2Model.from_pretrained`` dispatches on ``config.type``, so the
    experimental repo must come back as the experimental class.
    """
    local_dir = _skip_if_unreachable(repo)
    model = EsmFold2Model.from_pretrained(local_dir, load_esmc=False)
    assert isinstance(model, expected)
    on_meta = [k for k, t in model.state_dict().items() if t.is_meta]
    assert not on_meta, on_meta


# ---------------------------------------------------------------------------
# Kernel / library matrix
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["fused", "cuequivariance"])
def test_kernel_backends_agree(kernel_esmfold2_config, backend):
    """Whatever accelerators a user has installed, the answer must match.

    Compared on the final coordinates and pLDDT rather than intermediate
    residuals: the fused paths differ by ~O(10) fp32 / ~O(100) bf16 on the raw
    residual stream and only converge after the final LayerNorm.

    ``None`` is not a case: it sets the same backend on both sides, so it
    compared two runs of one configuration and measured CUDA's own
    nondeterminism rather than any kernel.
    """
    from esm.models.esmfold2.layers import CUE_AVAILABLE, TRITON_KERNELS_AVAILABLE

    if backend == "fused" and not TRITON_KERNELS_AVAILABLE:
        pytest.skip("triton kernels unavailable")  # ty:ignore[too-many-positional-arguments]
    if backend == "cuequivariance" and not CUE_AVAILABLE:
        pytest.skip("cuequivariance unavailable")  # ty:ignore[too-many-positional-arguments]

    torch.manual_seed(0)
    model = EsmFold2Model(kernel_esmfold2_config).eval().cuda()
    model.set_chunk_size(None)

    features, lm_hidden_states = esmfold2_inputs(model)
    features = {k: v.cuda() for k, v in features.items()}
    lm_hidden_states = lm_hidden_states.cuda()

    def run():
        # The sampler draws fresh noise per call; reseed for a fair comparison.
        torch.manual_seed(0)
        with torch.no_grad():
            return model(
                **features,
                lm_hidden_states=lm_hidden_states,
                num_loops=1,
                num_diffusion_samples=1,
                num_sampling_steps=2,
            )

    model.set_kernel_backend(None)
    reference = run()
    model.set_kernel_backend(backend)
    candidate = run()

    # Tolerances follow the measured noise, not a round number. On H100 the
    # coordinates differ by 2e-4 between two runs of the *same* backend - CUDA
    # matmul selection is not deterministic - so that is the floor; the fused and
    # cuequivariance deltas were 3.4e-4 and 4.3e-4. pLDDT was bit-identical.
    # rtol is 0 for coordinates: they are Angstroms with |x| ~ 21, so a relative
    # term would swamp the absolute bound.
    #
    # 3e-2 rather than 10x that floor, which failed on the CI GPU. The floor is
    # an H100 measured on a quiet device; gpu-ci runs -n3 against one shared GPU,
    # where kernel selection differs between the two forwards, and the sampler
    # amplifies it. It matches the 25x-of-floor headroom esmfold2_builds_test
    # uses for the same comparison, which survives that runner. A wrong kernel
    # misses by orders of magnitude, so the check keeps its teeth.
    torch.testing.assert_close(
        candidate["sample_atom_coords"].float(),
        reference["sample_atom_coords"].float(),
        atol=3e-2,
        rtol=0,
        msg="sample_atom_coords",
    )
    torch.testing.assert_close(
        candidate["plddt"].float(),
        reference["plddt"].float(),
        atol=1e-4,
        rtol=0,
        msg="plddt",
    )
