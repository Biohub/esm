"""Read a HuggingFace-layout ESMFold2 checkpoint into this implementation.

The upstream ``transformers`` port renames most of the module tree, splits our
packed ``Wqkv``, fuses two unfused projections, and drops or adds a level in a
few places. :func:`hf_state_dict_to_native` undoes all of it, and
:func:`is_hf_layout` decides whether it needs to run, from the key names alone.
``EsmFold2Model.from_pretrained`` calls both, so either layout just loads. Their
``config.json`` needs no translation: the published one is a superset carrying
both vocabularies, and :func:`drop_undeclared` discards the half we do not use.

The rules below are the inverse of their
``convert_esmfold2_checkpoint.rename_trunk_keys``. They are verified against it,
key for key, in ``esmfold2_hf_checkpoint_test.py`` -- a blind reversal of their
table does not work, because several of their rules map different sources onto
the same name and only the surrounding context tells them apart.
"""

import dataclasses
import re
import typing
from collections.abc import Iterable

import torch

from esm.models.esmc.checkpoint_layout import published_to_native_subtree

_ESMC_PREFIX = "esmc."

#: Names that appear only in one layout. Config metadata is not a reliable
#: discriminator -- a hand-converted repo can carry either -- but the tensors
#: cannot lie about how they are packed.
_HF_MARKERS = (".layers.", ".self_attn.", ".mlp.gate_up_proj.")
_NATIVE_MARKERS = (".blocks.", ".attn.", ".ffn.")

#: Their name -> ours, applied in order. Several are scoped: upstream reuses one
#: name (``mlp.gate_up_proj``, ``o_proj``) for modules we spell three ways.
_KEY_RULES = (
    # The trunk's atom encoder was renamed and re-parented.
    (r"^input_embedder\.atom_encoder\.", "inputs_embedder.atom_attention_encoder."),
    (r"^input_embedder\.pair_init_", "z_init_"),
    (r"^input_embedder\.", ""),
    # The diffusion module regained the level upstream flattened away.
    (
        r"^structure_head\.coords_linear\.",
        "structure_head.diffusion_module.atom_encoder.coords_linear.",
    ),
    (r"^structure_head\.(?!diffusion_module\.)", "structure_head.diffusion_module."),
    (
        r"^structure_head\.diffusion_module\.single_",
        "structure_head.diffusion_module.s_",
    ),
    # Diffusion conditioning: their single/pair are our s/z.
    (r"\.conditioning\.pair_transition_(\d+)\.", r".conditioning.z_transitions.\1."),
    (r"\.conditioning\.single_transition_(\d+)\.", r".conditioning.s_transitions.\1."),
    (r"\.conditioning\.pair_", ".conditioning.z_"),
    (r"\.conditioning\.single_", ".conditioning.s_"),
    # Alone among the transitions, these two never fused gate and up.
    (r"(_transitions\.\d+)\.mlp\.gate_up_proj", r"\1.a_proj"),
    (r"(_transitions\.\d+)\.mlp\.down_proj", r"\1.out_proj"),
    (r"\.fourier\.frequencies$", ".fourier.w"),
    (r"\.fourier\.phases$", ".fourier.b"),
    # Atom transformers lost a level and took the standard attention/MLP names.
    (
        r"(atom_(?:attention_)?(?:encoder|decoder))\.layers\.(\d+)\.",
        r"\1.atom_transformer.blocks.\2.",
    ),
    (r"(\.atom_transformer\.blocks\.\d+)\.self_attn\.", r"\1.attn."),
    (r"(\.atom_transformer\.blocks\.\d+)\.mlp\.gate_up_proj", r"\1.ffn.w_up"),
    (r"(\.atom_transformer\.blocks\.\d+)\.mlp\.down_proj", r"\1.ffn.w_down"),
    (r"(\.atom_transformer\.blocks\.\d+)\.adaln_linear", r"\1.adaln_modulation.1"),
    # One layer list upstream; two parallel stacks here.
    (
        r"\.token_transformer\.layers\.(\d+)\.input_layernorm\.",
        r".token_transformer.attn_blocks.\1.adaln.s_",
    ),
    (
        r"\.token_transformer\.layers\.(\d+)\.post_attention_layernorm\.",
        r".token_transformer.transition_blocks.\1.adaln.s_",
    ),
    (
        r"\.token_transformer\.layers\.(\d+)\.attn_gate\.",
        r".token_transformer.attn_blocks.\1.out_gate.",
    ),
    (
        r"\.token_transformer\.layers\.(\d+)\.mlp_gate\.",
        r".token_transformer.transition_blocks.\1.output_gate.",
    ),
    (
        r"\.token_transformer\.layers\.(\d+)\.mlp\.gate_up_proj",
        r".token_transformer.transition_blocks.\1.lin_swish",
    ),
    (
        r"\.token_transformer\.layers\.(\d+)\.mlp\.down_proj",
        r".token_transformer.transition_blocks.\1.lin_out",
    ),
    (
        r"\.token_transformer\.layers\.(\d+)\.(pair_norm|pair_bias_proj)\.",
        r".token_transformer.attn_blocks.\1.\2.",
    ),
    (
        r"\.token_transformer\.layers\.(\d+)\.self_attn\.",
        r".token_transformer.attn_blocks.\1.",
    ),
    (r"\.adaln\.s_cond_norm\.weight$", ".adaln.s_scale"),
    (r"(\.adaln\.s_(?:gate|shift))_proj", r"\1"),
    (r"(\.attn_blocks\.\d+\.)gate_proj", r"\1g_proj"),
    # Parcae is a submodule upstream and loose attributes here.
    (r"^parcae\.output_stack\.layers\.", "parcae_coda.blocks."),
    (r"^parcae\.out_proj", "parcae_readout"),
    (r"^parcae\.input_matrix_continuous", "parcae_b_cont"),
    (r"^parcae\.log_state_decay", "parcae_log_a"),
    (r"^parcae\.", "parcae_"),
    # Pair stacks; the triangle kernels keep the engine level upstream drops.
    (r"\.layers\.(\d+)\.", r".blocks.\1."),
    (r"\.(tri_mul_(?:in|out))\.", r".\1._engine."),
    (r"\.mlp\.gate_up_proj", ".ffn.w12"),
    (r"\.mlp\.down_proj", ".ffn.w3"),
    # The MSA encoder's W-prefixed projections and its two-part bias module.
    (
        r"\.msa_pair_weighted_averaging\.bias_norm\.",
        ".msa_pair_weighted_averaging.compute_bias.0.",
    ),
    (
        r"\.msa_pair_weighted_averaging\.bias_proj\.",
        ".msa_pair_weighted_averaging.compute_bias.1.",
    ),
    (
        r"\.msa_pair_weighted_averaging\.(gate|v)_proj\.",
        r".msa_pair_weighted_averaging.W\1.",
    ),
    (r"\.msa_pair_weighted_averaging\.o_proj\.", ".msa_pair_weighted_averaging.Wout."),
    (r"\.outer_product_mean\.input_proj\.", ".outer_product_mean.W."),
    (r"\.outer_product_mean\.output_proj\.", ".outer_product_mean.Wout."),
    # The confidence head's input embedder is flat here, its layernorms short.
    (
        r"^confidence_head\.input_embedder\.single_inputs_norm",
        "confidence_head.s_inputs_norm",
    ),
    (r"^confidence_head\.input_embedder\.single_to_pair", "confidence_head.s_to_z"),
    (r"^confidence_head\.input_embedder\.pair_norm", "confidence_head.z_norm"),
    (r"^confidence_head\.(\w+)_layernorm\.", r"confidence_head.\1_ln."),
    # The LM adapter's two Sequentials, which upstream gave names.
    (r"^language_model\.pair_input_norm", "language_model.base_z_linear.0"),
    (r"^language_model\.pair_proj", "language_model.base_z_linear.1"),
    (r"^language_model\.pair_output_norm", "language_model.base_z_mlp.1"),
    (
        r"^language_model\.single_to_pair\.output_fc1",
        "language_model.base_z_mlp.0.output_mlp.0",
    ),
    (
        r"^language_model\.single_to_pair\.output_fc2",
        "language_model.base_z_mlp.0.output_mlp.2",
    ),
    (r"^language_model\.single_to_pair\.", "language_model.base_z_mlp.0."),
    (r"^language_model\.layer_weights$", "language_model.base_z_combine"),
    # Attention projections: upstream splits what we pack.
    (r"\.attn\.[qkv]_proj\.weight$", ".attn.Wqkv.weight"),
    (r"\.attn_blocks\.(\d+)\.[kv]_proj\.weight$", r".attn_blocks.\1.kv_proj.weight"),
    (r"\.o_proj\.weight$", ".out_proj.weight"),
)

#: Fused destination -> the upstream parts it concatenates, in row order.
_PACKED = {"attn.Wqkv.weight": ("q", "k", "v"), "kv_proj.weight": ("k", "v")}


def to_native_key(key: str) -> str:
    """One upstream parameter name, spelled the way this implementation does."""
    for pattern, replacement in _KEY_RULES:
        key = re.sub(pattern, replacement, key)
    return key


def is_hf_layout(keys: Iterable[str]) -> bool:
    """Whether these state-dict keys are in the upstream port's layout.

    Raises when the keys look like neither, rather than guessing and loading a
    model full of randomly initialised parameters.
    """
    keys = list(keys)
    trunk = [k for k in keys if not k.startswith(_ESMC_PREFIX)]
    hf = sum(any(m in k for m in _HF_MARKERS) for k in trunk)
    native = sum(any(m in k for m in _NATIVE_MARKERS) for k in trunk)
    if not hf and not native:
        raise ValueError(
            "checkpoint matches neither the ESMFold2 nor the HuggingFace-port "
            f"tensor layout; unrecognised keys include {sorted(trunk)[:5]}"
        )
    return hf > native


def hf_state_dict_to_native(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite an upstream-port state dict onto this implementation's layout.

    Fails rather than dropping: two tensors landing on one name, or a packed
    projection missing a part, is a quietly half-loaded model.
    """
    out: dict[str, torch.Tensor] = {}
    packed: dict[str, tuple[tuple[str, ...], dict[str, torch.Tensor]]] = {}

    def emit(key: str, tensor: torch.Tensor) -> None:
        if key in out:
            raise ValueError(f"two checkpoint tensors both map onto {key!r}")
        out[key] = tensor

    for key, tensor in raw.items():
        if key.startswith(_ESMC_PREFIX):
            emit(key, tensor)
            continue
        native = to_native_key(key)
        order = next((o for s, o in _PACKED.items() if native.endswith(s)), None)
        if order is not None:
            # The part is named by the projection we replaced, and the row order
            # of the packed weight follows the model's own chunk order.
            part = key.rsplit(".", 2)[-2][0]
            parts = packed.setdefault(native, (order, {}))[1]
            if part in parts:
                raise ValueError(f"two checkpoint tensors both map onto {native!r}")
            parts[part] = tensor
        elif native.endswith(".a_proj.weight"):
            gate, up = torch.chunk(tensor, 2, dim=0)
            emit(native, gate.clone())
            emit(native.replace(".a_proj.", ".b_proj."), up.clone())
        else:
            emit(native, tensor)

    for native, (order, parts) in packed.items():
        if set(parts) != set(order):
            raise ValueError(
                f"{native!r} is missing parts: {sorted(set(order) - set(parts))}"
            )
        emit(native, torch.cat([parts[p] for p in order], dim=0))
    return published_to_native_subtree(out, _ESMC_PREFIX)


def drop_undeclared(config_class, values: dict) -> dict:
    """Keep only the fields ``config_class`` declares, recursing into nested configs.

    Upstream carries widths we derive (``head_dim``, ``intermediate_size``) and
    ``PretrainedConfig`` boilerplate (``model_type``, ``id2label``). HF's own
    ``AutoConfig`` accepts extras at every level, so dropping ours keeps one
    superset ``config.json`` loadable both ways. Silent: there is nothing here a
    caller could act on.
    """
    declared = typing.get_type_hints(config_class)
    kept = {}
    for name, value in values.items():
        if name not in declared:
            continue
        nested = declared[name]
        kept[name] = (
            drop_undeclared(nested, value)
            if isinstance(value, dict) and dataclasses.is_dataclass(nested)
            else value
        )
    return kept


__all__ = [
    "drop_undeclared",
    "hf_state_dict_to_native",
    "is_hf_layout",
    "to_native_key",
]
