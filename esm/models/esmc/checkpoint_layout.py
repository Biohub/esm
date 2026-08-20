"""Translation between the published ESMC checkpoint layout and this
implementation's in-memory layout.

Published checkpoints use one projection per tensor — ``q_proj``, ``k_proj``,
``v_proj``, ``gate_proj``, ``up_proj`` — which is the layout ``transformers``
reads and the only one its tensor-parallel plan can shard.

This implementation instead keeps the projections fused, because
``transformer_engine``'s ``LayerNormLinear`` / ``LayerNormMLP`` fuse each
LayerNorm into the GEMM that follows it and take a single packed weight. That
fusion is what the fp8 path runs on, so the module tree keeps Transformer
Engine's parameter names and the two layouts are reconciled here instead.

The conversion is exact in both directions: a fused weight is the row-wise
concatenation of its parts, so ``cat``/``chunk`` along dim 0 round-trips
bit-for-bit and no tensor is reshaped, transposed or reordered.

The two key vocabularies are disjoint, so both functions convert what they
recognise and pass everything else through untouched. That keeps checkpoints
already saved in either layout loadable.
"""

import re

import torch

_ENCODER_PREFIX = "esmc."

# Published name -> in-memory name, for keys directly under the encoder root.
_ROOT = {
    "embed_tokens.weight": "embed.weight",
    "norm.weight": "transformer.norm.weight",
}

# The masked-LM head is an ``nn.Sequential`` here, so its children are indices.
_LM_HEAD = {
    "lm_head.dense.weight": "lm_head.0.weight",
    "lm_head.dense.bias": "lm_head.0.bias",
    "lm_head.layer_norm.weight": "lm_head.2.weight",
    "lm_head.layer_norm.bias": "lm_head.2.bias",
    "lm_head.decoder.weight": "lm_head.3.weight",
    "lm_head.decoder.bias": "lm_head.3.bias",
}

# Published name -> in-memory name, within one transformer block. The two
# LayerNorms live inside the fused modules they feed rather than beside them.
_BLOCK = {
    "input_layernorm.weight": "attn.layernorm_qkv.layer_norm_weight",
    "input_layernorm.bias": "attn.layernorm_qkv.layer_norm_bias",
    "self_attn.q_norm.weight": "attn.q_ln.weight",
    "self_attn.k_norm.weight": "attn.k_ln.weight",
    "self_attn.o_proj.weight": "attn.out_proj.weight",
    "post_attention_layernorm.weight": "ffn.layer_norm_weight",
    "post_attention_layernorm.bias": "ffn.layer_norm_bias",
    "mlp.down_proj.weight": "ffn.fc2_weight",
}

# In-memory fused weight -> the published parts it concatenates, in row order.
# ``EsmcLayerNormMLP.forward`` splits fc1 as ``x1, x2 = chunk(2)`` and applies
# ``silu(x1) * x2``, so the gate rows come first; attention slices q, k, v in
# that order.
_FUSED = {
    "attn.layernorm_qkv.weight": (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
    ),
    "ffn.fc1_weight": ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
}

_PUBLISHED_BLOCK = re.compile(r"^layers\.(\d+)\.(.+)$")
_NATIVE_BLOCK = re.compile(r"^transformer\.blocks\.(\d+)\.(.+)$")


def _split_encoder_prefix(key: str) -> tuple[str, str]:
    """Return ``(prefix, remainder)``; the prefix is empty for a bare encoder."""
    if key.startswith(_ENCODER_PREFIX):
        return _ENCODER_PREFIX, key[len(_ENCODER_PREFIX) :]
    return "", key


def published_to_native_subtree(
    raw: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """Translate only the keys under ``prefix``, leaving the rest untouched.

    Used when an ESMC encoder is bundled inside a larger checkpoint, so the host
    model's own keys can never be caught by the ESMC key patterns.
    """
    inner = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    out = {k: v for k, v in raw.items() if not k.startswith(prefix)}
    out.update({prefix + k: v for k, v in published_to_native(inner).items()})
    return out


def published_to_native(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite a published checkpoint onto this implementation's layout.

    Concatenates the separate q/k/v and gate/up projections into the fused
    weights Transformer Engine expects. Keys already in the in-memory layout,
    and keys belonging to neither vocabulary, are returned unchanged.
    """
    out: dict[str, torch.Tensor] = {}
    # (prefix, block index) -> {published leaf: tensor}, gathered so the parts of
    # each fused weight can be concatenated once all of them have been seen.
    parts: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    part_leaves = {leaf for leaves in _FUSED.values() for leaf in leaves}

    for key, value in raw.items():
        prefix, rest = _split_encoder_prefix(key)
        if rest in _ROOT:
            out[prefix + _ROOT[rest]] = value
            continue
        if key in _LM_HEAD:
            out[_LM_HEAD[key]] = value
            continue
        match = _PUBLISHED_BLOCK.match(rest)
        if match is None:
            out[key] = value
            continue
        index, leaf = match.group(1), match.group(2)
        block = f"{prefix}transformer.blocks.{index}."
        if leaf in _BLOCK:
            out[block + _BLOCK[leaf]] = value
        elif leaf in part_leaves:
            parts.setdefault((prefix, index), {})[leaf] = value
        else:
            out[key] = value

    for (prefix, index), found in parts.items():
        block = f"{prefix}transformer.blocks.{index}."
        for fused, leaves in _FUSED.items():
            present = [leaf for leaf in leaves if leaf in found]
            if not present:
                continue
            if len(present) != len(leaves):
                raise ValueError(
                    f"block {index} has an incomplete set of projections for "
                    f"{fused!r}: found {present}, need {list(leaves)}"
                )
            out[block + fused] = torch.cat([found[leaf] for leaf in leaves], dim=0)
    return out


def native_to_published(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite this implementation's state dict onto the published layout.

    The exact inverse of :func:`published_to_native`: splits each fused weight
    back into its parts along dim 0.
    """
    root = {native: pub for pub, native in _ROOT.items()}
    lm_head = {native: pub for pub, native in _LM_HEAD.items()}
    block_leaves = {native: pub for pub, native in _BLOCK.items()}

    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        prefix, rest = _split_encoder_prefix(key)
        if rest in root:
            out[prefix + root[rest]] = value
            continue
        if key in lm_head:
            out[lm_head[key]] = value
            continue
        match = _NATIVE_BLOCK.match(rest)
        if match is None:
            out[key] = value
            continue
        index, leaf = match.group(1), match.group(2)
        block = f"{prefix}layers.{index}."
        if leaf in block_leaves:
            out[block + block_leaves[leaf]] = value
        elif leaf in _FUSED:
            leaves = _FUSED[leaf]
            # .clone(): safetensors refuses to write tensors that share storage.
            chunks = torch.chunk(value, len(leaves), dim=0)
            for name, chunk in zip(leaves, chunks, strict=True):
                out[block + name] = chunk.clone()
        else:
            out[key] = value
    return out
