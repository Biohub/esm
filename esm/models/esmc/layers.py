"""ESMC-specific transformer layers.

RoPE, QK-LayerNorm attention, SwiGLU feed-forward and the transformer stack used
by the ESMC encoder. Each fused module (Transformer Engine LayerNorm+Linear /
LayerNorm+MLP, xformers / flash-attn attention, Triton RoPE) has a pure-PyTorch
fallback with matching parameter names so state dicts are interchangeable.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from esm.layers.rotary import apply_rotary_emb_torch
from esm.models.esmc.kernels import (
    FLASH_ATTN_INSTALLED,
    FLASH_ATTN_ROTARY_INSTALLED,
    XFORMERS_INSTALLED,
    apply_triton_rotary,
    flash_attn_func,
    flash_attn_varlen_qkvpacked_func,
    te,
    xops,
)


class EsmcRotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE) as used by ESMC.

    Behaviourally this matches :class:`RotaryEmbedding`, with two differences
    that are load-bearing for ESMC's numerics:

    * The ``inv_freq`` buffer is recomputed on every device move and kept in
      fp32 even when the module is cast to bf16/fp16. CPU and CUDA ``pow``
      differ by ~1 fp32 ULP, which compounds across the deep attention stack.
    * The ``forward`` selects the Flash-Attention Triton RoPE kernel when it is
      available and the tensors live on CUDA, otherwise the pure-PyTorch path.

    Parameters
    ----------
    dim : int
        Size of a single attention head.
    base : float
        Frequency base for the sinusoidal positions.
    interleaved : bool
        If ``True`` rotate adjacent pairs (GPT-J style) instead of splitting
        the head dimension in half (GPT-NeoX style).
    scale_base : float | None
        Enables XPos scaling when set. ESMC does not use it; the ``forward``
        raises if it is configured.
    scaling_factor : float
        Linear scaling factor applied to position indices.
    pos_idx_in_fp32 : bool
        Compute position indices in fp32 to avoid bf16 rounding at long
        sequence lengths.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: float | None = None,
        scaling_factor: float = 1.0,
        pos_idx_in_fp32: bool = True,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        self.pos_idx_in_fp32 = pos_idx_in_fp32

        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._cos_k_cached: torch.Tensor | None = None
        self._sin_k_cached: torch.Tensor | None = None

        self.reset_parameters(device=device)

    def reset_parameters(self, device=None) -> None:
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        arange = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
        scale = (
            (arange + 0.4 * self.dim) / (1.4 * self.dim)
            if self.scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

    def _compute_inv_freq(self, device=None) -> torch.Tensor:
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen: int, device=None, dtype=None) -> None:
        if self.inv_freq.is_meta:
            self.reset_parameters(device=device)
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = (
                    torch.arange(seqlen, device=device, dtype=torch.float32)
                    / self.scaling_factor
                )
                inv_freq = (
                    self.inv_freq.to(torch.float32)
                    if self.inv_freq.dtype != torch.float32
                    else self.inv_freq
                )
            else:
                t = (
                    torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)  # ty:ignore[no-matching-overload]
                    / self.scaling_factor
                )
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)  # ty:ignore[invalid-argument-type]

            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                _scale: torch.Tensor = self.scale  # ty:ignore[invalid-assignment]
                power = (
                    torch.arange(seqlen, dtype=_scale.dtype, device=_scale.device)
                    - seqlen // 2
                ) / self.scale_base  # ty:ignore[unsupported-operator]
                scale = _scale.to(device=power.device) ** power.unsqueeze(-1)
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def _apply(self, fn, recurse=True):
        if self.inv_freq.is_meta:
            self.reset_parameters(device="cpu")
        result = super()._apply(fn, recurse=recurse)
        new_inv_freq = self._compute_inv_freq(device=self.inv_freq.device)
        self.register_buffer("inv_freq", new_inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        return result

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seqlen_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Parameters
        ----------
        q, k : torch.Tensor
            Tensors of shape ``(batch, seqlen, n_heads, head_dim)``.
        seqlen_offset : int
            Offset used in incremental decoding.
        """
        self._update_cos_sin_cache(
            q.shape[1] + seqlen_offset, device=q.device, dtype=q.dtype
        )
        assert self._cos_cached is not None and self._sin_cached is not None

        if self.scale is not None:
            raise NotImplementedError("XPos scaling is not supported in this path.")

        cos = self._cos_cached[seqlen_offset:]
        sin = self._sin_cached[seqlen_offset:]

        if FLASH_ATTN_ROTARY_INSTALLED and q.device.type == "cuda":
            assert apply_triton_rotary is not None
            q_rot = apply_triton_rotary(q, cos, sin, interleaved=self.interleaved)
            k_rot = apply_triton_rotary(k, cos, sin, interleaved=self.interleaved)
        else:
            q_rot = apply_rotary_emb_torch(q, cos, sin, self.interleaved)
            k_rot = apply_rotary_emb_torch(k, cos, sin, self.interleaved)
        return q_rot, k_rot


class EsmcTritonRotaryEmbedding(EsmcRotaryEmbedding):
    """RoPE variant that delegates to the Flash-Attention Triton kernel.

    Used inside :class:`EsmcFlashMultiHeadAttention` when Flash Attention 2 is
    available. The ``forward`` signature differs from
    :class:`EsmcRotaryEmbedding` because Flash Attention packs Q, K, V together.
    """

    def forward(
        self, qkv: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        """Apply RoPE in-place to a packed ``(N, 3, n_heads, head_dim)`` tensor."""
        self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        assert self._cos_cached is not None and self._sin_cached is not None
        assert apply_triton_rotary is not None

        apply_triton_rotary(
            qkv[:, 0],
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            inplace=True,
        )
        apply_triton_rotary(
            qkv[:, 1],
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            inplace=True,
        )
        return qkv


def esmc_swiglu_hidden_dim(expansion_ratio: float, d_model: int) -> int:
    """Round the hidden dim to the nearest multiple of 256 after expansion."""
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class EsmcLayerNormMLP(nn.Module):
    """LayerNorm + SwiGLU MLP (no bias), as used by ESMC.

    Shares the parameter names ``layer_norm_weight``, ``layer_norm_bias``,
    ``fc1_weight`` and ``fc2_weight`` with Transformer Engine's fused
    ``LayerNormMLP`` so the two are state-dict compatible: the fused kernel is
    used when available and this pure-PyTorch module is the fallback.
    """

    def __init__(
        self, hidden_size: int, ffn_hidden_size: int, eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.eps = eps
        self.layer_norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.layer_norm_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc1_weight = nn.Parameter(torch.empty(2 * ffn_hidden_size, hidden_size))
        self.fc2_weight = nn.Parameter(torch.empty(hidden_size, ffn_hidden_size))
        nn.init.normal_(self.fc1_weight, std=0.02)
        nn.init.normal_(self.fc2_weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        x = F.layer_norm(
            x,
            (self.hidden_size,),
            self.layer_norm_weight,
            self.layer_norm_bias,
            self.eps,
        )
        x = F.linear(x, self.fc1_weight)
        x1, x2 = x.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        return F.linear(x, self.fc2_weight)


def esmc_swiglu_ln_ffn(
    d_model: int, expansion_ratio: float, bias: bool, use_te: bool
) -> nn.Module:
    """LayerNorm + SwiGLU MLP, fused via Transformer Engine when ``use_te``."""
    assert not bias, "ESMC was trained with bias=False; bias=True not supported"
    hidden = esmc_swiglu_hidden_dim(expansion_ratio, d_model)
    if use_te:
        assert te is not None
        return te.LayerNormMLP(
            hidden_size=d_model,
            ffn_hidden_size=hidden,
            bias=bias,
            activation="swiglu",
            init_method=None,
            output_layer_init_method=None,
        )
    return EsmcLayerNormMLP(hidden_size=d_model, ffn_hidden_size=hidden)


def esmc_gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool) -> nn.Sequential:
    """LayerNorm + GELU MLP."""
    hidden = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden, bias=bias),
        nn.GELU(),
        nn.Linear(hidden, d_model, bias=bias),
    )


class EsmcLayerNormLinear(nn.Module):
    """LayerNorm followed by a Linear projection (no bias).

    Shares the parameter names ``layer_norm_weight``, ``layer_norm_bias`` and
    ``weight`` with Transformer Engine's fused ``LayerNormLinear`` so the two
    are state-dict compatible: the fused kernel is used when available and this
    pure-PyTorch module is the fallback.
    """

    def __init__(self, d_in: int, d_out: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_in = d_in
        self.eps = eps
        self.layer_norm_weight = nn.Parameter(torch.ones(d_in))
        self.layer_norm_bias = nn.Parameter(torch.zeros(d_in))
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(
            x, (self.d_in,), self.layer_norm_weight, self.layer_norm_bias, self.eps
        )
        return F.linear(x, self.weight)


def make_esmc_attn_layernorm_qkv(d_model: int, bias: bool, use_te: bool) -> nn.Module:
    """LayerNorm + fused QKV projection, fused via Transformer Engine when
    ``use_te``."""
    assert not bias, "ESMC was trained with bias=False; bias=True not supported"
    if use_te:
        assert te is not None
        return te.LayerNormLinear(d_model, d_model * 3, bias=bias, init_method=None)
    return EsmcLayerNormLinear(d_model, d_model * 3)


def make_esmc_attn_out_proj(d_model: int, bias: bool, use_te: bool) -> nn.Module:
    """Attention output projection, fused via Transformer Engine when ``use_te``."""
    if use_te:
        assert te is not None
        return te.Linear(d_model, d_model, bias=bias, init_method=None)
    return nn.Linear(d_model, d_model, bias=bias)


def esmc_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    n_heads: int,
    d_head: int,
    seq_id: torch.Tensor | None,
) -> torch.Tensor:
    """Scaled dot-product attention with an optional boolean mask.

    ``seq_id`` is the mask itself, broadcastable to ``(B, heads, L, L)``, with
    ``True`` meaning attend; :meth:`EsmcModel.forward` builds it. ``None`` means
    every position attends to every other, which is what lets the fused kernels
    take over.

    Dispatches in order of preference:

    1. xformers ``memory_efficient_attention`` - preferred fused kernel,
       requires ``xformers``, unmasked only.
    2. Flash Attention 2 (``flash_attn_func``) - secondary fused kernel,
       requires ``flash-attn``, unmasked only, fp16 / bf16 only.
    3. PyTorch's ``F.scaled_dot_product_attention`` - last-resort path; the only
       one that takes a mask, and the only fp32 one.
    """
    if seq_id is None and XFORMERS_INSTALLED and q.is_cuda:
        assert xops is not None
        b, s, _ = q.shape
        q4 = q.view(b, s, n_heads, d_head)
        k4 = k.view(b, s, n_heads, d_head)
        v4 = v.view(b, s, n_heads, d_head)
        context = xops.memory_efficient_attention(
            q4, k4, v4, attn_bias=None, scale=d_head**-0.5
        )
        return context.reshape(b, s, n_heads * d_head)
    if (
        seq_id is None
        and FLASH_ATTN_INSTALLED
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
    ):
        assert flash_attn_func is not None
        b, s, _ = q.shape
        q4 = q.view(b, s, n_heads, d_head)
        k4 = k.view(b, s, n_heads, d_head)
        v4 = v.view(b, s, n_heads, d_head)
        context = flash_attn_func(q4, k4, v4, dropout_p=0.0, softmax_scale=d_head**-0.5)
        return context.reshape(b, s, n_heads * d_head)
    b, s, _ = q.shape
    q = q.view(b, s, n_heads, -1).transpose(1, 2)
    k = k.view(b, s, n_heads, -1).transpose(1, 2)
    v = v.view(b, s, n_heads, -1).transpose(1, 2)
    context = F.scaled_dot_product_attention(q, k, v, seq_id)
    _, h, _, d_out = context.shape
    return context.transpose(1, 2).reshape(b, s, h * d_out)


class EsmcMultiHeadAttention(nn.Module):
    """Multi-head self-attention with QK LayerNorm and RoPE, as used by ESMC.

    Parameters
    ----------
    d_model : int
        Model hidden dimension.
    n_heads : int
        Number of attention heads.
    bias : bool
        Whether to use bias in the linear layers (ESMC uses ``False``).
    qk_layernorm : bool
        Whether to apply LayerNorm to queries and keys before computing scores.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
        use_te: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        assert not bias, "ESMC was trained with bias=False; bias=True not supported"
        self.layernorm_qkv = make_esmc_attn_layernorm_qkv(d_model, bias, use_te)
        self.out_proj = make_esmc_attn_out_proj(d_model, bias, use_te)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = EsmcRotaryEmbedding(d_model // n_heads)

    def _apply_rotary(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        seq_id: torch.Tensor | None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(context, attn_weights)``.

        ``attn_weights`` is ``None`` unless ``output_attentions=True`` - the
        fused SDPA backends don't expose attention probabilities, so capturing
        them forces a materialized ``softmax(Q @ K.T / sqrt(d)) @ V`` path with
        shape ``(B, H, L, L)``.
        """
        qkv = self.layernorm_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = self.q_ln(q).to(q.dtype)
        k = self.k_ln(k).to(q.dtype)
        q, k = self._apply_rotary(q, k)

        b, s, _ = q.shape

        if output_attentions:
            # Manual SDPA so attention probabilities are observable.
            q4 = q.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
            k4 = k.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
            v4 = v.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
            scale = self.d_head**-0.5
            attn_scores = (q4 @ k4.transpose(-2, -1)) * scale
            if seq_id is not None:
                attn_scores = attn_scores.masked_fill(~seq_id, float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = (attn_weights @ v4).transpose(1, 2).reshape(b, s, -1)
            return self.out_proj(context), attn_weights

        context = esmc_scaled_dot_product_attention(
            q, k, v, n_heads=self.n_heads, d_head=self.d_head, seq_id=seq_id
        )
        return self.out_proj(context), None


class EsmcFlashMultiHeadAttention(EsmcMultiHeadAttention):
    """Flash-Attention 2 variant of :class:`EsmcMultiHeadAttention`."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
        use_te: bool = False,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            bias=bias,
            qk_layernorm=qk_layernorm,
            use_te=use_te,
        )
        self.rotary = EsmcTritonRotaryEmbedding(d_model // n_heads)

    def forward(
        self,
        x: torch.Tensor,
        seq_id: torch.Tensor | None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if output_attentions:
            raise ValueError(
                "output_attentions=True is not supported with "
                "attn_implementation='flash_attention_2'. "
                "Re-load the model with attn_implementation='sdpa' (or 'eager')."
            )
        assert seq_id is not None and seq_id.dtype == torch.bool

        seqlens = seq_id.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        max_seqlen = int(seqlens.max().item())

        qkv = self.layernorm_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = self.q_ln(q).to(q.dtype)
        k = self.k_ln(k).to(q.dtype)

        # q/k/v are 2D (T, D) here: the parent model unpads the batch before the
        # transformer stack to produce the varlen-flat layout that
        # ``flash_attn_varlen_qkvpacked_func`` requires.
        T = q.shape[0]
        qkv_packed = torch.stack([q, k, v], dim=1).view(T, 3, self.n_heads, self.d_head)
        qkv_packed = self.rotary(qkv_packed, cu_seqlens, max_seqlen)

        assert flash_attn_varlen_qkvpacked_func is not None
        context = flash_attn_varlen_qkvpacked_func(
            qkv_packed, cu_seqlens, max_seqlen, softmax_scale=self.d_head**-0.5
        )
        n_out, h_out, d_out = context.shape
        return (self.out_proj(context.reshape(n_out, h_out * d_out)), None)


class EsmcUnifiedTransformerBlock(nn.Module):
    """A single ESMC transformer block: pre-norm attention + pre-norm FFN with
    residual scaling.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_heads : int
        Number of attention heads.
    use_flash_attn : bool
        Use the Flash Attention 2 kernel if available.
    bias : bool
        Whether linear layers include bias terms (ESMC uses ``False``).
    expansion_ratio : float
        Hidden-dim expansion ratio for the FFN.
    residue_scaling_factor : float
        Scales residual connections to stabilise deep networks
        (``sqrt(n_layers / 36)`` is the ESM3 scheme).
    qk_layernorm : bool
        Whether to apply QK LayerNorm in attention.
    ffn_type : str
        Feed-forward activation: ``"swiglu"`` or ``"gelu"``.
    use_te : bool
        Build the Transformer Engine fused LayerNorm modules.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_flash_attn: bool = False,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1.0,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",
        use_te: bool = False,
    ):
        super().__init__()

        attn_cls = (
            EsmcFlashMultiHeadAttention if use_flash_attn else EsmcMultiHeadAttention
        )
        self.attn = attn_cls(
            d_model, n_heads, bias=bias, qk_layernorm=qk_layernorm, use_te=use_te
        )

        if ffn_type == "swiglu":
            self.ffn = esmc_swiglu_ln_ffn(d_model, expansion_ratio, bias, use_te)
        elif ffn_type == "gelu":
            self.ffn = esmc_gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(
                f"Unknown ffn_type: {ffn_type!r}. Choose 'swiglu' or 'gelu'."
            )

        self.scaling_factor = residue_scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor | None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, seq_len, d_model)``.
        sequence_id : torch.Tensor | None
            Chain-ID tensor used to restrict attention to tokens within the
            same chain. SDPA blocks accept an integer tensor (``-1`` marks
            padding); the flash-attn block takes a ``bool`` padding mask - the
            caller selects which. ``None`` skips chain-aware masking (fast path).
        output_attentions : bool
            When ``True``, also returns the per-head attention weights.
        """
        attn_out, attn_weights = self.attn(
            x, sequence_id, output_attentions=output_attentions
        )
        x = x + attn_out / self.scaling_factor
        x = x + self.ffn(x) / self.scaling_factor
        return x, attn_weights


class EsmcTransformerStack(nn.Module):
    """Stack of :class:`EsmcUnifiedTransformerBlock` layers with a final
    LayerNorm.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer blocks.
    scale_residue : bool
        When ``True`` apply ESM3 residue scaling ``sqrt(n_layers / 36)``.
    bias : bool
        Bias flag forwarded to every sub-module.
    qk_layernorm : bool
        QK LayerNorm flag forwarded to every block.
    ffn_type : str
        FFN activation type (``"swiglu"`` or ``"gelu"``).
    expansion_ratio : float
        FFN expansion ratio.
    use_flash_attn : bool
        Use the Flash Attention 2 kernel when available.
    use_te : bool
        Build the Transformer Engine fused LayerNorm modules.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        scale_residue: bool = True,
        bias: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",
        expansion_ratio: float = 8 / 3,
        use_flash_attn: bool = False,
        use_te: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                EsmcUnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    use_flash_attn=use_flash_attn,
                    residue_scaling_factor=math.sqrt(n_layers / 36)
                    if scale_residue
                    else 1.0,
                    expansion_ratio=expansion_ratio,
                    bias=bias,
                    qk_layernorm=qk_layernorm,
                    ffn_type=ffn_type,
                    use_te=use_te,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        layers_to_collect: list[int] | None = None,
        output_attentions: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...] | None,
    ]:
        """Run the full transformer stack.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, seq_len, d_model)``.
        sequence_id : torch.Tensor | None
            Optional chain-id tensor forwarded to each block.
        layers_to_collect : list[int] | None
            Layer indices (0-based pre-block inputs plus ``n_layers`` for the
            post-norm output) whose hidden states should be returned.
        output_attentions : bool
            When ``True``, collects the per-block attention weights.

        Returns
        -------
        tuple
            ``(post_norm, pre_norm, hidden_states, attentions)`` where
            ``hidden_states`` is a (possibly empty) tuple of tensors and
            ``attentions`` is a tuple of per-block ``(B, H, L, L)`` tensors or
            ``None`` when ``output_attentions`` is ``False``.
        """
        if layers_to_collect is None:
            layers_to_collect = []

        collected: list[torch.Tensor] = []
        all_attentions: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx in layers_to_collect:
                collected.append(x)
            x, attn_weights = block(x, sequence_id, output_attentions=output_attentions)
            if output_attentions and attn_weights is not None:
                all_attentions.append(attn_weights)

        norm_x = self.norm(x)
        if len(self.blocks) in layers_to_collect:
            collected.append(norm_x)

        attentions = tuple(all_attentions) if output_attentions else None
        return norm_x, x, tuple(collected), attentions
