import torch
import torch.nn as nn
import torch.nn.functional as F

from esm.layers.attention import FlashMultiHeadAttention, MultiHeadAttention
from esm.layers.geom_attention import GeometricReasoningOriginalImpl
from esm.utils.structure.affine3d import Affine3D


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(
            d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias
        ),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


def gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    hidden_dim = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim, bias=bias),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model, bias=bias),
    )


class UnifiedTransformerBlock(nn.Module):
    """
    A unified transformer block that can optionally incorporate geometric attention.

    This class defines a transformer block that can be configured to use geometric attention
    alongside the standard multi-head attention mechanism. It is designed to be a flexible
    component of transformer-based models, allowing for the integration of geometric reasoning.

    Parameters
    ----------
    d_model : int
        The dimensionality of the input and output features of the transformer block.
    n_heads : int
        The number of attention heads in the multi-head attention mechanism.
    n_layers : int
        The number of layers in the transformer block.
    use_geom_attn : bool, optional
        Whether to use geometric attention in addition to the standard multi-head attention. Defaults to False.
    v_heads : int, optional
        The number of heads to use for the geometric attention mechanism, if enabled. Must be specified if `use_geom_attn` is True.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_geom_attn: bool = False,
        use_plain_attn: bool = True,
        use_flash_attn: bool = False,
        v_heads: int | None = None,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1,
        mask_and_zero_frameless: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",  # swiglu | gelu
    ):
        super().__init__()
        self.use_plain_attn = use_plain_attn
        if self.use_plain_attn:
            if use_flash_attn:
                self.attn = FlashMultiHeadAttention(
                    d_model, n_heads, bias, qk_layernorm=qk_layernorm
                )
            else:
                self.attn = MultiHeadAttention(
                    d_model, n_heads, bias, qk_layernorm=qk_layernorm
                )
        self.use_geom_attn = use_geom_attn
        if self.use_geom_attn:
            if v_heads is None:
                raise ValueError("v_heads must be specified when use_geom_attn is True")
            self.geom_attn = GeometricReasoningOriginalImpl(
                c_s=d_model,
                v_heads=v_heads,
                bias=bias,
                mask_and_zero_frameless=mask_and_zero_frameless,
            )
        if ffn_type == "swiglu":
            self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        self.scaling_factor = residue_scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor,
        frames: Affine3D,
        frames_mask: torch.Tensor,
        chain_id: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for the UnifiedTransformerBlock.

        Parameters
        ----------
        x : torch.Tensor[float]
            Input tensor to the transformer block, typically the output from the previous layer.
        sequence_id : torch.Tensor[int]
            Tensor containing sequence IDs for each element in the batch, used for attention masking.
        frames : Affine3D
            Affine3D containing geometric frame information for geometric attention.
        frames_mask : torch.Tensor[bool]
            Boolean mask tensor indicating valid frames for geometric attention.
        chain_id : torch.Tensor[int]
            Tensor containing chain IDs for each element, used for attention masking in geometric attention.
        output_attentions : bool
            Whether to return attention weights of shape (batch_size, num_heads, seq_len, seq_len).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor | None]
            The output tensor after applying the transformer block operations, and optionally the
            attention weights.
        """
        attn_weights = None
        if self.use_plain_attn:
            r1, attn_weights = self.attn(
                x, sequence_id, output_attentions=output_attentions
            )
            x = x + r1 / self.scaling_factor

        if self.use_geom_attn:
            r2 = self.geom_attn(x, frames, frames_mask, sequence_id, chain_id)
            x = x + r2 / self.scaling_factor

        r3 = self.ffn(x) / self.scaling_factor
        x = x + r3

        return x, attn_weights
