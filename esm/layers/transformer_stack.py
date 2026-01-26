import math

import torch
import torch.nn as nn

from esm.layers.blocks import UnifiedTransformerBlock
from esm.utils.structure.affine3d import Affine3D


class TransformerStack(nn.Module):
    """
    A stack of transformer blocks used in the ESM-3 model. Each block is a UnifiedTransformerBlock,
    which can either be geometric attention or standard multi-head attention.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads.
        v_heads (int): The number of voting heads.
        n_layers (int): The number of transformer blocks in the stack.
        n_layers_geom (int, optional): The number of transformer blocks that use geometric attention.
        scale_residue (bool, optional): Whether to scale the residue connections in each transformer block.
        mask_and_zero_frameless (bool, optional): Whether to mask and zero frameless positions in the input.
            Only applies in the geometric attention blocks, which is conditioned on the structure
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        v_heads: int | None,
        n_layers: int,
        n_layers_geom: int = 1,
        scale_residue: bool = True,
        mask_and_zero_frameless: bool = False,
        bias: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",  # swiglu | gelu
        expansion_ratio: float = 8 / 3,
        use_flash_attn: bool = False,
        return_hidden_states: bool = False,
    ):
        super().__init__()
        self.return_hidden_states = return_hidden_states
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    v_heads=v_heads,
                    use_geom_attn=i < n_layers_geom,
                    use_flash_attn=use_flash_attn,
                    residue_scaling_factor=(
                        math.sqrt(n_layers / 36) if scale_residue else 1.0
                    ),
                    expansion_ratio=expansion_ratio,
                    mask_and_zero_frameless=mask_and_zero_frameless,
                    bias=bias,
                    qk_layernorm=qk_layernorm,
                    ffn_type=ffn_type,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        affine: Affine3D | None = None,
        affine_mask: torch.Tensor | None = None,
        chain_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the TransformerStack.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).
            sequence_id (torch.Tensor): The sequence ID tensor of shape (batch_size, sequence_length).
            affine (Affine3D | None): The affine transformation tensor or None.
            affine_mask (torch.Tensor | None): The affine mask tensor or None.
            chain_id (torch.Tensor): The protein chain tensor of shape (batch_size, sequence_length).
                Only used in geometric attention.

        Returns:
            post_norm: The output tensor of shape (batch_size, sequence_length, d_model).
            pre_norm: The embedding of shape (batch_size, sequence_length, d_model).
        """
        *batch_dims, _ = x.shape
        if chain_id is None:
            chain_id = torch.ones(size=batch_dims, dtype=torch.int64, device=x.device)
        hiddens = []
        for block in self.blocks:
            x = block(x, sequence_id, affine, affine_mask, chain_id)
            if self.return_hidden_states:
                hiddens.append(x)
        return self.norm(x), x, hiddens
