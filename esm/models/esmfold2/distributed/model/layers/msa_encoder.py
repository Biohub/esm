# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Distributed MSAEncoder block / stack for ESMFold2 (inference-only).

Mirrors the serial ``MSAEncoderBlock`` (modeling_esmfold2.py) exactly:

    pair = pair + outer_product_mean(m, msa_mask)
    if not final:
        m = m + msa_pair_weighted_averaging(m, pair, pair_mask)
        m = m + msa_transition(m)
    pair = pair + tri_mul_out(pair, mask=pair_mask)
    pair = pair + tri_mul_in(pair,  mask=pair_mask)
    pair = pair + pair_transition(pair)

All residuals are explicit (added here). The serial block uses
``modeling_esmfold2.PairTransition`` for both ``msa_transition`` and
``pair_transition`` — that class returns ``ffn(norm(x))`` WITHOUT a residual,
so we use the bare ``MSATransitionDistributed`` (not ``TransitionDistributed``,
which folds the residual in) and add the residual in the block.
"""

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from esm.models.esmfold2.distributed.comm import Ring2DComm
from esm.models.esmfold2.distributed.manager import DistributedManager
from esm.models.esmfold2.distributed.model.layers.layernorm import (
    LayerNormParamsReplicated,
)
from esm.models.esmfold2.distributed.model.layers.linear import LinearParamsReplicated
from esm.models.esmfold2.distributed.model.layers.outer_product_mean import (
    OuterProductMeanDistributed,
)
from esm.models.esmfold2.distributed.model.layers.pair_averaging import (
    MSAPairWeightedAveragingDistributed,
)
from esm.models.esmfold2.distributed.model.layers.triangular_mult import (
    TriangleMultiplicativeBlockDistributed,
)
from esm.models.esmfold2.model import MSAEncoder as SerialMSAEncoder
from esm.models.esmfold2.model import MSAEncoderBlock as SerialMSAEncoderBlock
from esm.models.esmfold2.model import PairTransition as SerialPairTransition


class MSATransitionDistributed(nn.Module):
    """Bare LayerNorm + SwiGLU FFN (no residual) on a sharded representation.

    Matches ``modeling_esmfold2.PairTransition.forward`` which returns
    ``ffn(norm(x))``; the residual is added by the calling block.
    """

    def __init__(self, layer: SerialPairTransition, device_mesh: DeviceMesh) -> None:
        super().__init__()
        if not isinstance(layer, SerialPairTransition):
            raise TypeError(f"layer must be PairTransition, got {type(layer).__name__}")
        self.norm = LayerNormParamsReplicated(layer.norm, device_mesh)
        self.w12 = LinearParamsReplicated(layer.ffn.w12, device_mesh)
        self.w3 = LinearParamsReplicated(layer.ffn.w3, device_mesh)
        self.hidden_features = layer.ffn.hidden_features

    def forward(self, x: DTensor) -> DTensor:
        normed = self.norm(x)
        x12 = self.w12(normed)
        x1, x2 = x12.split(self.hidden_features, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class MSAEncoderBlockDistributed(nn.Module):
    """Distributed MSAEncoderBlock.

    Parameters
    ----------
    layer:
        Serial MSAEncoderBlock to distribute.
    dist_manager:
        DistributedManager with the CP group and subgroups set up.
    """

    def __init__(
        self,
        layer: SerialMSAEncoderBlock,
        dist_manager: DistributedManager,
        comm: str = "gather",
    ) -> None:
        super().__init__()
        if not isinstance(layer, SerialMSAEncoderBlock):
            raise TypeError(
                f"layer must be MSAEncoderBlock, got {type(layer).__name__}"
            )
        mesh = dist_manager.device_mesh_subgroups
        self.is_final_block = layer.is_final_block

        self.outer_product_mean = OuterProductMeanDistributed(
            layer.outer_product_mean, dist_manager
        )
        if not self.is_final_block:
            self.msa_pair_weighted_averaging = MSAPairWeightedAveragingDistributed(
                layer.msa_pair_weighted_averaging, dist_manager, comm=comm
            )
            self.msa_transition = MSATransitionDistributed(layer.msa_transition, mesh)

        ring_comm_out = Ring2DComm(
            dist_manager.group["cp"],
            dist_manager.subgroups["cp"][0],
            dist_manager.layout_subgroups["cp"],
        )
        ring_comm_in = Ring2DComm(
            dist_manager.group["cp"],
            dist_manager.subgroups["cp"][0],
            dist_manager.layout_subgroups["cp"],
        )
        self.tri_mul_out = TriangleMultiplicativeBlockDistributed(
            layer.tri_mul_out._engine, mesh, ring_comm_out
        )
        self.tri_mul_in = TriangleMultiplicativeBlockDistributed(
            layer.tri_mul_in._engine, mesh, ring_comm_in
        )
        # Serial pair_transition is a (residual-free) PairTransition.
        self.pair_transition = MSATransitionDistributed(layer.pair_transition, mesh)

    def forward(
        self,
        m: DTensor,
        pair: DTensor,
        msa_attention_mask: DTensor,
        pair_attention_mask: Optional[DTensor] = None,
    ) -> tuple[DTensor, DTensor]:
        pair = pair + self.outer_product_mean(m, msa_attention_mask)
        if not self.is_final_block:
            m = m + self.msa_pair_weighted_averaging(m, pair, pair_attention_mask)
            m = m + self.msa_transition(m)
        pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, mask=pair_attention_mask)
        pair = pair + self.pair_transition(pair)
        return m, pair


class MSAEncoderDistributed(nn.Module):
    """Distributed MSAEncoder: ModuleList of MSAEncoderBlockDistributed.

    Parameters
    ----------
    encoder:
        Serial MSAEncoder module.
    dist_manager:
        DistributedManager with the CP group and subgroups set up.
    """

    def __init__(
        self,
        encoder: SerialMSAEncoder,
        dist_manager: DistributedManager,
        comm: str = "gather",
    ) -> None:
        super().__init__()
        if not isinstance(encoder, SerialMSAEncoder):
            raise TypeError(f"encoder must be MSAEncoder, got {type(encoder).__name__}")
        self.blocks = nn.ModuleList(
            [
                MSAEncoderBlockDistributed(block, dist_manager, comm=comm)  # ty:ignore[invalid-argument-type]
                for block in encoder.blocks  # type: ignore[arg-type]
            ]
        )

    def forward(
        self,
        m: DTensor,
        pair: DTensor,
        msa_attention_mask: DTensor,
        pair_attention_mask: DTensor,
    ) -> DTensor:
        for block in self.blocks:
            m, pair = block(m, pair, msa_attention_mask, pair_attention_mask)
        return pair
