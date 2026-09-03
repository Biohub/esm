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

"""Distributed confidence head for ESMFold2 (#6, inference-only).

Keeps the pair sharded ``(Shard(0), Shard(1), Shard(2))`` through the expensive
front half — z_base build, the nested (already CP-wrapped) FoldingTrunk, row
pooling, and the PAE/PDE heads — so the full ``L×L×d_pair`` pair (and the trunk
activations) is never resident on any rank. Only small tensors are gathered:

  * ``single`` (B, L, d_single) after row pooling, and
  * ``pae_logits`` / ``pde_logits`` (B, L, L, bins) for the output contract,

after which the **serial** ``ConfidenceHead._finish`` runs the atom-space
pLDDT/resolved heads and the pTM/ipTM/pair_chains reductions on the gathered
tensors — one source of truth for that fiddly code, replicated and cheap (it is
per-token / single-channel-L², not the d_pair pair).

The wrapper consumes the sharded pair DTensor directly from the recycle engine's
CP tail (no ``full_tensor()`` re-gather of ``z``), which is what removes the
confidence-phase peak.
"""

from math import lcm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from esm.models.esmfold2.distributed.model.layers.confidence_zbase import (
    ConfidenceZBaseDistributed,
)
from esm.models.esmfold2.distributed.model.layers.layernorm import (
    LayerNormParamsReplicated,
)
from esm.models.esmfold2.distributed.model.layers.linear import LinearParamsReplicated
from esm.models.esmfold2.distributed.model.layers.pair_init import (
    build_sharded_distogram_bins,
    build_sharded_pair_mask,
)
from esm.models.esmfold2.distributed.model.layers.row_attention_pooling import (
    RowAttentionPoolingDistributed,
)
from esm.models.esmfold2.layers import gather_rep_atom_coords


class ConfidenceHeadCPWrapper(nn.Module):
    """Distributed wrapper around the serial ``ConfidenceHead``.

    The nested ``folding_trunk`` is already replaced by a ``TrunkCPWrapper`` via
    ``wrap_model_with_cp_trunks`` (so ``forward_sharded`` is available). This
    wrapper holds the distributed z_base / row-pool / pae-pde-head pieces and
    delegates the back-half to ``head._finish``.
    """

    def __init__(self, head, dist_manager) -> None:
        super().__init__()
        self.head = head  # serial ConfidenceHead (nested trunk already CP-wrapped)
        self.device_mesh = dist_manager.device_mesh_subgroups
        self.shard_factor = lcm(self.device_mesh.size(1), self.device_mesh.size(2))

        self.zbase = ConfidenceZBaseDistributed(head, dist_manager)
        self.row_pool = RowAttentionPoolingDistributed(
            head.row_attention_pooling, dist_manager
        )
        self.pae_ln = LayerNormParamsReplicated(head.pae_ln, self.device_mesh)
        self.pae_head = LinearParamsReplicated(head.pae_head, self.device_mesh)
        self.pde_ln = LayerNormParamsReplicated(head.pde_ln, self.device_mesh)
        self.pde_head = LinearParamsReplicated(head.pde_head, self.device_mesh)

    def forward_sharded(
        self,
        z_dt: DTensor,
        n_orig: int,
        *,
        s_inputs: torch.Tensor,
        x_pred: torch.Tensor,
        distogram_atom_idx: torch.Tensor,
        token_attention_mask: torch.Tensor,
        atom_to_token: torch.Tensor,
        atom_attention_mask: torch.Tensor,
        asym_id: torch.Tensor,
        mol_type: torch.Tensor,
        num_diffusion_samples: int = 1,
        relative_position_encoding: torch.Tensor | None = None,
        token_bonds_encoding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if num_diffusion_samples != 1:
            raise NotImplementedError(
                "distributed confidence head supports num_diffusion_samples==1"
            )
        mesh = self.device_mesh
        Lpad = z_dt.shape[1]
        pad = Lpad - n_orig
        pair_dtype = torch.float32

        # --- distogram bins: sharded block cdist (never the full [B, N, N]) -----
        rep_idx = distogram_atom_idx.long()
        rep_coords = gather_rep_atom_coords(
            x_pred, rep_idx
        )  # (B, n, 3) — replicated, cheap
        bins_p = build_sharded_distogram_bins(
            rep_coords, self.head.boundaries, mesh
        )  # (S0, S1, S2) [B, Lpad, Lpad]

        # --- pad aux tensors to the shard factor (z_dt is already padded) -------
        def _pad_pair(t):
            # An already-sharded DTensor is padded (built at the shard factor) —
            # pass it through; only a full tensor needs padding here.
            if isinstance(t, DTensor):
                return t
            return F.pad(t, (0, 0, 0, pad, 0, pad)) if pad else t

        s_in_p = F.pad(s_inputs, (0, 0, 0, pad)) if pad else s_inputs
        relpos_p = (
            _pad_pair(relative_position_encoding)
            if relative_position_encoding is not None
            else None
        )
        tokb_p = (
            _pad_pair(token_bonds_encoding)
            if token_bonds_encoding is not None
            else None
        )
        mask_p = (
            F.pad(token_attention_mask.float(), (0, pad))
            if pad
            else (token_attention_mask.float())
        )

        # --- z_base (sharded) ---------------------------------------------------
        pair_dt = self.zbase(
            z_dt.to(pair_dtype), s_in_p, bins_p, relpos_p, tokb_p
        )  # (S0, S1, S2) fp32, [B, Lpad, Lpad, d_pair]

        # --- nested FoldingTrunk (add-back, like serial pair.add_(trunk(pair))) -
        # Pair mask built SHARDED from the (padded) token mask — the per-block outer
        # product, never a full [B, Lpad, Lpad] (matches the recycle / MSA masks).
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            pair_mask_dt = build_sharded_pair_mask(mask_p, mesh).to(torch.bfloat16)
            delta_dt = self.head.folding_trunk.forward_sharded(
                pair_dt.to(torch.bfloat16), pair_attention_mask=pair_mask_dt
            )
        pair_dt = pair_dt + delta_dt.to(pair_dtype)

        # --- row pooling -> single (gather small) -------------------------------
        single_dt = self.row_pool(pair_dt, mask_p)  # (S0,S1,R) [B,Lpad,d_single]
        single = single_dt.full_tensor()[:, :n_orig, :].contiguous()

        # --- PAE / PDE heads on the sharded pair; gather (sliced) logits --------
        pae_logits_dt = self.pae_head(self.pae_ln(pair_dt))
        pde_logits_dt = self.pde_head(self.pde_ln(pair_dt))
        del pair_dt
        pae_logits = pae_logits_dt.full_tensor()[:, :n_orig, :n_orig, :].contiguous()
        del pae_logits_dt
        pde_logits = pde_logits_dt.full_tensor()[:, :n_orig, :n_orig, :].contiguous()
        del pde_logits_dt

        # --- serial back-half (atom-space + pTM/ipTM), shared with forward ------
        return self.head._finish(
            single=single,
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            x_pred=x_pred,
            distogram_atom_idx=distogram_atom_idx,
            token_attention_mask=token_attention_mask,
            atom_to_token=atom_to_token,
            atom_attention_mask=atom_attention_mask,
            asym_id=asym_id,
            mol_type=mol_type,
            num_diffusion_samples=num_diffusion_samples,
        )
