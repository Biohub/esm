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

"""Distributed diffusion structure head for ESMFold2.

``DiffusionModuleCPWrapper`` is a drop-in for ``DiffusionModule`` that shards the
per-step diffusion working set: the conditioned pair ``z`` (via
``DiffusionConditioningDistributed``) and the token transformer's L×L attention
(via ``DiffusionTransformerDistributed``). The atom encoder/decoder and the
single-repr steps run **replicated** on full tensors (cheap 1-D / windowed work).
``wrap_model_with_cp_structure_head`` installs it by swapping
``structure_head.diffusion_module`` — the serial ``sample()`` loop (noise
schedule, augmentation) is untouched and keeps driving it.

Scope: ``num_diffusion_samples == 1`` (multi-sample batch expansion of the pair
bias is a TODO). The trunk pair ``z_trunk`` still arrives **full** from the
recycle gather, so it remains a per-step floor; the conditioned ``z`` and the
attention are what get sharded here. Removing the ``z_trunk`` floor needs the
pair kept sharded across the recycle→parcae→structure-head boundary (follow-on).
Inference-only.
"""

from math import lcm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

_PAIR_PL = [Shard(0), Shard(1), Shard(2)]
_ROW_PL = [Shard(0), Shard(1), Replicate()]


class DiffusionModuleCPWrapper(nn.Module):
    """Drop-in for ``DiffusionModule`` that distributes conditioning + token attn."""

    def __init__(self, serial_module: nn.Module, dist_manager) -> None:
        super().__init__()
        from esm.models.esmfold2.distributed.model.layers.diffusion_conditioning import (
            DiffusionConditioningDistributed,
        )
        from esm.models.esmfold2.distributed.model.layers.diffusion_transformer import (
            DiffusionTransformerDistributed,
        )
        from esm.models.esmfold2.layers import DiffusionModule as SerialDiffusionModule

        if not isinstance(serial_module, SerialDiffusionModule):
            raise TypeError(
                f"expected DiffusionModule, got {type(serial_module).__name__}"
            )
        self.serial = serial_module
        self.device_mesh = dist_manager.device_mesh_subgroups
        self.shard_factor = lcm(self.device_mesh.size(1), self.device_mesh.size(2))
        self.cond = DiffusionConditioningDistributed(
            serial_module.conditioning, self.device_mesh
        )
        self.token_transformer = DiffusionTransformerDistributed(
            serial_module.token_transformer
        )

    # ``DiffusionStructureHead.set_kernel_backend`` forwards to its
    # ``diffusion_module``, so this wrapper needs the same no-op hook the trunk
    # and MSA-encoder wrappers have: the distributed path runs its own kernels,
    # and a single-GPU backend choice must not reach it. Without this, calling
    # ``model.set_kernel_backend(...)`` after ``wrap_model_with_cp`` raises
    # AttributeError.
    def set_kernel_backend(self, _backend: str | None) -> None:
        return

    def set_chunk_size(self, _chunk_size: int | None) -> None:
        return

    def forward(
        self,
        *,
        x_noisy,
        t_hat,
        ref_pos,
        ref_charge,
        ref_mask,
        ref_element,
        ref_atom_name_chars,
        ref_space_uid,
        tok_idx,
        s_inputs,
        s_trunk,
        z_trunk,
        relative_position_encoding,
        asym_id,
        residue_index,
        entity_id,
        token_index,
        sym_id,
        sigma_data=None,
        token_attention_mask=None,
        num_diffusion_samples: int = 1,
        return_token_repr: bool = False,
        return_atom_repr: bool = False,
        inference_cache=None,
    ):
        if num_diffusion_samples != 1:
            raise NotImplementedError(
                "DiffusionModuleCPWrapper supports num_diffusion_samples==1 "
                "(pair-bias batch expansion not yet sharded)."
            )
        m = self.serial
        mesh = self.device_mesh
        bsz = x_noisy.shape[0]
        sigma = m.sigma_data if sigma_data is None else float(sigma_data)
        t = torch.as_tensor(t_hat, dtype=torch.float32, device=x_noisy.device).reshape(
            -1
        )
        if t.numel() == 1:
            t = t.expand(bsz)

        # z_trunk may arrive as a full tensor (pad here) or as an already-padded,
        # already-sharded DTensor from the CP tail (#7) — in which case the token
        # axis is its padded length and the other (token-space) tensors are padded
        # to match. Padded token rows carry no atoms (atom_to_token < L), so they
        # don't affect the atom-space coordinate output.
        zt_is_dtensor = isinstance(z_trunk, DTensor)
        if zt_is_dtensor:
            L = s_inputs.shape[1]  # original token length
            pad = z_trunk.shape[1] - L  # to the pre-padded sharded length
        else:
            L = z_trunk.shape[1]
            pad = (self.shard_factor - L % self.shard_factor) % self.shard_factor

        def _pad_pair(x):  # (B, L, L, C)
            return F.pad(x, (0, 0, 0, pad, 0, pad)) if pad else x

        def _pad_tok(x):  # (B, L, C)
            return F.pad(x, (0, 0, 0, pad)) if pad else x

        def _pad_mask(x):  # (B, L)
            return F.pad(x, (0, pad)) if pad else x

        # Step 1: conditioning -> full s, sharded (padded) z. Only materialise +
        # distribute the full z_trunk on the cache miss (first step); afterwards
        # the sharded z is reused from the cache.
        # relative_position_encoding may arrive already padded + sharded (from
        # PairInitDistributed) — use it directly instead of re-materialising the
        # full L×L tensor to distribute it.
        def _rp_to_dt():
            if isinstance(relative_position_encoding, DTensor):
                return relative_position_encoding
            return distribute_tensor(
                _pad_pair(relative_position_encoding).contiguous(), mesh, _PAIR_PL
            )

        z_cached = inference_cache is not None and "z_cp" in inference_cache
        if z_cached:
            zt_dt = rp_dt = None
        elif zt_is_dtensor:
            zt_dt = z_trunk  # already padded + sharded; do not re-distribute
            rp_dt = _rp_to_dt()
        else:
            zt_dt = distribute_tensor(_pad_pair(z_trunk).contiguous(), mesh, _PAIR_PL)
            rp_dt = _rp_to_dt()
        s, z_dt = self.cond(
            t_hat=t,
            s_inputs=s_inputs,
            z_trunk_dt=zt_dt,
            rel_pos_dt=rp_dt,
            sigma_data=sigma,
            num_diffusion_samples=num_diffusion_samples,
            inference_cache=inference_cache,
        )

        # Step 2: normalise noisy coords (replicated)
        denom = torch.sqrt(t * t + sigma * sigma)
        r_noisy = x_noisy / denom[:, None, None]

        # Step 3: atom encoder (replicated, full)
        a, q_skip, c_skip, p_skip, enc_int = m.atom_encoder(
            ref_pos=ref_pos,
            atom_attention_mask=ref_mask,
            ref_space_uid=ref_space_uid,
            ref_charge=ref_charge,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            atom_to_token=tok_idx,
            r_l=r_noisy,
            s_i=s_trunk,
            num_diffusion_samples=num_diffusion_samples,
            return_intermediates=return_atom_repr,
            inference_cache=inference_cache,
        )

        # Step 4: add conditioned s (replicated)
        a = a + m.s_to_token(m.s_step_norm(s))

        # Step 5: token transformer (distributed). Pad token axis; row-shard a/s/
        # mask; z_dt already 2-D sharded at padded length; gather + slice a back.
        a_dt = distribute_tensor(_pad_tok(a).contiguous(), mesh, _ROW_PL)
        s_dt = distribute_tensor(_pad_tok(s).contiguous(), mesh, _ROW_PL)
        mask_dt = None
        if token_attention_mask is not None:
            mask_dt = distribute_tensor(
                _pad_mask(token_attention_mask.to(a.dtype)).contiguous(), mesh, _ROW_PL
            )
        a_dt = self.token_transformer(a_dt, s_dt, z_dt, key_mask_dt=mask_dt)
        a = a_dt.full_tensor()
        if pad:
            a = a[:, :L, :]

        # Step 6: token norm (replicated)
        a = m.token_norm(a)

        # Step 7: atom decoder (replicated, full)
        r_update, dec_int = m.atom_decoder(
            a_i=a,
            q_l=q_skip,
            c_l=c_skip,
            p_lm=p_skip,
            atom_to_token=tok_idx,
            atom_attention_mask=ref_mask,
            num_diffusion_samples=num_diffusion_samples,
            return_intermediates=return_atom_repr,
        )

        # Step 8: denoised output (replicated)
        sigma2, t2 = sigma * sigma, t * t
        out = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy
        out = out + ((sigma * t) / torch.sqrt(sigma2 + t2))[:, None, None] * r_update

        atom_intermediates = None
        if return_atom_repr:
            all_ints = enc_int + dec_int
            if all_ints:
                atom_intermediates = torch.stack(all_ints, dim=2)
        return {
            "x_denoised": out,
            "token_repr": a if return_token_repr else None,
            "atom_intermediates": atom_intermediates,
        }


def wrap_model_with_cp_structure_head(model: nn.Module, dist_manager) -> list[str]:
    """Replace each ``DiffusionStructureHead``'s ``diffusion_module`` with the CP
    wrapper. The serial ``sample()`` loop keeps driving it. Returns replaced paths."""
    from esm.models.esmfold2.layers import DiffusionModule as SerialDiffusionModule

    replaced: list[str] = []
    for name, module in model.named_modules():
        dm_child = getattr(module, "diffusion_module", None)
        if isinstance(dm_child, SerialDiffusionModule):
            module.diffusion_module = DiffusionModuleCPWrapper(
                dm_child, dist_manager
            ).to(device=dist_manager.device, dtype=next(dm_child.parameters()).dtype)
            replaced.append(f"{name}.diffusion_module" if name else "diffusion_module")
    return replaced
