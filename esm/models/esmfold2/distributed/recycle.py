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

"""Context-parallel orchestrator for ESMFold2's recycle loop.

``CPRecycleEngine`` owns the sharded version of ``EsmFold2Model._run_one_loop``.
The model file stays CP-agnostic: ``_run_one_loop`` simply delegates to an
installed engine (see the ``_cp_recycle_engine`` seam there), which is set by
``wrap_model_with_cp``.

Why an orchestrator: the serial loop materialises the full ``L×L`` pair on every
rank at *each* module boundary (MSA encoder, LM encoder, trunk) via the wrappers'
``forward()`` (``distribute_tensor`` → … → ``full_tensor``). Keeping the running
pair sharded across the whole loop — calling each wrapper's ``forward_sharded``
and running the parcae combine on local shards — removes those per-iteration
gathers. The pair is gathered exactly once, when the loop returns (until the
diffusion / confidence heads are also distributed, at which point even that
final gather can go).

All per-iteration tensors stay DTensors with placements
``(Shard(0), Shard(1), Shard(2))``; the channel-dim parcae combine is computed on
``to_local()`` shards (channel is replicated, so it needs no communication) and
re-wrapped — mirroring ``LinearParamsReplicated.forward``. Validated bit-exact
vs. the serial path on a 2×2 grid.
"""

from math import lcm

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from esm.models.esmfold2.distributed.model.layers.layernorm import (
    LayerNormParamsReplicated,
)
from esm.models.esmfold2.layers import NUM_RES_TYPES, maybe_subsample_msa

_PLACEMENTS = [Shard(0), Shard(1), Shard(2)]


class CPRecycleEngine:
    """Sharded drop-in for ``EsmFold2Model._run_one_loop``.

    Parameters
    ----------
    model:
        The ESMFold2 model being wrapped. Used to read ``parcae_input_norm`` (to
        build a replicated-param copy) at construction; the (already CP-wrapped)
        ``folding_trunk`` / ``msa_encoder`` / ``lm_encoder`` and config are read
        per call so the engine always sees the current submodules.
    dist_manager:
        The DistributedManager providing the CP device mesh.
    """

    def __init__(self, model, dist_manager) -> None:
        self.dist_manager = dist_manager
        self.device_mesh = dist_manager.device_mesh_subgroups
        # device_mesh is (dp, cp_axis_0, cp_axis_1)
        self.cp_axis_0 = self.device_mesh.size(1)
        self.cp_axis_1 = self.device_mesh.size(2)
        self.shard_factor = lcm(self.cp_axis_0, self.cp_axis_1)

        # Replicated-param copy of the channel-dim LayerNorm: a plain nn.LayerNorm
        # cannot run on a DTensor (mixed plain-tensor/DTensor params raise), so the
        # parcae input norm is wrapped to hold replicated DTensor params.
        self.parcae_input_norm = LayerNormParamsReplicated(
            model.parcae_input_norm, self.device_mesh
        )

    def parcae_finish(self, model, z_dt: DTensor, pair_mask: torch.Tensor) -> DTensor:
        """Sharded post-recycle parcae tail, keeping the pair sharded.

        Mirrors the serial ``z = parcae_readout(z); z = parcae_coda(z, mask)``:
        ``parcae_readout`` is a channel-wise Linear run on the local shard
        (weight replicated, no token mixing); ``parcae_coda`` is a CP-wrapped
        ``FoldingTrunk`` driven via ``forward_sharded`` (no gather/re-distribute).
        ``z_dt`` is the 2-D-sharded (padded) pair from ``run_loop(gather=False)``;
        ``pair_mask`` is either the full (unpadded) token pair mask or an already
        padded + sharded DTensor (from ``PairInitDistributed``). Returns sharded z."""
        mesh = self.device_mesh

        # parcae_readout: channel Linear on the local shard.
        z_local = z_dt.to_local()
        z_local = F.linear(z_local, model.parcae_readout.weight)
        z_dt = DTensor.from_local(z_local.contiguous(), mesh, _PLACEMENTS)

        # parcae_coda: CP-wrapped trunk → forward_sharded (needs a padded, sharded mask).
        if isinstance(pair_mask, DTensor):
            pair_mask_dt = pair_mask.to(z_dt.dtype)
        else:
            n = pair_mask.shape[-1]
            pad = (self.shard_factor - n % self.shard_factor) % self.shard_factor
            pm = F.pad(pair_mask, (0, pad, 0, pad)) if pad else pair_mask
            pair_mask_dt = distribute_tensor(
                pm.to(z_dt.dtype).contiguous(), mesh, _PLACEMENTS
            )
        return model.parcae_coda.forward_sharded(z_dt, pair_attention_mask=pair_mask_dt)

    def _prepare_msa_iter(self, model, msa_inputs, tok_mask):
        """Per-iteration MSA feature prep — a faithful copy of the serial block
        in ``EsmFold2Model._run_one_loop`` (kept here so the engine doesn't
        mutate the serial inference path). Returns plain tensors; sharding
        happens inside ``MSAEncoderCPWrapper.forward_sharded``."""
        msa_i, mask_i, hd_i, dv_i = maybe_subsample_msa(
            msa_inputs["msa"],
            msa_inputs["msa_attention_mask"],
            msa_inputs["has_deletion"],
            msa_inputs["deletion_value"],
            max_depth=msa_inputs["max_depth"],
            enabled=msa_inputs["subsample_enabled"],
        )
        B_msa, M, L_msa = msa_i.shape
        msa_oh = F.one_hot(
            msa_i.permute(0, 2, 1).long(), num_classes=NUM_RES_TYPES
        ).float()
        msa_attn = (
            mask_i.permute(0, 2, 1).float()
            if mask_i is not None
            else tok_mask[:, :, None].expand(-1, -1, M).float()
        )
        # Bias-free MSAEncoder.embed requires zeroed padding.
        msa_oh = msa_oh * msa_attn.unsqueeze(-1)
        hd = (
            hd_i.permute(0, 2, 1).float()
            if hd_i is not None
            else torch.zeros(B_msa, L_msa, M, device=msa_i.device)
        )
        dv = (
            dv_i.permute(0, 2, 1).float()
            if dv_i is not None
            else torch.zeros(B_msa, L_msa, M, device=msa_i.device)
        )
        return msa_oh, msa_attn, hd, dv, msa_inputs["x_inputs"]

    def run_loop(
        self,
        model,
        *,
        z: torch.Tensor,
        z_init: torch.Tensor,
        lm_z: torch.Tensor | None,
        msa_inputs: dict | None,
        pair_mask: torch.Tensor,
        a: torch.Tensor,
        b_mat: torch.Tensor,
        tok_mask: torch.Tensor,
        total_steps: int,
        gather: bool = True,
        n_orig: int | None = None,
    ):
        """Sharded equivalent of ``_run_one_loop``.

        ``z`` / ``z_init`` may arrive either as full plain tensors (distributed
        here, the serial-fallback contract) or as already-padded, already-sharded
        ``(Shard0, Shard1, Shard2)`` DTensors built by ``PairInitDistributed`` — in
        which case the full L×L pair is never resident on any rank and ``n_orig``
        (the pre-pad token count) must be supplied so the returned length is the
        original, not the padded, count.

        With ``gather=True`` (default) returns the gathered full tensor (drop-in
        for the serial loop). With ``gather=False`` returns
        ``(z_dt, n_orig)`` — the **sharded** ``(Shard0, Shard1, Shard2)`` pair
        DTensor (still padded to the shard factor) and the original token length
        ``n_orig`` — so a CP tail can keep the pair sharded through
        parcae / distogram / structure / confidence instead of re-gathering. The
        padded rows/cols carry no signal (masked); the caller slices to
        ``n_orig`` after any later gather."""
        lm_cfg = model.config.lm_encoder
        lm_dropout_p = getattr(lm_cfg, "lm_dropout", 0.0)
        # Per-loop LM dropout: the serial path applies a fresh F.dropout over the
        # *full* lm_z each iteration (training=True even under eval). We reproduce
        # it the same way — dropout on the full (unpadded) lm_z, then pad +
        # distribute — so with the run's synchronized RNG (all ranks share the
        # seed and draw in the same order) every rank gets the identical mask, and
        # it matches the serial / gather-fallback path. Drawing the mask on a
        # per-rank shard instead would desync the ranks.
        per_loop_lm_dropout = (
            lm_z is not None
            and getattr(lm_cfg, "per_loop_lm_dropout", False)
            and lm_dropout_p > 0.0
        )

        mesh = self.device_mesh
        loop_dtype = z.dtype
        # z / z_init may be pre-sharded DTensors (PairInitDistributed) or full
        # tensors (serial fallback). For the DTensor case N is the ORIGINAL token
        # count (passed as n_orig); z.shape[1] would be the padded length.
        z_is_dt = isinstance(z, DTensor)
        z_init_is_dt = isinstance(z_init, DTensor)
        if z_is_dt:
            assert n_orig is not None, "n_orig required when z is a pre-sharded DTensor"
            N = n_orig
        else:
            N = z.shape[1]
        pad = (self.shard_factor - N % self.shard_factor) % self.shard_factor

        def _pad_pair(t: torch.Tensor) -> torch.Tensor:  # (B, L, L, C)
            return F.pad(t, (0, 0, 0, pad, 0, pad)) if pad else t

        def _pad_mask(t: torch.Tensor) -> torch.Tensor:  # (B, L, L)
            return F.pad(t, (0, pad, 0, pad)) if pad else t

        def _distribute_lm(lm_full: torch.Tensor) -> DTensor:
            return distribute_tensor(
                _pad_pair(lm_full.to(z_init.dtype)).contiguous(), mesh, _PLACEMENTS
            )

        # Distribute the persistent loop tensors ONCE (vs. per-boundary in serial).
        # Pre-sharded DTensors (already padded to shard_factor) are used as-is —
        # no full L×L is ever materialised on a rank.
        z_dt = (
            z
            if z_is_dt
            else distribute_tensor(_pad_pair(z).contiguous(), mesh, _PLACEMENTS)
        )
        z_init_dt = (
            z_init
            if z_init_is_dt
            else distribute_tensor(_pad_pair(z_init).contiguous(), mesh, _PLACEMENTS)
        )
        # pair_mask may arrive as a pre-sharded (padded) DTensor (PairInitDistributed)
        # or a full tensor (serial fallback).
        if isinstance(pair_mask, DTensor):
            pair_mask_dt = pair_mask.to(loop_dtype)
        else:
            pair_mask_dt = distribute_tensor(
                _pad_mask(pair_mask).to(loop_dtype).contiguous(), mesh, _PLACEMENTS
            )

        # lm_z may arrive as a sharded (Shard0,Shard1,Shard2) DTensor (#14 —
        # produced full-L×L-free by LanguageModelShimDistributed) or as a full
        # tensor (serial-produced, e.g. the unit tests). Either way the loop needs a
        # per-iteration sharded DTensor.
        lm_is_dt = isinstance(lm_z, DTensor)
        # Per-rank generator for SHARD-LOCAL dropout (sharded lm_z only): each rank
        # drops its own tile with a distinct, fresh-per-loop mask. This is NOT
        # bit-exact with the serial full-L×L dropout mask (torch's sequential philox
        # can't be cheaply tiled) — a valid-but-different stochastic sample, like the
        # ring MSA path; validated by end-to-end pLDDT/pTM parity. Crucially it uses
        # a *separate* generator, leaving the global RNG untouched so the MSA
        # subsample below stays synchronized ACROSS ranks (all ranks pick the same
        # subset — required for the distributed MSA encoder to be consistent).
        lm_local_static: torch.Tensor | None = None
        lm_shape = lm_stride = None
        lm_drop_gen: torch.Generator | None = None
        lm_z_dt_static: DTensor | None = None
        if lm_is_dt:
            lm_cast = lm_z.to(z_init.dtype)
            if per_loop_lm_dropout:
                lm_local_static = lm_cast.to_local()  # ty:ignore[unresolved-attribute]
                lm_shape, lm_stride = lm_cast.shape, lm_cast.stride()
                lm_drop_gen = torch.Generator(device=lm_local_static.device)
                lm_drop_gen.manual_seed(0x5F14 + torch.distributed.get_rank())
            else:
                lm_z_dt_static = lm_cast  # ty:ignore[invalid-assignment]
        elif lm_z is not None and not per_loop_lm_dropout:
            lm_z_dt_static = _distribute_lm(lm_z)

        overwrite = model.config.msa_encoder.overwrite

        for _ in range(total_steps):
            # (1) LM dropout FIRST, matching the serial loop's RNG-draw order
            #     (dropout before MSA subsample) so the synchronized RNG stays in
            #     lockstep with the serial / gather-fallback path.
            if lm_z is None:
                lm_z_i = None
            elif not per_loop_lm_dropout:
                lm_z_i = lm_z_dt_static
            elif lm_is_dt:
                # Shard-local dropout on the local tile (see generator note above):
                # keep with prob (1-p), scale by 1/(1-p) — matches F.dropout.
                keep = 1.0 - lm_dropout_p
                mask = (
                    torch.rand(
                        lm_local_static.shape,  # ty:ignore[unresolved-attribute]
                        generator=lm_drop_gen,
                        device=lm_local_static.device,  # ty:ignore[unresolved-attribute]
                        dtype=lm_local_static.dtype,  # ty:ignore[unresolved-attribute]
                    )
                    >= lm_dropout_p
                )
                drop_local = lm_local_static * mask / keep  # ty:ignore[unsupported-operator]
                lm_z_i = DTensor.from_local(
                    drop_local.contiguous(),
                    device_mesh=mesh,
                    placements=_PLACEMENTS,
                    shape=lm_shape,
                    stride=lm_stride,
                )
            else:
                lm_i_full = F.dropout(lm_z, p=lm_dropout_p, training=True)
                lm_z_i = _distribute_lm(lm_i_full)

            refined_lm_z: DTensor | None = None
            if lm_z_i is not None and model.lm_encoder is not None:
                refined_lm_z = model.lm_encoder.forward_sharded(
                    lm_z_i, pair_attention_mask=pair_mask_dt
                )

            z_inject = z_init_dt
            if lm_z_i is not None and model.lm_encoder is None:
                z_inject = z_inject + lm_z_i.to(z_inject.dtype)

            # (2) MSA prep (RNG draw) AFTER LM dropout, matching serial order.
            if model.msa_encoder is not None and msa_inputs is not None:
                msa_oh, msa_attn, hd, dv, x_inputs = self._prepare_msa_iter(
                    model, msa_inputs, tok_mask
                )
                msa_pair = model.msa_encoder.forward_sharded(
                    z_inject, x_inputs, msa_oh, hd, dv, msa_attn
                ).to(z_inject.dtype)
                z_inject = msa_pair if overwrite else (z_inject + msa_pair)

            if refined_lm_z is not None:
                z_inject = z_inject + refined_lm_z.to(z_inject.dtype)

            # parcae combine (channel-dim): norm via replicated params, then the
            # affine recurrence on local shards (channel is replicated, so a*z and
            # F.linear are purely local — raw F.linear on a DTensor would try to
            # flatten the sharded token axis and fail).
            inj_dt = self.parcae_input_norm(z_inject)
            z_local = z_dt.to_local()
            out_local = a * z_local + F.linear(
                inj_dt.to_local().to(z_local.dtype), b_mat
            )
            z_dt = DTensor.from_local(out_local, mesh, _PLACEMENTS)

            z_dt = model.folding_trunk.forward_sharded(
                z_dt, pair_attention_mask=pair_mask_dt
            )

        if not gather:
            # Keep the pair sharded for an end-to-end sharded tail; padded to the
            # shard factor. Caller slices to N after any later gather.
            return z_dt, N

        z_full = z_dt.full_tensor()
        if pad:
            z_full = z_full[:, :N, :N, :]
        return z_full
