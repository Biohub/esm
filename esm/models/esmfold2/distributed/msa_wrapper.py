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

"""End-to-end CP runtime for ESMFold2's MSAEncoder.

``MSAEncoderCPWrapper`` is a drop-in replacement for the serial ``MSAEncoder``
(same plain-tensor in/out signature) that shards the MSA encoder's pair-space
work across the 2D CP grid, mirroring ``TrunkCPWrapper`` for the folding trunk.
Without it, the serial MSA encoder materialises the full L×L pair on every rank
(the dominant cost for large complexes) before the trunk even runs.
"""

from math import lcm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor


class MSAEncoderCPWrapper(nn.Module):
    """Drop-in replacement for ``MSAEncoder`` that runs distributed.

    The small per-residue embedding (``embed`` + ``project_inputs``) runs serial
    per rank (cheap), then m / pair / masks are zero-padded to a multiple of the
    CP shard factor, distributed as DTensors, processed by
    ``MSAEncoderDistributed``, and the gathered pair is sliced back.
    """

    def __init__(
        self,
        serial_encoder: nn.Module,
        dist_manager,
        comm: str = "gather",
        bf16: bool = True,
    ) -> None:
        super().__init__()
        from esm.models.esmfold2.distributed.manager import DistributedManager
        from esm.models.esmfold2.distributed.model.layers.msa_encoder import (
            MSAEncoderDistributed,
        )
        from esm.models.esmfold2.model import MSAEncoder as SerialMSAEncoder

        if not isinstance(serial_encoder, SerialMSAEncoder):
            raise TypeError(f"expected MSAEncoder, got {type(serial_encoder).__name__}")
        if not isinstance(dist_manager, DistributedManager):
            raise TypeError(
                f"expected DistributedManager, got {type(dist_manager).__name__}"
            )

        # The distributed path ignores the serial chunk-size knob.
        serial_encoder.set_chunk_size(None)

        # The embedding projections stay serial (tiny); reference them directly.
        self.embed = serial_encoder.embed
        self.project_inputs = serial_encoder.project_inputs
        self.dist_encoder = MSAEncoderDistributed(
            serial_encoder, dist_manager, comm=comm
        )

        # bf16 MSA encoder: the recycle pair is fp32 and the MSA encoder otherwise
        # runs fp32 (its distributed tri-mul's custom_fwd disables autocast), which
        # made it the largest recycle-loop peak contributor (~16 GB @ L1422). Cast
        # the heavy dist_encoder to bf16 (mirrors TrunkCPWrapper's bf16=True, which
        # was validated quality-neutral); the pair is cast to bf16 at the boundary
        # in forward[_sharded] and the output cast back to the caller's dtype. The
        # tiny embed/project_inputs are left fp32 (shared with the serial model;
        # m is cast to bf16 after embedding).
        self._bf16 = bf16
        if self._bf16:
            self.dist_encoder = self.dist_encoder.to(torch.bfloat16)

        self.dist_manager = dist_manager
        self.device_mesh = dist_manager.device_mesh_subgroups
        # device_mesh is (dp, cp_axis_0, cp_axis_1)
        self.cp_axis_0 = self.device_mesh.size(1)
        self.cp_axis_1 = self.device_mesh.size(2)
        self.shard_factor = lcm(self.cp_axis_0, self.cp_axis_1)

    def set_chunk_size(self, _chunk_size: int | None) -> None:
        return

    def set_kernel_backend(self, _backend: str | None) -> None:
        return

    def forward_sharded(
        self,
        x_pair_dt: DTensor,
        x_inputs: torch.Tensor,
        msa_oh: torch.Tensor,
        has_deletion: torch.Tensor,
        deletion_value: torch.Tensor,
        msa_attention_mask: torch.Tensor,
    ) -> DTensor:
        """Run the distributed MSA encoder on an already-sharded pair DTensor.

        ``x_pair_dt`` must already be padded to a multiple of
        ``self.shard_factor`` and distributed ``[Shard(0), Shard(1), Shard(2)]``
        on ``self.device_mesh``. The MSA inputs (``m`` and the masks) are still
        embedded serially per rank (cheap), padded to match the pair's padded
        length, and distributed here. Returns the pair-space output as a DTensor
        — **no** ``full_tensor()`` gather.

        This is the entry point a CP orchestrator uses to keep the pair sharded
        across the MSA→trunk boundary; the plain-tensor :meth:`forward` is a thin
        adapter around it.
        """
        # Serial embedding (matches MSAEncoder.forward), per rank on full tensors.
        m_feat = torch.cat(
            [msa_oh, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1
        )
        m = self.embed(m_feat) + self.project_inputs(x_inputs).unsqueeze(2)

        # The incoming pair is already padded; pad m / masks to match it so the
        # token (L) axis lines up shard-for-shard.
        n = m.shape[1]
        n_padded = x_pair_dt.shape[1]
        pad = n_padded - n
        if pad < 0:
            raise ValueError(f"sharded pair length {n_padded} < MSA token length {n}")
        if pad:
            m = F.pad(m, (0, 0, 0, 0, 0, pad))  # pad L (dim 1)
            msa_attention_mask = F.pad(msa_attention_mask, (0, 0, 0, pad))  # pad L

        # Cast the heavy inputs to bf16 so the (bf16) dist_encoder runs bf16 end to
        # end; restore the caller's dtype on the output.
        orig_dtype = x_pair_dt.dtype
        if self._bf16:
            m = m.to(torch.bfloat16)
            x_pair_dt = x_pair_dt.to(torch.bfloat16)  # ty:ignore[invalid-assignment]
            msa_attention_mask = msa_attention_mask.to(torch.bfloat16)

        mesh = self.device_mesh
        m_dt = distribute_tensor(
            m.contiguous(), mesh, [Shard(0), Shard(1), Replicate()]
        )
        msa_mask_dt = distribute_tensor(
            msa_attention_mask.to(m.dtype).contiguous(),
            mesh,
            [Shard(0), Shard(1), Replicate()],
        )
        # Pair attention mask built SHARDED from the (padded) token mask — the
        # per-block outer product, never a full [B, L, L]. Padded rows/cols are
        # zero, so they contribute nothing.
        from esm.models.esmfold2.distributed.model.layers.pair_init import (
            build_sharded_pair_mask,
        )

        pair_mask_dt = build_sharded_pair_mask(msa_attention_mask[:, :, 0], mesh).to(
            m.dtype
        )

        out_dt = self.dist_encoder(m_dt, x_pair_dt, msa_mask_dt, pair_mask_dt)
        if self._bf16:
            out_dt = out_dt.to(orig_dtype)
        return out_dt

    def forward(
        self,
        x_pair: torch.Tensor,
        x_inputs: torch.Tensor,
        msa_oh: torch.Tensor,
        has_deletion: torch.Tensor,
        deletion_value: torch.Tensor,
        msa_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        n = x_pair.shape[1]
        pad = (self.shard_factor - n % self.shard_factor) % self.shard_factor
        if pad:
            x_pair = F.pad(x_pair, (0, 0, 0, pad, 0, pad))  # pad both L dims

        pair_dt = distribute_tensor(
            x_pair.contiguous(), self.device_mesh, [Shard(0), Shard(1), Shard(2)]
        )

        out_dt = self.forward_sharded(
            pair_dt, x_inputs, msa_oh, has_deletion, deletion_value, msa_attention_mask
        )
        out = out_dt.full_tensor()
        if pad:
            out = out[:, :n, :n, :]
        return out


def wrap_model_with_cp_msa_encoder(
    model: nn.Module, dist_manager, comm: str = "gather", bf16: bool = True
) -> list[str]:
    """Replace every ``MSAEncoder`` submodule with ``MSAEncoderCPWrapper``.

    Walks ``model.named_modules()`` and rebinds each attribute that points at a
    serial ``MSAEncoder``. Returns the list of replaced submodule paths. Models
    without an MSA encoder (``model.msa_encoder is None``) are left untouched.

    ``comm`` selects the MSA pair-weighted-averaging communication strategy:
    ``"gather"`` (default, bit-exact all-gather) or ``"ring"`` (boltz-style
    online-softmax ring, for large grids / deep MSAs).
    """
    mesh = dist_manager.device_mesh_subgroups
    cp0, cp1 = mesh.size(1), mesh.size(2)
    if cp0 != cp1:
        raise ValueError(
            f"CP grid must be square (cp_axis_0 == cp_axis_1), got {cp0}×{cp1}"
        )

    from esm.models.esmfold2.model import MSAEncoder as SerialMSAEncoder

    targets: list[tuple[str, nn.Module, str, nn.Module]] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if child is None:
                continue
            if isinstance(child, SerialMSAEncoder):
                full = f"{parent_name}.{child_name}" if parent_name else child_name
                targets.append((full, parent, child_name, child))

    replaced: list[str] = []
    for full, parent, child_name, child in targets:
        wrapped = MSAEncoderCPWrapper(child, dist_manager, comm=comm, bf16=bf16).to(
            device=dist_manager.device, dtype=next(child.parameters()).dtype
        )
        setattr(parent, child_name, wrapped)
        replaced.append(full)
    return replaced


def wrap_model_with_cp(
    model: nn.Module,
    dist_manager,
    comm: str = "gather",
    bf16: bool = True,
    offload_esmc: bool = True,
    wrap_structure: bool = True,
    tp_esmc: bool = False,
) -> list[str]:
    """Wrap both the folding trunk(s) and the MSA encoder for 2D CP.

    Convenience over calling ``wrap_model_with_cp_trunks`` and
    ``wrap_model_with_cp_msa_encoder`` separately. Returns the combined list of
    replaced submodule paths.

    ``comm`` selects the MSA pair-weighted-averaging communication strategy:
    ``"gather"`` (default, bit-exact all-gather) or ``"ring"`` (boltz-style
    online-softmax ring). It affects only the MSA encoder; the folding trunk
    and the OuterProductMean (whose contracted MSA-depth axis is replicated)
    are unaffected.

    ``bf16`` (default True): run the distributed trunk in bf16 — quality-neutral,
    ~1.3-1.6x faster, lower peak VRAM (see ``wrap_model_with_cp_trunks``).

    ``offload_esmc`` (default True): offload the ESM-C LM (~12 GB) to CPU after
    its one-shot use, freeing it for the trunk/diffusion. Sets
    ``model._offload_esmc``. Validated with bf16: 2.14x end-to-end + 18.5 GB lower
    peak vs fp32, pLDDT unchanged. Set False for high-throughput tiny-fold or
    concurrent-fold use (the per-fold ~12 GB CPU<->GPU transfer isn't worth it).
    """
    from esm.models.esmfold2.distributed.utils import wrap_model_with_cp_trunks

    model._offload_esmc = offload_esmc  # ty:ignore[unresolved-attribute]
    replaced = wrap_model_with_cp_trunks(model, dist_manager, bf16=bf16)
    replaced += wrap_model_with_cp_msa_encoder(
        model, dist_manager, comm=comm, bf16=bf16
    )

    # Install the CP recycle orchestrator so the pair stays sharded across the
    # recycle loop (no per-iteration full_tensor() round-trips). The model's
    # _run_one_loop delegates to this when present (CP-agnostic seam). Only the
    # main model owns the loop (parcae_input_norm + _run_one_loop); guard so
    # wrapping a bare submodule is a no-op.
    if hasattr(model, "_run_one_loop") and hasattr(model, "parcae_input_norm"):
        from esm.models.esmfold2.distributed.recycle import CPRecycleEngine

        model._cp_recycle_engine = CPRecycleEngine(model, dist_manager)  # ty:ignore[unresolved-attribute]

        # Sharded pair-init builder: constructs z_init / rel_pos / token_bonds as
        # 2-D-sharded DTensors so the full L×L pair is never resident on a rank
        # during the init phase (the dominant per-rank peak that made CP OOM at
        # short L). The model's forward delegates via the _cp_pair_init seam.
        from esm.models.esmfold2.distributed.model.layers.pair_init import (
            PairInitDistributed,
        )

        model._cp_pair_init = PairInitDistributed(model, dist_manager)

        # Install the distributed LM->pair builder (#14) so lm_z is produced as a
        # sharded DTensor instead of a full L×L tensor on every rank (the binding
        # per-rank peak per the T0 phase sweep). The model's forward delegates via
        # the _cp_language_model seam when present.
        lm = getattr(model, "language_model", None)
        if lm is not None:
            from esm.models.esmfold2.distributed.model.layers.single_to_pair import (
                LanguageModelShimDistributed,
            )

            model._cp_language_model = LanguageModelShimDistributed(lm, dist_manager)

        # Distributed confidence head (#6): consumes the sharded pair from the CP
        # tail (no full-L×L z re-gather). Its nested FoldingTrunk is already
        # CP-wrapped by wrap_model_with_cp_trunks above.
        conf = getattr(model, "confidence_head", None)
        if conf is not None:
            from esm.models.esmfold2.distributed.confidence_wrapper import (
                ConfidenceHeadCPWrapper,
            )

            model._cp_confidence_head = ConfidenceHeadCPWrapper(conf, dist_manager)

        # Distributed distogram head: distogram_head(z + zᵀ) off the sharded pair
        # (one transpose + local Linear), gathering only the bins-channel logits.
        disto = getattr(model, "distogram_head", None)
        if disto is not None:
            from esm.models.esmfold2.distributed.distogram_wrapper import (
                DistogramHeadCPWrapper,
            )

            model._cp_distogram_head = DistogramHeadCPWrapper(disto, dist_manager)

    # Distribute the diffusion structure head (conditioned-z + token attention
    # sharded; atom encoder/decoder replicated). The serial sample() loop keeps
    # driving the (now wrapped) diffusion_module.
    if wrap_structure:
        from esm.models.esmfold2.distributed.structure_wrapper import (
            wrap_model_with_cp_structure_head,
        )

        replaced += wrap_model_with_cp_structure_head(model, dist_manager)

    # Tensor-parallel ESM-C (the dominant replicated floor). Phase 1: shard the
    # SwiGLU MLP (~67% of ESM-C weights) over the CP ranks via TE-native TP.
    if tp_esmc:
        from esm.models.esmfold2.distributed.esmc_tp import tp_shard_esmc_mlp

        n_tp = tp_shard_esmc_mlp(model, dist_manager)
        if n_tp:
            replaced.append(f"esmc.transformer.blocks[*].ffn (TP x{n_tp})")

    return replaced
