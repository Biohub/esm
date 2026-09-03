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

"""Distributed DiffusionConditioning for ESMFold2's structure head.

The conditioning produces, from the trunk pair ``z_trunk``, the per-head bias
source ``z`` (``B, L, L, c_z``) that biases the diffusion token attention — the
**dominant** diffusion tensor (``L²·c_z`` ≈ 4.6 GB at L=3000, far above the
per-layer attention scores). To get any diffusion memory win this ``z`` must be
sharded, so this module produces it as a 2-D-sharded DTensor
``(Shard(0), Shard(1), Shard(2))`` rather than the full tensor.

The ``z`` path (``z_input_norm`` → ``z_proj`` → ``z_transitions``) is entirely
pair-channel-wise (``TransitionLayer`` is LayerNorm + SwiGLU over the last dim),
so it runs **as-is on the local pair shard** using the serial submodule's
identically-replicated params. The ``s`` path (token single repr, cheap 1-D) runs
**replicated/full** exactly as serial.

Caller owns padding: ``z_trunk`` / ``rel_pos`` must already be padded so the token
axis is a multiple of the CP shard factor. Inference-only.
"""

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Shard

from esm.models.esmfold2.layers import (
    DiffusionConditioning as SerialDiffusionConditioning,
)

_PAIR_PL = [Shard(0), Shard(1), Shard(2)]


class DiffusionConditioningDistributed(nn.Module):
    """Distributed ``DiffusionConditioning``: full ``s``, 2-D-sharded ``z``.

    Thin shell over the serial module — reuses its submodules, running the ``z``
    path on the local pair shard and the ``s`` path on the full tensor. Bit-exact
    with serial (the ``z`` path is pointwise in the pair indices).
    """

    def __init__(self, layer: SerialDiffusionConditioning, device_mesh) -> None:
        super().__init__()
        if not isinstance(layer, SerialDiffusionConditioning):
            raise TypeError(
                f"layer must be DiffusionConditioning, got {type(layer).__name__}"
            )
        self.layer = layer
        self.device_mesh = device_mesh

    def forward(
        self,
        t_hat: torch.Tensor,
        s_inputs: torch.Tensor,
        z_trunk_dt: DTensor,
        rel_pos_dt: DTensor,
        sigma_data: float | None = None,
        num_diffusion_samples: int = 1,
        inference_cache: dict | None = None,
    ) -> tuple[torch.Tensor, DTensor]:
        """Mirrors ``DiffusionConditioning.forward`` (z cached across rollout).

        ``z_trunk_dt`` / ``rel_pos_dt`` are 2-D-sharded DTensors (already padded);
        returns ``(s_full, z_sharded_dt)``.
        """
        layer = self.layer
        mesh = self.device_mesh
        sigma = layer.sigma_data if sigma_data is None else float(sigma_data)
        # base (unexpanded) batch from s_inputs — z_trunk_dt is None on the cached
        # path, and s_inputs always carries the base batch.
        base_batch = s_inputs.shape[0]
        target_batch = base_batch * num_diffusion_samples

        # --- z path (cached), on the local pair shard ---
        if inference_cache is not None and "z_cp" in inference_cache:
            z_dt = inference_cache["z_cp"]
        else:
            zt_local = z_trunk_dt.to_local().to(torch.float32)
            zr_local = rel_pos_dt.to_local().to(torch.float32)
            z_local = torch.cat([zt_local, zr_local], dim=-1)
            z_local = layer.z_proj(layer.z_input_norm(z_local))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for block in layer.z_transitions:
                    z_local = z_local + block(z_local)
            z_dt = DTensor.from_local(
                z_local.contiguous(), device_mesh=mesh, placements=_PAIR_PL
            )
            if inference_cache is not None:
                inference_cache["z_cp"] = z_dt

        # --- s path (full / replicated), identical to serial ---
        s_inputs_eff = s_inputs
        if s_inputs_eff.shape[0] != target_batch:
            s_inputs_eff = s_inputs_eff.repeat_interleave(num_diffusion_samples, 0)
        s = layer.s_proj(layer.s_input_norm(s_inputs_eff.to(torch.float32)))

        t = torch.as_tensor(t_hat, dtype=torch.float32, device=s.device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(target_batch)
        elif t.shape[0] != target_batch:
            t = t.repeat_interleave(num_diffusion_samples, 0)
        t_noise = 0.25 * torch.log((t / sigma).clamp(min=1e-20))
        n = layer.fourier(t_noise)
        n = layer.noise_proj(layer.noise_norm(n))
        s = s + n.unsqueeze(1)
        for block in layer.s_transitions:
            s = s + block(s)

        return s, z_dt
