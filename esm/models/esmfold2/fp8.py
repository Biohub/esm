"""fp8 inference for the ESMC backbone, on PyTorch's native ``torch._scaled_mm``.

Weights are stored as fp8 (e4m3) with one fp32 scale per output row.
Activations are quantized per row on the fly. The fp8 GEMM needs compute
capability >= 8.9 (Ada or Hopper). Elsewhere the weight is upcast and the matmul
runs in the input dtype, so fp8 still halves weight memory.
"""

import functools
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from esm.models.esmc.layers import EsmcLayerNormLinear, EsmcLayerNormMLP

try:
    from esm.models.esmfold2.kernels.fp8_quantize import quantize_rows_fp8
except (ImportError, RuntimeError):
    quantize_rows_fp8: Any = None

logger = logging.getLogger(__name__)

_FP8 = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8).max


@functools.cache
def _has_fp8_gemm(device: torch.device) -> bool:
    """Whether ``device`` runs fp8 GEMMs. Warns once per device on a fallback."""
    fp8 = device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 9)
    if not fp8:
        logger.warning(
            "fp8 ESMC on %s: no fp8 GEMM here (needs compute capability >= 8.9). "
            "Weights stay fp8, but each matmul upcasts them and runs without the "
            "fp8 speedup.",
            device,
        )
    elif quantize_rows_fp8 is None:
        logger.warning(
            "fp8 ESMC on %s: Triton is unavailable, so activations are quantized "
            "with slower PyTorch ops.",
            device,
        )
    return fp8


def _quantize_rows(
    x: Tensor, *, swiglu: bool = False, norm: tuple[Tensor, Tensor, float] | None = None
) -> tuple[Tensor, Tensor]:
    """Quantize a 2D tensor to fp8 with one fp32 scale per row.

    ``swiglu`` and ``norm`` fold the preceding op in, as in :func:`quantize_rows_fp8`.
    """
    # Triton only emits e4m3 on compute capability >= 8.9.
    if quantize_rows_fp8 is not None and _has_fp8_gemm(x.device):
        return quantize_rows_fp8(x, swiglu=swiglu, norm=norm)
    return _quantize_rows_torch(x, swiglu=swiglu, norm=norm)


def _quantize_rows_torch(
    x: Tensor, *, swiglu: bool = False, norm: tuple[Tensor, Tensor, float] | None = None
) -> tuple[Tensor, Tensor]:
    """Reference for :func:`quantize_rows_fp8`, and the path without Triton."""
    x = x.float()
    if swiglu:
        gate, up = x.chunk(2, dim=-1)
        x = F.silu(gate) * up
    if norm is not None:
        weight, bias, eps = norm
        x = F.layer_norm(x, x.shape[-1:], weight.float(), bias.float(), eps)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / _FP8_MAX
    return (x / scale).to(_FP8), scale


class Fp8Linear(nn.Module):
    """A frozen, bias-free Linear whose weight lives in fp8.

    Weight and scale are stored as integer views so ``module.to(dtype)``
    cannot cast them.
    """

    weight_fp8: Tensor
    weight_scale: Tensor

    def __init__(self, weight_fp8: Tensor, weight_scale: Tensor) -> None:
        super().__init__()
        self.out_features, self.in_features = weight_fp8.shape
        self.register_buffer("weight_fp8", weight_fp8.view(torch.uint8))
        self.register_buffer("weight_scale", weight_scale.view(torch.int32))

    @classmethod
    def from_weight(cls, weight: Tensor) -> "Fp8Linear":
        return cls(*_quantize_rows(weight.detach()))

    def forward(self, x: Tensor) -> Tensor:
        if not _has_fp8_gemm(x.device):
            weight = self.weight_fp8.view(_FP8).float() * self.weight_scale.view(
                torch.float32
            )
            return F.linear(x, weight.to(x.dtype))
        return self.fp8_gemm(x)

    def fp8_gemm(
        self,
        x: Tensor,
        *,
        swiglu: bool = False,
        norm: tuple[Tensor, Tensor, float] | None = None,
    ) -> Tensor:
        """Quantize ``x`` (after the optional fused op) and run the fp8 GEMM."""
        lead = x.shape[:-1]
        xq, x_scale = _quantize_rows(
            x.reshape(-1, x.shape[-1]), swiglu=swiglu, norm=norm
        )
        out = torch._scaled_mm(
            xq,
            self.weight_fp8.view(_FP8).t(),
            scale_a=x_scale,
            scale_b=self.weight_scale.view(torch.float32).t(),
            out_dtype=torch.bfloat16,
        )
        return out.reshape(*lead, self.out_features)


class _LayerNormThen(nn.Module):
    """LayerNorm (reusing the source module's parameters) before fp8 GEMMs."""

    def __init__(self, layer_norm_weight: Tensor, layer_norm_bias: Tensor, eps: float):
        super().__init__()
        self.eps = eps
        self.layer_norm_weight = layer_norm_weight
        self.layer_norm_bias = layer_norm_bias

    def _norm(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x, x.shape[-1:], self.layer_norm_weight, self.layer_norm_bias, self.eps
        )

    def _norm_args(self) -> tuple[Tensor, Tensor, float]:
        return self.layer_norm_weight, self.layer_norm_bias, self.eps


class Fp8LayerNormLinear(_LayerNormThen):
    """fp8 counterpart of :class:`EsmcLayerNormLinear`."""

    def __init__(self, src: EsmcLayerNormLinear) -> None:
        super().__init__(src.layer_norm_weight, src.layer_norm_bias, src.eps)
        self.linear = Fp8Linear.from_weight(src.weight)

    def forward(self, x: Tensor) -> Tensor:
        if not _has_fp8_gemm(x.device):
            return self.linear(self._norm(x))
        return self.linear.fp8_gemm(x, norm=self._norm_args())


class Fp8LayerNormMLP(_LayerNormThen):
    """fp8 counterpart of :class:`EsmcLayerNormMLP`. ``fc1`` packs ``[gate; up]``."""

    def __init__(
        self,
        layer_norm_weight: Tensor,
        layer_norm_bias: Tensor,
        eps: float,
        fc1: Fp8Linear,
        fc2: Fp8Linear,
    ) -> None:
        super().__init__(layer_norm_weight, layer_norm_bias, eps)
        self.fc1 = fc1
        self.fc2 = fc2

    @classmethod
    def from_module(cls, src: EsmcLayerNormMLP) -> "Fp8LayerNormMLP":
        return cls(
            src.layer_norm_weight,
            src.layer_norm_bias,
            src.eps,
            Fp8Linear.from_weight(src.fc1_weight),
            Fp8Linear.from_weight(src.fc2_weight),
        )

    def forward(self, x: Tensor) -> Tensor:
        if not _has_fp8_gemm(x.device):
            x1, x2 = self.fc1(self._norm(x)).chunk(2, dim=-1)
            return self.fc2(F.silu(x1) * x2)
        hidden = self.fc1.fp8_gemm(x, norm=self._norm_args())
        return self.fc2.fp8_gemm(hidden, swiglu=True)


def _fp8_counterpart(module: nn.Module) -> nn.Module | None:
    if isinstance(module, EsmcLayerNormMLP):
        return Fp8LayerNormMLP.from_module(module)
    if isinstance(module, EsmcLayerNormLinear):
        return Fp8LayerNormLinear(module)
    if isinstance(module, nn.Linear) and module.bias is None:
        return Fp8Linear.from_weight(module.weight)
    return None


def _swap_children(parent: nn.Module) -> int:
    swapped = 0
    for name, child in list(parent.named_children()):
        new = _fp8_counterpart(child)
        if new is None:
            swapped += _swap_children(child)
        else:
            setattr(parent, name, new)
            swapped += 1
    return swapped


def convert_esmc_to_fp8_(esmc: nn.Module) -> int:
    """Swap every projection in ``esmc``'s transformer blocks for its fp8
    counterpart, in place. Returns the number of modules swapped.

    Each source weight is freed as soon as its fp8 copy exists. Works on any
    device, so a checkpoint can be converted on the CPU before it is moved.
    """
    with torch.no_grad():
        return _swap_children(esmc.get_submodule("transformer.blocks"))
