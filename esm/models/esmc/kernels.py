"""Optional accelerated kernels used by the ESMC layers.

ESMC runs on pure PyTorch by default, but selects fused kernels when the
matching package is installed and the model is being built for CUDA. These flags
report only what is importable; the target device decides what is usable.

The kernels, in the order ESMC prefers them:

* ``transformer_engine`` — fused LayerNorm+Linear / LayerNorm+MLP with an
  fp32 reduction inside the LayerNorm. Recommended on GPU for accurate bf16
  inference; without it the pure-PyTorch fallback drifts ~O(10) in fp32 and
  ~O(100) in bf16 on the unnormalized residual stream (perplexity stays
  within rounding noise).
* ``xformers`` — preferred fused attention kernel with a deterministic
  reduction order.
* ``flash-attn`` — secondary fused attention kernel (fp16 / bf16 only), plus
  the Triton RoPE kernel and the varlen packing helpers.
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

try:
    import transformer_engine.pytorch as te

    TE_INSTALLED = True
except ImportError:
    te: Any = None
    TE_INSTALLED = False

try:
    import xformers.ops as xops

    XFORMERS_INSTALLED = True
except ImportError:
    xops: Any = None
    XFORMERS_INSTALLED = False

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input

    FLASH_ATTN_INSTALLED = True
except ImportError:
    flash_attn_func: Any = None
    flash_attn_varlen_qkvpacked_func: Any = None
    pad_input: Any = None
    unpad_input: Any = None
    FLASH_ATTN_INSTALLED = False

try:
    from flash_attn.ops.triton.rotary import apply_rotary as apply_triton_rotary

    FLASH_ATTN_ROTARY_INSTALLED = True
except ImportError:
    apply_triton_rotary: Any = None
    FLASH_ATTN_ROTARY_INSTALLED = False


if not TE_INSTALLED:
    logger.warning(
        "ESMC: Transformer Engine is not installed; falling back to "
        "pure-PyTorch LayerNorm+Linear / LayerNorm+MLP. Outputs will differ "
        "numerically - measured on the unnormalized residual stream (before "
        "the final LayerNorm), ~O(10) max-diff in fp32 and ~O(100) in bf16; "
        "after the final LayerNorm these shrink to a few ULP and perplexity "
        "stays within rounding noise. Install with "
        "`pip install transformer-engine[pytorch]` to enable fused fp32-"
        "reduction LayerNorm."
    )

if not XFORMERS_INSTALLED and not FLASH_ATTN_INSTALLED:
    logger.warning(
        "ESMC: neither xformers nor flash-attn is installed; falling back "
        "to PyTorch ``F.scaled_dot_product_attention``. The attention "
        "reduction order in bf16 differs from a fused kernel by ~1 bf16 "
        "ULP per attention block; compounded across the 80-block stack "
        "this reaches ~O(100) max-diff on the unnormalized residual stream. "
        "Install an xformers build matching your PyTorch and CUDA versions "
        "for a fused attention kernel."
    )

if torch.cuda.is_available() and not FLASH_ATTN_ROTARY_INSTALLED:
    logger.warning(
        "ESMC: flash-attn rotary kernel not installed; falling back to "
        "pure-PyTorch RoPE. For faster GPU inference run `pip install flash-attn`."
    )
