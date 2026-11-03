"""Optional accelerated kernels used by the ESMC layers.

ESMC runs on pure PyTorch by default, but selects fused kernels when the
matching package is installed and the model is being built for CUDA. These flags
report only what is importable; the target device decides what is usable.

The kernels, in the order ESMC prefers them:

* ``flash-attn`` — fused attention kernel (fp16 / bf16 only), plus the
  Triton RoPE kernel and the varlen packing helpers.
"""

import warnings
from typing import Any

import torch

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
    # ``flash_attn.layers.rotary.apply_rotary_emb`` wraps the same Triton kernel in an
    # ``autograd.Function``; the bare ``flash_attn.ops.triton.rotary.apply_rotary`` driver
    # returns a tensor with no ``grad_fn``, so a backward through it silently skips RoPE.
    from flash_attn.layers.rotary import apply_rotary_emb as apply_triton_rotary

    FLASH_ATTN_ROTARY_INSTALLED = True
except ImportError:
    apply_triton_rotary: Any = None
    FLASH_ATTN_ROTARY_INSTALLED = False


FLASH_ATTN_WHEEL_URL = (
    "https://github.com/evolutionaryscale/wheels/releases/download/"
    "py312-pt211-cu13-sm80-90/flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl"
)

if torch.cuda.is_available() and not (
    FLASH_ATTN_INSTALLED and FLASH_ATTN_ROTARY_INSTALLED
):
    # warnings, not logging, and forced: the esmfold2 Triton modules ignore
    # every UserWarning process-wide.
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(
            "ESMC: flash-attn is not installed or failed to import; attention and "
            "RoPE fall back to pure PyTorch. PyPI cannot carry our wheel, so install "
            f'it by hand:\n\n    pip install "flash_attn @ {FLASH_ATTN_WHEEL_URL}"\n\n'
            "It needs Linux x86-64, Python 3.12, torch 2.11, CUDA 13, compute "
            "capability 8.0-9.0 and libstdc++ from GCC 13+ (Ubuntu 24.04, or "
            "LD_PRELOAD a newer libstdc++ on 22.04)."
        )
