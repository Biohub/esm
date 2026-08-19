# coding=utf-8
# Copyright 2026 Biohub. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""ESMFold2 has to import and stay usable with no CUDA device.

``@triton.autotune`` resolves ``driver.active`` at *decoration* time, so merely
importing the vendored kernels on a machine with no GPU driver raises
RuntimeError. ``layers.py`` guarded that import with ``except ImportError``,
which does not catch it, and the whole ``esmfold2`` package became unimportable
on CPU - every test module that touched it failed at collection. Nothing caught
that, because every machine the suite ran on had a driver.

The check runs in a subprocess with ``CUDA_VISIBLE_DEVICES=""``. That is the same
branch a driverless machine takes, and it cannot be done in-process: which
kernels are loaded is decided once, at import.
"""

import os
import subprocess
import sys

#: Asserts the fallback chain's bottom rung: the package imports, the optional
#: accelerators are all off, and the PyTorch path is what is left.
_PROBE = """
import torch

assert not torch.cuda.is_available(), "CUDA_VISIBLE_DEVICES='' did not hide the GPU"

from esm.models.esmfold2 import layers

assert layers.TRITON_KERNELS_AVAILABLE is False, "Triton claimed to be usable"
assert layers._fused_pair_bias is None
assert layers._fused_trimul_with_residual is None
assert layers._FusedLNLinearSwiGLU is None
assert layers._FusedDropoutResidual is None

# The public surface has to come with it, not just the leaf module.
from esm.models.esmfold2 import EsmFold2Config, EsmFold2Model

assert EsmFold2Config is not None and EsmFold2Model is not None
print("CPU_ONLY_OK")
"""


def test_esmfold2_imports_and_disables_triton_without_a_cuda_device():
    """No device -> no Triton, and the import still succeeds."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        # An empty device list is what a driverless box looks like to torch.
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=600,
    )
    assert result.returncode == 0, (
        f"esmfold2 is not importable without a CUDA device:\n{result.stderr[-4000:]}"
    )
    assert "CPU_ONLY_OK" in result.stdout
