import os
import warnings

import torch

DeviceLike = torch.device | str | None


def is_mps_available() -> bool:
    """Check if Apple Metal Performance Shaders (MPS) backend is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def is_cuda_available() -> bool:
    """Check if NVIDIA CUDA backend is available."""
    return torch.cuda.is_available()


def is_mlx_available() -> bool:
    """Check if Apple MLX (native Metal ML framework) is available.

    MLX is Apple's purpose-built ML framework for M-series chips.
    It uses lazy evaluation and unified memory (zero CPU/GPU copy).
    When available, ``load_esmc(device="mlx")`` routes to the native
    MLX backend instead of PyTorch MPS.
    """
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    """Resolve a requested device string or torch.device to an available device.

    Supports 'auto', 'cuda', 'mps', and 'cpu', with graceful fallback
    cuda -> mps -> cpu and clear warnings when falling back from an
    explicitly requested device.

    Args:
        requested: Device specification ('auto', 'cuda', 'mps', 'cpu', None,
            or a torch.device instance). Defaults to 'auto'.

    Returns:
        torch.device: The resolved device.
    """
    if requested is None:
        requested = "auto"

    if isinstance(requested, torch.device):
        device_type = requested.type
        device_index = requested.index
    else:
        req_str = str(requested).strip()
        if ":" in req_str:
            device_type, idx_str = req_str.split(":", 1)
            try:
                device_index = int(idx_str)
            except ValueError:
                device_index = None
        else:
            device_type = req_str
            device_index = None

    device_type = device_type.lower()

    if device_type == "auto":
        if is_cuda_available():
            return torch.device("cuda")
        elif is_mps_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    elif device_type == "cuda":
        if is_cuda_available():
            if device_index is not None:
                return torch.device(f"cuda:{device_index}")
            return torch.device("cuda")
        elif is_mps_available():
            warnings.warn(
                f"CUDA device '{requested}' requested but CUDA is not available. "
                "Falling back to Apple Metal (MPS).",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("mps")
        else:
            warnings.warn(
                f"CUDA device '{requested}' requested but CUDA is not available. "
                "Falling back to CPU.",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("cpu")

    elif device_type == "mps":
        if is_mps_available():
            return torch.device("mps")
        elif is_cuda_available():
            warnings.warn(
                "Apple Metal (MPS) device requested but MPS is not available. "
                "Falling back to CUDA.",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("cuda")
        else:
            warnings.warn(
                "Apple Metal (MPS) device requested but MPS is not available. "
                "Falling back to CPU.",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("cpu")

    elif device_type == "cpu":
        return torch.device("cpu")

    else:
        return torch.device(requested)


def get_default_model_dtype(device: torch.device | str) -> torch.dtype:
    """Get the recommended default model dtype for a given device.

    On CUDA: Defaults to bfloat16 for efficiency.
    On MPS: Defaults to float32 due to historical kernel coverage limitations.
        Can be overridden via the ESM3_MPS_DTYPE environment variable
        ('bfloat16'/'bf16' or 'float16'/'fp16').
    On CPU: Defaults to float32.

    Args:
        device: Device to determine dtype for.

    Returns:
        torch.dtype: The recommended torch dtype.
    """
    if isinstance(device, str):
        dev_type = device.lower().strip()
        if dev_type == "auto":
            dev = resolve_device("auto")
            dev_type = dev.type
    elif isinstance(device, torch.device):
        dev_type = device.type
    else:
        dev_type = "cpu"

    if dev_type.startswith("cuda"):
        return torch.bfloat16
    elif dev_type.startswith("mps"):
        env_dtype = os.environ.get("ESM3_MPS_DTYPE", "").strip().lower()
        if env_dtype in ("bfloat16", "bf16"):
            return torch.bfloat16
        elif env_dtype in ("float16", "fp16", "half"):
            return torch.float16
        return torch.float32
    else:
        return torch.float32


def empty_cache(device: torch.device | str | None = None) -> None:
    """Clear cached memory for CUDA and/or MPS devices safely."""
    if is_cuda_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
    if (
        is_mps_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
    ):
        try:
            torch.mps.empty_cache()
        except RuntimeError:
            pass


def synchronize(device: torch.device | str | None = None) -> None:
    """Synchronize device execution for CUDA and/or MPS devices safely."""
    if is_cuda_available():
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
    if (
        is_mps_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "synchronize")
    ):
        try:
            torch.mps.synchronize()
        except RuntimeError:
            pass


def load_esmc(
    model_name: str = "biohub/ESMC-300M", device: str | torch.device | None = "auto"
):
    """Load an ESMC model, routing to the optimal backend for the current hardware.

    Backend selection for ``device="auto"``:
      - CUDA available        → ``EsmcForMaskedLM`` on CUDA (bfloat16)
      - MLX available (M-chip)→ ``EsmcMLX`` — native Metal, no PyTorch overhead
      - MPS only              → ``EsmcForMaskedLM`` on MPS (float32)
      - CPU fallback          → ``EsmcForMaskedLM`` on CPU

    Pass ``device="mlx"`` to force the MLX backend regardless of other devices.
    Pass ``device="mps"`` to force the PyTorch MPS backend.

    Returns either an ``EsmcForMaskedLM`` (PyTorch) or ``EsmcMLX`` instance.
    Both expose the same ``__call__(input_ids) -> logits`` interface.

    Example::

        model = load_esmc("biohub/ESMC-300M")  # auto-selects best backend
        model = load_esmc("biohub/ESMC-600M", "mlx")  # 600M, native MLX
        model = load_esmc("biohub/ESMC-6B", "mlx")  # 6B, native MLX (≥16 GB RAM)
        model = load_esmc("biohub/ESMC-6B", "cuda")  # 6B on CUDA

    """
    dev_str = str(device).lower().strip() if device is not None else "auto"

    use_mlx = dev_str == "mlx" or (
        dev_str == "auto" and not is_cuda_available() and is_mlx_available()
    )

    if use_mlx:
        from esm.models.esmc.mlx_model import EsmcMLX  # optional dep

        return EsmcMLX.from_pretrained(model_name)

    from esm.models.esmc import EsmcForMaskedLM

    torch_device = resolve_device(device)
    dtype = get_default_model_dtype(torch_device)
    return EsmcForMaskedLM.from_pretrained(
        model_name, device=torch_device, dtype=dtype or None
    )
