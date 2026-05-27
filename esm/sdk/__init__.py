import os

from esm.sdk.api import ESM3InferenceClient, ESMCInferenceClient
from esm.sdk.forge import (
    ESM3ForgeInferenceClient,
    ESMCForgeInferenceClient,
    SequenceStructureForgeInferenceClient,
)
from esm.utils.forge_context_manager import ForgeBatchExecutor

# Note: please do not import ESM3SageMakerClient here since that requires AWS SDK.


def client(
    model="esm3-sm-open-v1",
    url="https://biohub.ai",
    token=os.environ.get("ESM_API_KEY", ""),
    request_timeout=None,
) -> ESM3InferenceClient:
    """
    Args:
        model: Name of the esm3 model to use.
        url: URL of a Forge/Biohub Platform server.
        token: User's API token.
        request_timeout: Amount of time to wait for a request to finish.
            Default is wait indefinitely.
    """
    if not model.startswith("esm3"):
        raise ValueError(f"Invalid model name: {model}")
    return ESM3ForgeInferenceClient(model, url, token, request_timeout)


def esmc_client(
    model="esmc-300m-2024-12",
    url="https://biohub.ai",
    token=os.environ.get("ESM_API_KEY", ""),
    request_timeout=None,
) -> ESMCInferenceClient:
    """
    Args:
        model: Name of the esmc model to use.
        url: URL of a Forge/Biohub Platform server.
        token: User's API token.
        request_timeout: Amount of time to wait for a request to finish.
            Default is wait indefinitely.
    """
    if not model.startswith("esmc"):
        raise ValueError(f"Invalid model name: {model}")
    return ESMCForgeInferenceClient(model, url, token, request_timeout)


def esmfold2_client(
    model="esmfold2-fast-2026-05",
    url="https://biohub.ai",
    token=os.environ.get("ESM_API_KEY", ""),
    request_timeout=None,
) -> SequenceStructureForgeInferenceClient:
    """
    Args:
        model: Name of the ESMFold2 model to use.
        url: URL of a Forge/Biohub Platform server.
        token: User's API token.
        request_timeout: Amount of time to wait for a request to finish.
            Default is wait indefinitely.
    """
    if not model.startswith("esmfold2"):
        raise ValueError(f"Invalid model name: {model}")
    return SequenceStructureForgeInferenceClient(
        model=model, url=url, token=token, request_timeout=request_timeout
    )


def batch_executor(max_attempts: int = 10, show_progress: bool = True):
    """
    Args:
        max_attempts: Maximum number of attempts to make before giving up.
        show_progress: Whether to display a tqdm progress bar.

    Usage:
        with batch_executor(show_progress=False) as executor:
            executor.execute_batch(fn, **kwargs)
    """
    return ForgeBatchExecutor(max_attempts=max_attempts, show_progress=show_progress)
