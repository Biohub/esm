import os

import huggingface_hub
import huggingface_hub.errors
import torch

from esm.models.esm3 import ESM3
from esm.sdk import ESM3ForgeInferenceClient
from esm.sdk.api import ESM3InferenceClient


def get_local_client() -> ESM3InferenceClient:
    try:
        huggingface_hub.whoami()
    except huggingface_hub.errors.LocalTokenNotFoundError:
        raise ValueError("Hugging Face token not found.")
    return ESM3.from_pretrained(device=torch.device("cuda"))


def get_forge_client(model_name: str) -> ESM3InferenceClient:
    forge_token = os.environ.get("ESM_API_KEY", None)
    if forge_token is None:
        raise ValueError(
            "Forge/Biohub Platform API key not found. Please set the ESM_API_KEY environment variable."
        )
    return ESM3ForgeInferenceClient(
        model=model_name, url="https://biohub.ai", token=forge_token
    )
