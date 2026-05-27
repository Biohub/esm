import inspect
from typing import Callable

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from huggingface_hub import load_torch_model

from esm.models.esm3 import ESM3
from esm.models.esmc import ESMC
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import StructureTokenDecoder, StructureTokenEncoder
from esm.tokenization import get_esm3_model_tokenizers, get_esmc_model_tokenizers
from esm.utils.constants.esm3 import data_root
from esm.utils.constants.models import (
    ESM3_FUNCTION_DECODER_V0,
    ESM3_OPEN_SMALL,
    ESM3_STRUCTURE_DECODER_V0,
    ESM3_STRUCTURE_ENCODER_V0,
    ESMC_6B,
    ESMC_300M,
    ESMC_600M,
)

ModelBuilder = Callable[[torch.device | str], nn.Module]


def ESM3_structure_encoder_v0(device: torch.device | str = "cpu"):
    with init_empty_weights():
        model = StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        ).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_structure_encoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict, assign=True)
    model = model.to(device).to(torch.float32)
    return model


def ESM3_structure_decoder_v0(device: torch.device | str = "cpu"):
    with init_empty_weights():
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_structure_decoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict, assign=True)
    model = model.to(device)
    return model


def ESM3_function_decoder_v0(device: torch.device | str = "cpu"):
    with init_empty_weights():
        model = FunctionTokenDecoder().eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_function_decoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict, assign=True)
    model = model.to(device)
    return model


def ESMC_300M_202412(device: torch.device | str = "cpu", use_flash_attn: bool = True):
    with init_empty_weights():
        model = ESMC(
            d_model=960,
            n_heads=15,
            n_layers=30,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
        ).eval()
    load_torch_model(model, data_root("esmc-300"))
    model = model.to(device)
    return model


def ESMC_600M_202412(device: torch.device | str = "cpu", use_flash_attn: bool = True):
    with init_empty_weights():
        model = ESMC(
            d_model=1152,
            n_heads=18,
            n_layers=36,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
        ).eval()
    load_torch_model(model, data_root("esmc-600"))
    model = model.to(device)
    return model


def ESMC_6B_202412(device: torch.device | str = "cpu", use_flash_attn: bool = True):
    with init_empty_weights():
        model = ESMC(
            d_model=2560,
            n_heads=40,
            n_layers=80,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=use_flash_attn,
        ).eval()
    load_torch_model(model, data_root("esmc-6b"))
    model = model.to(device)
    return model


def ESM3_sm_open_v0(device: torch.device | str = "cpu"):
    with init_empty_weights():
        model = ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=ESM3_structure_encoder_v0,
            structure_decoder_fn=ESM3_structure_decoder_v0,
            function_decoder_fn=ESM3_function_decoder_v0,
            tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL),
        ).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_sm_open_v1.pth", map_location=device
    )
    model.load_state_dict(state_dict, assign=True)
    model = model.to(device)
    return model


LOCAL_MODEL_REGISTRY: dict[str, ModelBuilder] = {
    ESM3_OPEN_SMALL: ESM3_sm_open_v0,
    ESM3_STRUCTURE_ENCODER_V0: ESM3_structure_encoder_v0,
    ESM3_STRUCTURE_DECODER_V0: ESM3_structure_decoder_v0,
    ESM3_FUNCTION_DECODER_V0: ESM3_function_decoder_v0,
    ESMC_600M: ESMC_600M_202412,
    ESMC_300M: ESMC_300M_202412,
    ESMC_6B: ESMC_6B_202412,
}


def load_local_model(
    model_name: str,
    device: torch.device = torch.device("cpu"),
    use_flash_attn: bool = True,
) -> nn.Module:
    if model_name not in LOCAL_MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in local model registry.")
    builder = LOCAL_MODEL_REGISTRY[model_name]
    kwargs = {}
    if "use_flash_attn" in inspect.signature(builder).parameters:
        kwargs["use_flash_attn"] = use_flash_attn
    return builder(device, **kwargs)


# Register custom versions of ESM3 for use with the local inference API
def register_local_model(model_name: str, model_builder: ModelBuilder) -> None:
    LOCAL_MODEL_REGISTRY[model_name] = model_builder
