import inspect
import warnings
from typing import Callable

import torch
import torch.nn as nn
from accelerate import init_empty_weights

from esm.models.esm3 import ESM3
from esm.models.esmc.compatibility import ESMC
from esm.models.esmc.config import ESMC_6B_HF_REPO, ESMC_300M_HF_REPO, ESMC_600M_HF_REPO
from esm.models.esmc.model import EsmcForMaskedLM
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import StructureTokenDecoder, StructureTokenEncoder
from esm.tokenization import get_esm3_model_tokenizers
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


def _deprecated_esmc(
    new_name: str, repo: str, device: torch.device | str, use_flash_attn: bool
) -> ESMC:
    """Load an ESMC checkpoint behind one of the retired date-stamped names."""
    warnings.warn(
        f"{new_name} is deprecated; load the model directly with "
        f'EsmcForMaskedLM.from_pretrained("{repo}"). The weights now come from '
        f"the HuggingFace Hub rather than a local data root, so the first call "
        f"downloads them.",
        DeprecationWarning,
        stacklevel=3,
    )
    with warnings.catch_warnings():
        # The wrapper warns on its own; one deprecation per call is enough.
        warnings.simplefilter("ignore", DeprecationWarning)
        # dtype=None keeps fp32 on every device: the date-stamped factories
        # loaded raw fp32 weights, and only ESMC.from_pretrained cast to bf16.
        return ESMC(
            model=EsmcForMaskedLM.from_pretrained(
                repo,
                device=device,
                dtype=None,
                attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
            )
        ).eval()


def ESMC_300M_202412(
    device: torch.device | str = "cpu", use_flash_attn: bool = True
) -> ESMC:
    """Deprecated. Use ``EsmcForMaskedLM.from_pretrained(ESMC_300M_HF_REPO)``."""
    return _deprecated_esmc(
        "ESMC_300M_202412", ESMC_300M_HF_REPO, device, use_flash_attn
    )


def ESMC_600M_202412(
    device: torch.device | str = "cpu", use_flash_attn: bool = True
) -> ESMC:
    """Deprecated. Use ``EsmcForMaskedLM.from_pretrained(ESMC_600M_HF_REPO)``."""
    return _deprecated_esmc(
        "ESMC_600M_202412", ESMC_600M_HF_REPO, device, use_flash_attn
    )


def ESMC_6B_202412(
    device: torch.device | str = "cpu", use_flash_attn: bool = True
) -> ESMC:
    """Deprecated. Use ``EsmcForMaskedLM.from_pretrained(ESMC_6B_HF_REPO)``."""
    return _deprecated_esmc("ESMC_6B_202412", ESMC_6B_HF_REPO, device, use_flash_attn)


LOCAL_MODEL_REGISTRY: dict[str, ModelBuilder] = {
    ESM3_OPEN_SMALL: ESM3_sm_open_v0,
    ESM3_STRUCTURE_ENCODER_V0: ESM3_structure_encoder_v0,
    ESM3_STRUCTURE_DECODER_V0: ESM3_structure_decoder_v0,
    ESM3_FUNCTION_DECODER_V0: ESM3_function_decoder_v0,
    ESMC_300M: ESMC_300M_202412,
    ESMC_600M: ESMC_600M_202412,
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
