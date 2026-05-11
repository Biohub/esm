# Model names
ESM3_OPEN_SMALL = "esm3_sm_open_v1"
ESM3_OPEN_SMALL_ALIAS_1 = "esm3-open-2024-03"
ESM3_OPEN_SMALL_ALIAS_2 = "esm3-sm-open-v1"
ESM3_OPEN_SMALL_ALIAS_3 = "esm3-open"
ESM3_STRUCTURE_ENCODER_V0 = "esm3_structure_encoder_v0"
ESM3_STRUCTURE_DECODER_V0 = "esm3_structure_decoder_v0"
ESM3_FUNCTION_DECODER_V0 = "esm3_function_decoder_v0"
ESMC_600M = "esmc_600m"
ESMC_300M = "esmc_300m"

ESMC_FOLD_MODELS_WITHOUT_MSA = ["esmc-fold-flash-2604"]


def forge_only_return_single_layer_hidden_states(model_name: str):
    return model_name.startswith("esmc-6b")


def model_is_locally_supported(x: str):
    return x in {
        ESM3_OPEN_SMALL,
        ESM3_OPEN_SMALL_ALIAS_1,
        ESM3_OPEN_SMALL_ALIAS_2,
        ESM3_OPEN_SMALL_ALIAS_3,
    }


def normalize_model_name(x: str):
    if x in {ESM3_OPEN_SMALL_ALIAS_1, ESM3_OPEN_SMALL_ALIAS_2, ESM3_OPEN_SMALL_ALIAS_3}:
        return ESM3_OPEN_SMALL
    return x
