import os

import pytest
import torch

from esm.sdk import client  # pyright: ignore
from esm.sdk.api import (  # pyright: ignore
    ESMProtein,
    ESMProteinTensor,
    ForwardAndSampleOutput,
    GenerationConfig,
    LogitsConfig,
    LogitsOutput,
    SamplingConfig,
    SamplingTrackConfig,
)
from esm.sdk.forge import SequenceStructureForgeInferenceClient  # pyright: ignore

API_TOKEN = os.environ.get("ESM_API_KEY", "")
URL = os.environ.get("URL")


@pytest.mark.sdk
def test_oss_esm3_client():
    assert URL is not None

    sequence = "MALWMRLLPLLALLAL___PDPAAA"
    model = "esm3-small-2024-03"
    esm3_client = client(model=model, url=URL, token=API_TOKEN)
    protein = ESMProtein(sequence)

    encoded_protein = esm3_client.encode(input=protein)
    assert isinstance(encoded_protein, ESMProteinTensor)

    decoded_protein = esm3_client.decode(input=encoded_protein)
    assert isinstance(decoded_protein, ESMProtein)

    logits_config = LogitsConfig(sequence=True, return_embeddings=True)
    result = esm3_client.logits(input=encoded_protein, config=logits_config)
    assert isinstance(result, LogitsOutput)
    assert result.logits is not None
    assert isinstance(result.logits.sequence, torch.Tensor)

    sampling_config = SamplingConfig(sequence=SamplingTrackConfig(temperature=0.1))
    result = esm3_client.forward_and_sample(
        input=encoded_protein, sampling_configuration=sampling_config
    )
    assert isinstance(result, ForwardAndSampleOutput)

    generation_config = GenerationConfig(track="sequence", num_steps=4)
    result = esm3_client.generate(input=protein, config=generation_config)
    assert isinstance(result, ESMProtein)


@pytest.mark.sdk
def test_oss_esmc_client():
    assert URL is not None

    sequence = "MALWMRLLPLLALLALAVPDPAAA"
    model = "esmc-300m-2024-12"
    esmc_client = client(model=model, url=URL, token=API_TOKEN)

    protein = ESMProtein(sequence)
    encoded_protein = esmc_client.encode(input=protein)
    assert isinstance(encoded_protein, ESMProteinTensor)

    decoded_protein = esmc_client.decode(input=encoded_protein)
    assert isinstance(decoded_protein, ESMProtein)

    logits_config = LogitsConfig(
        sequence=True, return_embeddings=True, return_hidden_states=True
    )
    result = esmc_client.logits(input=encoded_protein, config=logits_config)
    assert isinstance(result, LogitsOutput)
    assert result.logits is not None
    assert isinstance(result.logits.sequence, torch.Tensor)


@pytest.mark.sdk
def test_oss_sequence_structure_forge_inference_client():
    assert URL is not None

    sequence = "MALWMRLLPLLALLALAVPDPAAA"
    model = "esm3-small-2024-03"
    client = SequenceStructureForgeInferenceClient(
        model=model, url=URL, token=API_TOKEN
    )

    encoded_protein = client.fold(sequence=sequence)
    assert isinstance(encoded_protein, ESMProtein)
