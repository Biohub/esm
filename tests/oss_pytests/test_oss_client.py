import os

import pytest
import torch

from esm.sdk import client, esmc_client  # pyright: ignore
from esm.sdk.api import (  # pyright: ignore
    ESMProtein,
    ESMProteinTensor,
    FoldingConfig,
    ForwardAndSampleOutput,
    GenerationConfig,
    LogitsConfig,
    LogitsOutput,
    SAEConfig,
    SamplingConfig,
    SamplingTrackConfig,
)
from esm.sdk.forge import SequenceStructureForgeInferenceClient  # pyright: ignore

API_TOKEN = os.environ.get("ESM_API_KEY", "")
URL = os.environ.get("URL")
assert URL is not None, "URL must be set in the environment"


@pytest.mark.sdk
def test_oss_esm3_client():
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
    sequence = "MALWMRLLPLLALLALAVPDPAAA"
    model = "esmc-300m-2024-12"
    sae_model = "esmc-300m-2024-12-sae-layer23-k64-codebook65536"
    esmc = esmc_client(model=model, url=URL, token=API_TOKEN)

    protein = ESMProtein(sequence)
    encoded_protein = esmc.encode(input=protein)
    assert isinstance(encoded_protein, ESMProteinTensor)
    assert encoded_protein.sequence is not None

    decoded_protein = esmc.decode(input=encoded_protein)
    assert isinstance(decoded_protein, ESMProtein)

    logits_config = LogitsConfig(
        sequence=True,
        return_embeddings=True,
        return_hidden_states=True,
        sae_config=SAEConfig(models=[sae_model], normalize_features=False),
    )
    result = esmc.logits(input=encoded_protein, config=logits_config)
    assert isinstance(result, LogitsOutput)
    assert result.logits is not None
    assert isinstance(result.logits.sequence, torch.Tensor)
    assert result.sae_outputs is not None
    assert sae_model in result.sae_outputs

    # SAE features are returned sparse, one row per non-pad token.
    features = result.sae_outputs[sae_model].to_dense()
    assert features.shape[0] == encoded_protein.sequence.shape[0]
    assert (features != 0).any()


@pytest.mark.sdk
def test_oss_sequence_structure_forge_inference_client():
    sequence = "MALWMRLLPLLALLALAVPDPAAA"
    model = "esm3-small-2024-03"
    client = SequenceStructureForgeInferenceClient(
        model=model, url=URL, token=API_TOKEN
    )

    encoded_protein = client.fold(sequence=sequence)
    assert isinstance(encoded_protein, ESMProtein)


@pytest.mark.sdk
def test_oss_esmfold2_client():
    sequence = "MALWMRLLPLLALLALAVPDPAAA"
    model = "esmfold2-fast-2026-05"
    fold_client = SequenceStructureForgeInferenceClient(
        model=model, url=URL, token=API_TOKEN
    )

    config = FoldingConfig(num_sampling_steps=1, num_loops=1)
    folded_protein = fold_client.fold(sequence=sequence, config=config)
    assert isinstance(folded_protein, ESMProtein)
    assert folded_protein.coordinates is not None
    assert folded_protein.plddt is not None
