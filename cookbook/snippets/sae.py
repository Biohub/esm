import numpy as np
import torch

from cookbook.snippets.sparse_utils import max_pool, remove_indexes
from esm.sdk import batch_executor
from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig, SAEConfig
from esm.sdk.forge import ESMCForgeInferenceClient


def get_sae_features_single(
    client: ESMCForgeInferenceClient,
    sae_config: SAEConfig,
    sequence: str,
    pool: bool = True,
) -> torch.Tensor:
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    if isinstance(protein_tensor, ESMProteinError):
        raise ValueError(
            f"Error encoding sequence {sequence}: {protein_tensor.error_msg}"
        )

    # We wrap the SAEConfig in the LogitsConfig, which is normally used to return embeddings and hidden states.
    output = client.logits(
        protein_tensor, config=LogitsConfig(sae_config=sae_config), return_bytes=False
    )
    if isinstance(output, ESMProteinError):
        raise ValueError(
            f"Error getting SAE features for sequence {sequence}: {output.error_msg}"
        )
    if output.sae_outputs is None:
        raise ValueError(f"SAE outputs missing for sequence {sequence}: {output}")
    sae_tensor = output.sae_outputs[sae_config.models[0]]

    if pool:
        # Remove BOS / EOS tokens before pooling.
        sae_features = remove_indexes(sae_tensor, {0, -1})
        pooled_sae_features = max_pool(sae_features, axis=0)
        return pooled_sae_features
    else:
        return sae_tensor


def get_sae_features(
    client: ESMCForgeInferenceClient,
    sae_config: SAEConfig,
    sequences: list[str],
    pool: bool = True,
) -> list[np.ndarray]:
    with batch_executor() as executor:
        results = executor.execute_batch(
            user_func=get_sae_features_single,
            client=client,
            sae_config=sae_config,
            sequence=sequences,
            pool=pool,
        )
    # Re-raise any errors from the batch
    for result in results:
        if isinstance(result, Exception):
            raise result
    return results
