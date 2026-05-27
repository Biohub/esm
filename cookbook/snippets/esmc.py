import math
import os

import torch

from esm.models.esmc import ESMC
from esm.sdk import batch_executor, esmc_client
from esm.sdk.api import (
    ESMCInferenceClient,
    ESMProtein,
    ESMProteinError,
    ESMProteinTensor,
    LogitsConfig,
    LogitsOutput,
)
from esm.sdk.forge import ESM3ForgeInferenceClient, ESMCForgeInferenceClient
from esm.tokenization import get_esmc_model_tokenizers


def main(client: ESMCInferenceClient | ESM3ForgeInferenceClient):
    # ================================================================
    # Example usage: one single protein
    # ================================================================
    protein = ESMProtein(sequence="AAAAA")

    # Use logits endpoint. Using bf16 for inference optimization
    protein_tensor = client.encode(protein)
    assert isinstance(
        protein_tensor, ESMProteinTensor
    ), f"Expected ESMProteinTensor but got error: {protein_tensor}"
    output = client.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True),
    )
    assert isinstance(
        output, LogitsOutput
    ), f"LogitsOutput was expected but got error: {output}"
    assert output.logits is not None and output.logits.sequence is not None
    assert output.embeddings is not None
    assert output.hidden_states is not None
    print(
        f"Client returned logits with shape: {output.logits.sequence.shape}, embeddings with shape: {output.embeddings.shape}, and hidden states with shape {output.hidden_states.shape}"
    )

    # request a specific hidden layer.
    assert isinstance(
        protein_tensor, ESMProteinTensor
    ), f"Expected ESMProteinTensor but got error: {protein_tensor}"
    output = client.logits(
        protein_tensor, LogitsConfig(return_hidden_states=True, ith_hidden_layer=1)
    )
    assert isinstance(
        output, LogitsOutput
    ), f"LogitsOutput was expected but got error: {output}"
    assert output.hidden_states is not None
    print(f"Client returned hidden states with shape {output.hidden_states.shape}")


def raw_forward(model: ESMC):
    protein = ESMProtein(sequence="AAAAA")
    assert protein.sequence is not None
    sequences = [protein.sequence, protein.sequence]
    # ================================================================
    # Example usage: directly use the model
    # ================================================================
    input_ids = model._tokenize(sequences)
    output = model(input_ids)
    logits, embeddings, hiddens = (
        output.sequence_logits,
        output.embeddings,
        output.hidden_states,
    )
    print(
        f"Raw model returned logits with shape: {logits.shape}, embeddings with shape: {embeddings.shape} and hidden states with shape {hiddens.shape}"
    )


def compute_pseudoperplexity(
    forge_client: ESMCForgeInferenceClient, sequence: str
) -> float:
    """Compute L-pass pseudoperplexity for a protein sequence via Forge/Biohub Platform.

    Masks each position one at a time, retrieves logits from Forge/Biohub Platform, and returns
    exp(-mean(log_prob_true_aa)).  Uses batch_executor for parallel requests.

    Example::

        forge_client = ESMCForgeInferenceClient(
            model="esmc-6b-2024-12",
            url="https://biohub.ai",
            token=os.environ["ESM_API_KEY"],
        )
        pppl = compute_pseudoperplexity(forge_client, "MKTLLILAVL...")
    """
    L = len(sequence)
    masked_sequences = [sequence[:i] + "_" + sequence[i + 1 :] for i in range(L)]

    def _get_logits(client: ESMCForgeInferenceClient, sequence: str) -> LogitsOutput:
        protein = ESMProtein(sequence=sequence)
        protein_tensor = client.encode(protein)
        if isinstance(protein_tensor, ESMProteinError):
            raise protein_tensor
        output = client.logits(protein_tensor, LogitsConfig(sequence=True))
        if isinstance(output, ESMProteinError):
            raise output
        return output

    with batch_executor() as executor:
        logit_outputs = executor.execute_batch(
            _get_logits, client=forge_client, sequence=masked_sequences
        )

    # Build vocab from the tokenizer to map amino acid characters to token indices
    vocab: dict[str, int] = get_esmc_model_tokenizers().get_vocab()

    log_probs = []
    for i in range(L):
        output = logit_outputs[i]
        if isinstance(output, Exception):
            raise output
        logits = output.logits.sequence  # shape: (L+2, V)
        position_logits = logits[i + 1]  # +1 for BOS token
        log_softmax = torch.log_softmax(position_logits, dim=-1)
        true_aa_idx = vocab[sequence[i]]
        log_probs.append(log_softmax[true_aa_idx].item())

    return math.exp(-sum(log_probs) / L)


if __name__ == "__main__":
    if os.environ.get("ESM_API_KEY", ""):
        print("ESM_API_KEY found. Trying to use model from Forge/Biohub Platform...")
        main(esmc_client(model="esmc-300m-2024-12"))
    else:
        print("No ESM_API_KEY found. Trying to load model locally...")
        print(
            "To try this script with a Forge/Biohub Platform API, please run ESM_API_KEY=your_api_key python esm3.py"
        )
        main(ESMC.from_pretrained("esm3_sm_open_v1"))
        model = ESMC.from_pretrained("esmc_300m")
        main(model)
        raw_forward(model)
