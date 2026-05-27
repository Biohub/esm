import os

from esm.models.esm3 import ESM3
from esm.sdk import client
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    ESMProteinTensor,
    GenerationConfig,
    LogitsConfig,
    LogitsOutput,
    SamplingConfig,
    SamplingTrackConfig,
)
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex
from esm.utils.types import FunctionAnnotation


def get_sample_protein() -> ESMProtein:
    protein = ProteinChain.from_rcsb("1utn")
    protein = ESMProtein.from_protein_chain(protein)
    protein.function_annotations = [
        # Peptidase S1A, chymotrypsin family: https://www.ebi.ac.uk/interpro/structure/PDB/1utn/
        FunctionAnnotation(label="peptidase", start=100, end=114),
        FunctionAnnotation(label="chymotrypsin", start=190, end=202),
    ]
    return protein


def get_sample_protein_complex() -> ESMProtein:
    protein = ProteinComplex.from_rcsb("7a3w")
    protein = ESMProtein.from_protein_complex(protein)
    return protein


def main(client: ESM3InferenceClient):
    # Single step decoding
    protein = get_sample_protein()
    protein.function_annotations = None
    protein = client.encode(protein)
    single_step_protein = client.forward_and_sample(
        protein, SamplingConfig(structure=SamplingTrackConfig(topk_logprobs=2))
    )
    single_step_protein.protein_tensor.sequence = protein.sequence
    single_step_protein = client.decode(single_step_protein.protein_tensor)

    # Generate with partial sequence.
    prompt = (
        "___________________________________________________DQATSLRILNNGHAFNVEFDDSQDKAVLK"
        "GGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIK"
        "TKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPP___________________________________________________________"
    )
    protein = ESMProtein(sequence=prompt)
    protein = client.generate(
        protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7)
    )
    assert isinstance(protein, ESMProtein), f"ESMProtein was expected but got {protein}"

    # Folding
    protein = get_sample_protein()
    sequence_length = len(protein.sequence)  # type: ignore
    num_steps = int(sequence_length / 16)
    protein.coordinates = None
    protein.function_annotations = None
    protein.sasa = None
    folded_protein = client.generate(
        protein,
        GenerationConfig(track="structure", schedule="cosine", num_steps=num_steps),
    )
    assert isinstance(
        folded_protein, ESMProtein
    ), f"ESMProtein was expected but got {protein}"
    folded_protein.to_pdb("./sample_folded.pdb")

    # Inverse folding
    protein = get_sample_protein()
    protein.sequence = None
    protein.sasa = None
    protein.function_annotations = None
    inv_folded_protein = client.generate(
        protein,
        GenerationConfig(track="sequence", schedule="cosine", num_steps=num_steps),
    )
    assert isinstance(inv_folded_protein, ESMProtein)
    inv_folded_protein.to_pdb("./sample_inv_folded.pdb")

    # Function prediction
    protein = get_sample_protein()
    protein.function_annotations = None
    protein_with_function = client.generate(
        protein,
        GenerationConfig(track="function", schedule="cosine", num_steps=num_steps),
    )
    assert isinstance(
        protein_with_function, ESMProtein
    ), f"{protein_with_function} is not an ESMProtein"

    # Logits
    protein = get_sample_protein()
    protein.coordinates = None
    protein.function_annotations = None
    protein.sasa = None
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    assert isinstance(
        logits_output, LogitsOutput
    ), f"LogitsOutput was expected but got {logits_output}"
    assert (
        logits_output.logits is not None
        and logits_output.logits.sequence is not None
        and logits_output.embeddings is not None
    )

    # Chain of Thought (Function -> Secondary Structure -> Structure -> Sequence)
    cot_protein = get_sample_protein()
    cot_protein.sequence = "_" * len(cot_protein.sequence)  # type: ignore
    cot_protein.coordinates = None
    cot_protein.sasa = None
    cot_protein_tensor = client.encode(cot_protein)
    for cot_track in ["secondary_structure", "structure", "sequence"]:
        cot_protein_tensor = client.generate(
            cot_protein_tensor,
            GenerationConfig(track=cot_track, schedule="cosine", num_steps=10),
        )
    assert isinstance(
        cot_protein_tensor, ESMProteinTensor
    ), f"ESMProteinTensor was expected but got {cot_protein_tensor}"
    cot_protein = client.decode(cot_protein_tensor)

    assert isinstance(
        cot_protein, ESMProtein
    ), f"ESMProtein was expected but got {cot_protein}"
    cot_protein.to_pdb("./sample_cot.pdb")

    # Protein Complex
    protein = get_sample_protein_complex()
    sequence_length = len(protein.sequence)  # type: ignore
    num_steps = 1
    folded_protein = client.generate(
        protein,
        GenerationConfig(
            track="structure", schedule="cosine", num_steps=num_steps, temperature=0.0
        ),
    )
    assert isinstance(
        folded_protein, ESMProtein
    ), f"ESMProtein was expected but got {protein}"
    folded_protein.to_pdb("./sample_folded_complex.pdb")

    # Batch examples.

    # Batch generation.
    prompts = [ESMProtein(sequence=("_" * (10 + 2 * i))) for i in range(5)]
    configs = [
        GenerationConfig(track="sequence", schedule="cosine", num_steps=(i + 1))
        for i in range(5)
    ]
    proteins = client.batch_generate(prompts, configs)

    # Batch folding.
    # Take the list of proteins batch generated from last step.
    configs = [
        GenerationConfig(track="structure", schedule="cosine", num_steps=(i + 1))
        for i in range(5)
    ]
    # Generate again for the structure track.
    proteins = client.batch_generate(proteins, configs)
    # Now write sequence and structure to PDB files.
    for i, p in enumerate(proteins):
        assert isinstance(p, ESMProtein), f"ESMProtein was expected but got {p}"
        p.to_pdb(f"./batch_gen_{i}.pdb")

    # Batch generation returns ESMProteinError for specific prompts that have issues.
    prompts = [ESMProtein(sequence=("_" * (10 + 2 * i))) for i in range(5)]
    # Mock error situation. The third prompt has no masks to be sampled.
    prompts[2].sequence = "ANTVPYQ"
    configs = [
        GenerationConfig(track="sequence", schedule="cosine", num_steps=(i + 1))
        for i in range(5)
    ]
    proteins = client.batch_generate(prompts, configs)
    # Should still get results. But third result is a ESMProteinError.
    for i, p in enumerate(proteins):
        if i == 2:
            assert isinstance(
                p, ESMProteinError
            ), f"ESMProteinError was expected but got {p}"

        else:
            assert isinstance(p, ESMProtein), f"ESMProtein was expected but got {p}"


if __name__ == "__main__":
    if os.environ.get("ESM_API_KEY", ""):
        print("ESM_API_KEY found. Trying to use model from Forge/Biohub Platform...")
        main(client())
    else:
        print("No ESM_API_KEY found. Trying to load model locally...")
        print(
            "To try this script with a Forge/Biohub Platform API, please run ESM_API_KEY=your_api_key python esm3.py"
        )
        main(ESM3.from_pretrained("esm3_sm_open_v1"))
