import os
from typing import cast

import numpy as np

from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    GenerationConfig,
    InverseFoldingConfig,
)
from esm.sdk.forge import (
    ESM3ForgeInferenceClient,
    SequenceStructureForgeInferenceClient,
)
from esm.utils.structure.protein_chain import ProteinChain
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


def convert_none_to_nan(data):
    """Recursively convert None values in any deeply nested structure (e.g., list of lists of lists) to np.nan."""
    if isinstance(data, list):
        return [convert_none_to_nan(x) for x in data]
    elif data is None:
        return np.nan
    else:
        return data


def fold(
    sequence_structure_client: SequenceStructureForgeInferenceClient,
    esm3_client: ESM3InferenceClient,
):
    protein = get_sample_protein()
    protein.coordinates = None
    protein.function_annotations = None
    protein.sasa = None
    assert protein.sequence is not None, "Protein sequence must be set to fold"

    # Folding with esm3 client
    config = GenerationConfig(track="structure", num_steps=1, temperature=0)
    esm3_client_folded_protein = esm3_client.generate(protein, config)
    assert isinstance(
        esm3_client_folded_protein, ESMProtein
    ), f"Using ESM3 client, ESMProtein was expected but got {esm3_client_folded_protein}"

    # Folding with folding client
    sequence_structure_client_folded_protein = sequence_structure_client.fold(
        protein.sequence, potential_sequence_of_concern=False
    )
    assert isinstance(
        sequence_structure_client_folded_protein, ESMProtein
    ), f"Using sequence_structure client, ESMProtein was expected but got {sequence_structure_client_folded_protein}"
    sequence_structure_client_folded_protein.to_pdb("folded_protein.pdb")
    print("Saving folded protein to folded_protein.pdb")


def inverse_fold(
    sequence_structure_client: SequenceStructureForgeInferenceClient,
    esm3_client: ESM3InferenceClient,
):
    protein = get_sample_protein()
    protein.sequence = None
    protein.sasa = None
    protein.function_annotations = None
    assert (
        protein.coordinates is not None
    ), "Protein coordinates must be set to inverse fold"

    # Inverse Folding with esm3 client
    config = GenerationConfig("sequence", num_steps=1, temperature=0.1)
    esm3_client_inv_folded_protein = cast(
        ESMProtein, esm3_client.generate(protein, config)
    )
    assert isinstance(
        esm3_client_inv_folded_protein, ESMProtein
    ), f"Using ESM3 client, ESMProtein was expected but got {esm3_client_inv_folded_protein}"

    # Inverse Folding with inverse folding client
    sequence_structure_client_inv_folded_protein = (
        sequence_structure_client.inverse_fold(
            protein.coordinates,
            config=InverseFoldingConfig(temperature=0.1),
            potential_sequence_of_concern=False,
        )
    )
    assert isinstance(
        sequence_structure_client_inv_folded_protein, ESMProtein
    ), f"Using sequence_structure client, ESMProtein was expected but got {sequence_structure_client_inv_folded_protein}"
    print(
        f"Inverse folded protein: {sequence_structure_client_inv_folded_protein.sequence}"
    )


if __name__ == "__main__":
    if not os.environ.get("ESM_API_KEY", ""):
        print(
            "Please export your Forge/Biohub Platform API key as ESM_API_KEY environment variable."
        )
    client = SequenceStructureForgeInferenceClient(token=os.environ["ESM_API_KEY"])
    esm3_client = ESM3ForgeInferenceClient(
        model="esm3-medium-2024-08", token=os.environ["ESM_API_KEY"]
    )
    fold(client, esm3_client)
    inverse_fold(client, esm3_client)
