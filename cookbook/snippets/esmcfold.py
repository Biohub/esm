import os

from esm.sdk.forge import SequenceStructureForgeInferenceClient
from esm.sdk.api import FoldingConfig
from esm.utils.structure import input_builder


def get_GLP1_sequence() -> str:
    glp1_sequence = "MKTIIALSYIFCLVFADYKDDDDLEVLFQGPARPQGATVSLWETVQKWREYRRQCQRSLTEDPPPATDLFCNRTFDEYACWPDGEPGSFVNVSCPWYLPWASSVPQGHVYRFCTAEGLWLQKDNSSLPWRDLSECEESKRGERSSPEEQLLFLYIIYTVGYALSFSALVIASAILLGFRHLHCTRNYIHLNLFASFILRALSVFIKDAALKWMYSTAAQQHQWDGLLSYQDSLSCRLVFLLMQYCVAANYYWLLVEGVYLYTLLAFSVFSEQWIFRLYVSIGWGVPLLFVVPWGIVKYLYEDEGCWTRNSNMNYWLIIRLPILFAIGVNFLIFVRVICIVVSKLKANLMCKTDIKCRLAKSTLTLIPLLGTHEVIFAFVMDEHARGTLRFIKLFTELSFTSFQGLMVAILYCFVNNEVQLEFRKSWERWRLEHLHIQRDSSMKPLKCPTSSLSSGATAGSSMYTATCQASCSPAGLEVLFQGPHHHHHHH"
    return glp1_sequence


def get_semaglutide_peptide_sequence() -> str:
    semaglutide_peptide = "HAEGTFTSDVSSYLEGQAAKEFIAWLVRGRG"  # From PDB: 7KI0
    return semaglutide_peptide


def get_input() -> input_builder.StructurePredictionInput:
    """
    Builds the input for folding semaglutide in complex with GLP-1 receptor
    """
    glp1_sequence = get_GLP1_sequence()
    glp1_chain_id = "A"
    peptide_sequence = get_semaglutide_peptide_sequence()
    peptide_chain_id = "B"
    peptide_aib_index = 1

    glp1_receptor = input_builder.ProteinInput(
        id=glp1_chain_id,
        sequence=glp1_sequence,  # GLP-1 receptor
    )
    peptide = input_builder.ProteinInput(
        id=peptide_chain_id,
        sequence=peptide_sequence,
        modifications=[
            input_builder.Modification(position=peptide_aib_index, ccd="AIB")
        ],
    )
    folding_input = input_builder.StructurePredictionInput(
        sequences=[
            glp1_receptor,
            peptide,
        ],
    )
    return folding_input


# --- Main folding loop

MODEL = "esmc-fold-2025-04"
URL = "https://forge.evolutionaryscale.ai"
API_TOKEN = os.environ.get("ESM_FORGE_API_TOKEN", "")

client = SequenceStructureForgeInferenceClient(url=URL, model=MODEL, token=API_TOKEN)

folding_config = FoldingConfig(
    num_recycles=3,
    num_sampling_steps=200,
    seed=2,
)
inputs = get_input()

result = client.fold_all_atom(inputs, config=folding_config)

# export to mmcif for visualization in pymol
with open("./folded.cif", "w") as f:
    f.write(result.complex.to_mmcif())