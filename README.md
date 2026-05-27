<div align="center">
  <img src="_assets/header.png" style="width: 60%; height: auto;" />

# A world model of protein biology: ESMC, ESMFold2, & ESM Atlas


[ESMC & ESMFold2 Preprint](https://biohub.ai/papers/esm_protein.pdf) &sdot;  [Atlas](https://biohub.ai/esm/protein/atlas) &sdot; [Tutorials](https://github.com/Biohub/esm/tree/main/cookbook/tutorials) &sdot; [Slack](https://bit.ly/esm-slack)<br>
</div>

We are releasing a world model for protein biology: a scientific engine for prediction, design, and discovery. Built on the latest generation of Evolutionary Scale Modeling (ESM), this system learns from the protein sequences produced by evolution and uses that knowledge to represent, map, predict, and design proteins across scales — from atomic interactions to evolutionary relationships spanning billions of years. The system includes three artifacts: ESMC, ESMFold2, and ESM Atlas.

**[ESMC](https://biohub.ai/esm/protein)** is a state-of-the-art protein language model that has learned the rules of protein biology from training on billions of protein sequences. ESMC defines a new scaling frontier relative to ESM2, achieving stronger performance in emergent long-range structural understanding as model scale increases


<div align="center">
  <img src="_assets/esmc_graphic.png" width="40%"/>
</div>



**[ESMFold2](https://huggingface.co/Biohub/ESMFold2)**, built on the ESMC 6B model, is a state-of-the-art structure prediction model that has been validated for the design of protein-protein interactions. ESMFold2 surpasses other models in DockQ pass-rate on Foldbench protein-protein and antibody-antigen complexes, and can be used in single-sequence mode for an order of magnitude speedup in folding.

<div align="center">
  <img src="_assets/esmfold2_folding.png" width="40%"/>
</div>


ESMFold2 is validated in the lab across five therapeutic targets. Inversion of ESMFold2 enables generation of de novo minibinders and antibody-derived scFvs with high hit rates, nanomolar affinities, target specificity, and functional activity. We're planning to release a notebook that walks through the full design loop from target sequence to ranked binder candidates. The full protocol is also described in the [preprint](https://biohub.ai/papers/esm_protein.pdf).

<div align="center">
  <img src="_assets/esmfold2_binder.png" width="60%"/>
</div>


The **[ESM Atlas](https://biohub.ai/esm/protein/atlas)** is a map of 6.8 billion proteins covering the full breadth of life’s biodiversity. ESMFold2’s folding throughput enabled the prediction of more than one billion predicted structures. The Atlas is organized according to the internal world model of ESMC. We make this world model interpretable by training sparse autoencoders (SAEs). SAEs are unsupervised neural networks trained to decompose ESMC internal representations into a sparse set of ~16,000 interpretable features that reveal the functional relationships between proteins that ESMC has learned. Each feature is summarized in natural language with an agentic pipeline that maps features onto known biology from protein databases. We release a collection of SAEs trained on different model scales, layers, and at different levels of granularity. Learn more about how to use the ESM Atlas on the [Biohub Platform](https://biohub.ai/).

For information on using ESM3, see the [ESM3 README](https://github.com/Biohub/esm/blob/main/_assets/ESM3_README.md).

## Table of Contents

- [ESMC](#esmc)
- [ESMC Sparse Autoencoders](#esmc-sparse-autoencoders)
- [ESMFold2](#esmfold2)
- [Frontier-Safety](#frontier-safety)
- [Licenses](#licenses)
- [Citations](#citations)

## ESMC
<a name="esmc"></a>

[ESMC](https://biohub.ai/esm/protein) is a state-of-the-art protein language model that has learned representations of protein biology from training on billions of protein sequences.

Codebase, model weights, and model variants for ESMC are available through [Hugging Face](https://huggingface.co/collections/Biohub/esmc-model-family).

There are two primary ways of running the ESM models: through the [**Biohub Platform**](https://biohub.ai/) or locally with Hugging Face. The Biohub Platform enables users to easily run inference with ESM models with minimal setup. Users interested in customizing or fine-tuning ESM models can use the models from Hugging Face.

### Running ESMC Locally
<a name="running-esmc-locally"></a>

The following code demonstrates how to run ESMC locally

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import login

# login with your Hugging Face credentials
login()

# example GFP sequence
sequences = ["MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"]

model = AutoModelForMaskedLM.from_pretrained(
    "Biohub/ESMC-6B",
    device_map="auto",
).eval()
tokenizer = AutoTokenizer.from_pretrained("Biohub/ESMC-6B")

inputs = tokenizer(sequences, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.inference_mode():
    output = model(**inputs)
```

By default, the model returns only the final layer representations. To return hidden states from **all transformer layers**, set:
```python
output = model(**inputs, output_hidden_states=True)
```

### Running ESMC Through the Biohub Platform
<a name="running-esmc-through-biohub-platform"></a>

The code below shows how to access ESMC using the Biohub Platform. API tokens can be created in the [developer console](https://biohub.ai/developer-console/api-keys).

Note that our API migrated from forge.evolutionaryscale.ai to [biohub.ai](https://biohub.ai), so some code classes reference “Forge”.

To get started with ESM, install the python library using `pip`:

```
pip install "esm @ git+https://github.com/Biohub/esm.git@c94ed8d763bbd7088b296949e5b401e8ea12073a"
```

Then import the necessary libraries and instantiate your desired model.

```py
from esm.sdk import esmc_client

model = esmc_client(
    model="esmc-600m-2024-12", url="https://biohub.ai", token="<your API token>"
)

protein_tensor = model.encode(protein)
logits_output = model.logits(
  protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
)

print(logits_output.logits, logits_output.embeddings)
```

For tutorials on how to use ESMC, see our [tutorials](https://github.com/Biohub/esm/tree/main/cookbook/tutorials).

## ESMC Sparse Autoencoders (SAE)
<a name="esmc-sparse-autoencoders"></a>

Sparse autoencoders (SAE) are an unsupervised method for decomposing representations of large transformer language models into interpretable units. We released SAEs trained on ESMC to reveal the interpretable units of functional organization that ESMC's world model has learned.

The sparse autoencoder used in the Atlas and analyzed in the paper, `ESMC-6B-sae-layer60-k64-codebook16384`, is built on the ESMC 6B model. We also provide human-interpretable, agent-generated feature descriptions for this SAE's codebook.

Codebase, model weights, and model variants for ESMC SAEs are available through [Hugging Face](https://huggingface.co/collections/biohub/esmc-saes-for-hidden-states-all-layers).

```python
import torch
from transformers import AutoModel, AutoTokenizer

sequence = "MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASADGHRGPSAAFAPAAAEPKLFGGFNSSDTVTSPQRAGPLAGGVTTFVALYDYESRTETDLSFKKGERLQIVNNTEGDWWLAHSLSTGQTGYIPSNYVAPSDSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYCLSVSDFDNAKGLNVKHYKIRKLDSGGFYITSRTQFNSLQQLVAYYSKHADGLCHRLTTVCPTSKPQTQGLAKDAWEIPRESLRLEVKLGQGCFGEVWMGTWNGTTRVAIKTLKPGTMSPEAFLQEAQVMKKLRHEKLVQLYAVVSEEPIYIVTEYMSKGSLLDFLKGETGKYLRLPQLVDMAAQIASGMAYVERMNYVHRDLRAANILVGENLVCKVADFGLARLIEDNEYTARQGAKFPIKWTAPEAALYGRFTIKSDVWSFGILLTELTTKGRVPYPGMVNREVLDQVERGYRMPCPPECPESLHDLMCQCWRKEPEERPTFEYLQAFLEDYFTSTEPQYQPGENL"

model = AutoModel.from_pretrained("Biohub/ESMC-6B", device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("Biohub/ESMC-6B")
sae = AutoModel.from_pretrained(
    "Biohub/ESMC-6B-sae-k64-codebook16384",
    allow_patterns=["config.json", "layer_30.safetensors", "layer_60.safetensors"],
    device=model.device,
)
sae.initialize_layers([30, 60])
model.add_sae_models([sae.layers["30"], sae.layers["60"]])

inputs = tokenizer(sequence, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    output = model(**inputs)

output["sae_outputs"]["layer60"]  # sparse.coo tensor
print(output["sae_outputs"]["layer60"].shape)

```

For tutorials on how to use ESMC SAEs, see our [tutorials](https://github.com/Biohub/esm/tree/main/cookbook/tutorials).

## ESMFold2
<a name="esmfold2"></a>

[ESMFold2](https://huggingface.co/Biohub/ESMFold2) is a state-of-the-art protein structure prediction model that combines ESMC (6B parameter) language model embeddings with a diffusion-based structure prediction architecture.

The model predicts high-resolution, all-atom 3D protein structures directly from amino acid sequences, with optional multiple sequence alignment (MSA) input for enhanced accuracy on challenging targets. ESMFold2 achieves state-of-the-art performance matching or exceeding AlphaFold3 across diverse evaluation datasets, while offering improved computational efficiency through optimized diffusion sampling and architectural innovations.

Codebase, model weights, and model variants for ESMFold2 are available through [Hugging Face](https://huggingface.co/collections/biohub/esmfold2-model-family)

### Running ESMFold2 Locally

```python
from esm.models.esmfold2 import (
    DNAInput,
    ESMFold2InputBuilder,
    LigandInput,
    Modification,
    ProteinInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

HHAI_SEQ = (
    "MIEIKDKQLTGLRFIDLFAGLGGFRLALESCGAECVYSNEWDKYAQEVYEMNFGEKPEGDITQVNEKTIPDH"
    "DILCAGFPCQAFSISGKQKGFEDSRGTLFFDIARIVREKKPKVVFMENVKNFASHDNGNTLEVVKNTMNELD"
    "YSFHAKVLNALDYGIPQKRERIYMICFRNDLNIQNFQFPKPFELNTFVKDLLLPDSEVEHLVIDRKDLVMTN"
    "QEIEQTTPKTVRLGIVGKGGQGERIYSTRGIAITLSAYGGGIFAKTGGYLVNGKTRKLHPRECARVMGYPDS"
    "YKVHPSTSQAYKQFGNSVVINVLQYIAYNIGSSLNFKPY"
)

model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()

spi = StructurePredictionInput(
    sequences=[
        ProteinInput(id="A", sequence=HHAI_SEQ),
        DNAInput(
            id="B",
            sequence="GATAGCGCTATC",
            modifications=[Modification(position=5, ccd="C36")],
        ),
        DNAInput(
            id="C",
            sequence="TGATAGCGCTATC",
            modifications=[Modification(position=6, ccd="C36")],
        ),
        LigandInput(id="L", ccd=["SAH"]),
    ]
)

result = ESMFold2InputBuilder().fold(
    model, spi, num_loops=3, num_sampling_steps=50, num_diffusion_samples=1, seed=0
)

print(f"pLDDT mean: {float(result.plddt.mean()):.3f}, pTM: {float(result.ptm):.3f}, ipTM: {float(result.iptm):.3f}")

with open("1mht_pred.cif", "w") as f:
    f.write(result.complex.to_mmcif())
```

### Running ESMFold2 Through the Biohub Platform

Install the  `esm` Python package

```
pip install "esm @ git+https://github.com/Biohub/esm.git@c94ed8d763bbd7088b296949e5b401e8ea12073a"
```

Import the necessary libraries.

```py
from esm.sdk.forge import SequenceStructureForgeInferenceClient
from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig, LogitsOutput
```

Call the inference client with the selected model of choice and replace <your API token> with your token name.

```py
client = SequenceStructureForgeInferenceClient(model="esmfold2-fast-2026-05", url="https://biohub.ai", token="<your API token>")

gfp_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
gfp_input = input_builder.StructurePredictionInput(
    sequences=[gfp_sequence]
)

config = FoldingConfig(num_loops=3, num_sampling_steps=32, seed=0)
result = client.fold_all_atom(gfp_input, config=config)

with open("result.cif", "w") as f:
    f.write(result.complex.to_mmcif())
```

For tutorials on how to use ESMFold2, see our [tutorials](https://github.com/Biohub/esm/tree/main/cookbook/tutorials).


## Frontier Safety
<a name="frontier-safety"></a>

Biohub has established a safety team to assess the benefits and potential risks of our models and tools prior to release, and develop mitigations where necessary. To do this, we follow a structured approach that includes assessing both biosafety and biosecurity risks as well as existing, comparable open-source models and tools. We actively engage with the scientific community, stakeholders and domain experts to advance innovation as well as best practices for responsible development. Risk assessment was conducted for each of the components of this release, including our ESMC Cambrian models, ESMFold2, ESMC SAEs, ESM Atlas, and binder design system.

Informed by our risk assessments, we are releasing the source code and model weights for ESMC 6B, ESMFold2, and ESMC SAEs. We are also releasing our ESM Atlas dataset and binder design system openly. Biohub values open science, and we share our research with the scientific community so that others can evaluate, reproduce, and build upon our work.

Evaluations: Prior to release, we conducted evaluations to inform our understanding of capability uplift for specific misuse-relevant functional tasks. The full details of these evaluations are available in our corresponding paper appendix.

The Biohub Platform: We implement guardrails that detect and restrict the use of keywords and sequences corresponding to controlled pathogens and toxins on our freely accessible platform. For further details regarding these guardrails, please refer to our Biohub Platform Resources page. We recognize there are many legitimate reasons to use AI models to understand and model these sequences and proteins. If you are a researcher whose work is impacted by these guardrails, you can request elevated access to our platform via [Biohub.ai](http://Biohub.ai).

Please follow our [Acceptable Use Policy](https://biohub.org/acceptable-use-policy/) when using the model.

## Licenses
<a name="licenses"></a>

These models are available under the [MIT license](https://github.com/Biohub/esm/blob/main/LICENSE.md).

## Citations
<a name="citations"></a>

If you use ESM in your work, please cite one of the following:

#### ESMC, SAEs, and ESMFold2
```
@misc{candido2026language,
  title  = {Language Modeling Materializes a World Model of Protein Biology},
  author = {Candido, Salvatore and Hayes, Thomas and Derry, Alexander and Rao, Roshan
            and Lin, Zeming and Verkuil, Robert and Wu, Bryan and Lee, Jin Sub
            and Bruguera, Elise S. and Keval, Jehan A. and Kopylov, Mykhailo
            and Pak, John E. and Wu, Wesley and Thomas, Neil and Mataraso, Samson
            and Hsu, Alvin and Trotman-Grant, Ashton C. and Fatras, Kilian
            and dos Santos Costa, Allan and Badkundri, Rohil and Ak{\i}n, Halil
            and Oktay, Deniz and Deaton, Jonathan and Montabana, Elizabeth
            and Sitwala, Hrishita and Yu, Yue and Wiggert, Marius
            and Carlin, Dylan Alexander and Goering, Anthony W. and Blazejewski, Tomasz
            and Sandora, McCullen and Hla, Michael and Jia, Tina Z.
            and Kloker, Leon H. and Sofroniew, Nicholas J. and Uehara, Masatoshi
            and Pannu, Jassi and Bachas, Sharrol and Liu, Daniel S.
            and Sercu, Tom and Rives, Alexander},
  year   = {2026},
  url    = {https://biohub.ai/papers/esm_protein.pdf},
  note   = {Preprint}
}
```

#### ESM3
```
@article {hayes2024simulating,
	author = {Hayes, Thomas and Rao, Roshan and Akin, Halil and Sofroniew, Nicholas J. and Oktay, Deniz and Lin, Zeming and Verkuil, Robert and Tran, Vincent Q. and Deaton, Jonathan and Wiggert, Marius and Badkundri, Rohil and Shafkat, Irhum and Gong, Jun and Derry, Alexander and Molina, Raul S. and Thomas, Neil and Khan, Yousuf A. and Mishra, Chetan and Kim, Carolyn and Bartie, Liam J. and Nemeth, Matthew and Hsu, Patrick D. and Sercu, Tom and Candido, Salvatore and Rives, Alexander},
	title = {Simulating 500 million years of evolution with a language model},
	year = {2025},
	doi = {10.1126/science.ads0018},
	URL = {http://dx.doi.org/10.1126/science.ads0018},
	journal = {Science}
}
```

#### ESM Github (Code / Weights)

```
@software{evolutionaryscale_2024,
  author = {{EvolutionaryScale Team}},
  title = {evolutionaryscale/esm},
  year = {2024},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.14219303},
  URL = {https://doi.org/10.5281/zenodo.14219303}
}
```
