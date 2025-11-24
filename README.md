# ESM

<div align="center">
  <img src="_assets/logo.png" width="50"/>

[ESM3](https://www.science.org/doi/10.1126/science.ads0018) &sdot; [ESM C](https://www.evolutionaryscale.ai/blog/esm-cambrian) &sdot;
[Slack](https://bit.ly/3FKwcWd) &sdot; [Tutorials](https://github.com/evolutionaryscale/esm/tree/main/cookbook/tutorials) <br>
</div>


- [Installation ](#installation-)
- [Available Models](#available-models-)
- [ESM 3](#esm-3-)
  - [Quickstart for ESM3 Open](#esm3-quickstart-)
  - [ESM3 98B via Forge API](#esm3-forge)
  - [ESM3 Example Usage](#esm3-example-usage)
- [ESM C](#esm-c-)
  - [Quickstart for ESM C Open Models](#esm-c-open-)
  - [ESM C 6B via Forge API](#esm-c-forge-)
  - [ESM C via SageMaker for Commercial Use  ](#esm-c-sagemaker-)
  - [ESM C Example Usage](#esm-c-example-)
- [Responsible Development](#responsible-development-)
- [Licenses](#licenses-)
- [Citations  ](#citations-)


This repository contains flagship protein models for EvolutionaryScale, as well as access to the API. [ESM3](https://www.evolutionaryscale.ai/papers/esm3-simulating-500-million-years-of-evolution-with-a-language-model) is our flagship multimodal protein generative model, and can be used for generation and prediction tasks. [ESM C](https://www.evolutionaryscale.ai/blog/esm-cambrian) is our best protein representation learning model, and can be used to embed protein sequences.

## Installation <a name="installation"></a>

To get started with ESM, install the python library using pip:

```bash
pip install esm
```

## Installation instructions for local development
After cloning the repo, use pixi's official installation site (https://pixi.sh/dev/installation/) to install pixi package manager.
In the root of the esm repo, run `pixi install`. This installs the `default` environment.

To install the development environment which includes some additional dependencies, run `pixi install --environment dev`.
To run tests locally, run `pixi run -e dev cov-test`.
To run pre-commit checks on all files, run `pixi run -e dev lint-all`.

To run the `kneedle.py` script, create a `.env` file in the root directory following the structure of `.env.example` (located in the root directory) with your Hugging Face API key. This allows you to download the ESM3 model weights locally. Add the `.fasta` file on which kneedle should be run to the `data` folder in the repo's root directory.


## Available Models <a name="available-models"></a>

### ESM 3 Family

| Model | Model Size | Release Date | Note |
|-------|------------|--------------|------|
| **Flagship Models** | | | Most users will be interested in using one of these models. |
| esm3-large-2024-03 | 98B | 2024-03 | |
| esm3-medium-2024-08 | 7B | 2024-08 | |
| esm3-small-2024-08 | 1.4B | 2024-08 | |
| **Published Models** | | | These models were used to generate all of the results in the ESM3 paper and are provided to facilitate reproducibility. |
| esm3-large-2024-03 | 98B | 2024-03 | |
| esm3-medium-2024-03 | 7B | 2024-03 | |
| esm3-small-2024-03 | 1.4B | 2024-03 | |
| **Experimental Models** | | | These models are provided for early use by researchers and are still under development. |
| esm3-medium-multimer-2024-09 | 7B | 2024-09 | |

### ESM C Models

| Model | Model Size | Number of Layers | Release Date |
|-------|------------|------------------|--------------|
| esmc-6b-2024-12 | 6B | 80 | 2024-12 |
| esmc-600m-2024-12 | 600M | 36 | 2024-12 |
| esmc-300m-2024-12 | 300M | 30 | 2024-12 |

## ESM 3  <a name="esm3"></a>

[ESM3](https://www.evolutionaryscale.ai/papers/esm3-simulating-500-million-years-of-evolution-with-a-language-model) is a frontier generative model for biology, able to jointly reason across three fundamental biological properties of proteins: sequence, structure, and function. These three data modalities are represented as tracks of discrete tokens at the input and output of ESM3. You can present the model with a combination of partial inputs across the tracks, and ESM3 will provide output predictions for all the tracks.

ESM3 is a _generative_ masked language model. You can prompt it with partial sequence, structure, and function keywords, and iteratively sample masked positions until all positions are unmasked. This iterative sampling is what the `.generate()` function does.

<!--![ESM3 Diagram](_assets/esm3_diagram.png)-->
<img src="_assets/esm3_diagram.png" alt="ESM3 Diagram" width="400" />

The ESM3 architecture is highly scalable due to its transformer backbone and all-to-all reasoning over discrete token sequences. At its largest scale, ESM3 was trained with 1.07e24 FLOPs on 2.78 billion proteins and 771 billion unique tokens, and has 98 billion parameters.
Learn more by reading the [blog post](https://www.evolutionaryscale.ai/blog/esm3-release) and [the paper (Hayes et al., 2024)](https://www.science.org/doi/10.1126/science.ads0018).

ESM3-open, with 1.4B parameters, is the smallest and fastest model in the family.

### Quickstart for ESM3-open <a name="esm3-quickstart"></a>

The weights are stored on HuggingFace Hub under [HuggingFace/EvolutionaryScale/esm3](https://huggingface.co/EvolutionaryScale/esm3).

```py
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Will instruct you how to get an API key from huggingface hub, make one with "Read" permission.
login()

# This will download the model weights and instantiate the model on your machine.
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda") # or "cpu"

# Generate a completion for a partial Carbonic Anhydrase (2vvb)
prompt = "___________________________________________________DQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPP___________________________________________________________"
protein = ESMProtein(sequence=prompt)
# Generate the sequence, then the structure. This will iteratively unmask the sequence track.
protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))
# We can show the predicted structure for the generated sequence.
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
protein.to_pdb("./generation.pdb")
# Then we can do a round trip design by inverse folding the sequence and recomputing the structure
protein.sequence = None
protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8))
protein.coordinates = None
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
protein.to_pdb("./round_tripped.pdb")
```

Congratulations, you just generated your first proteins with ESM3!

### EvolutionaryScale Forge: Access to larger ESM3 models
 <a name="esm3-forge"></a>

You can access all scales of ESM3 models [EvolutionaryScale Forge](https://forge.evolutionaryscale.ai).

We encourage users to interact with the Forge API through the python `esm` library instead of the command line.
The python interface enables you to interactively load proteins, build prompts, and inspect generated proteins
with the `ESMProtein` and config classes used to interact with the local model.

In any example script you can replace a local `ESM3` model with a Forge API client:

```py
# Instead of loading the model locally on your machine:
model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda") # or "cpu"
# just replace the line with this:
model: ESM3InferenceClient = esm.sdk.client("esm3-medium-2024-08", token="<your forge token>")
# and now you're interfacing with the model running on our remote servers.
...
```

and the exact same code will work.
This enables a seamless transition from smaller and faster models, to our largest and most capable protein language models for protein design work.

## Downloading the data
In order to download the same training data that ESM uses, we must download UniRef (2023_02), MGnify90 (2023_02), and all non-restricted JGI studies available as of July 31 2023.

UniProt contains all of the major releases. We can acquire this specific snapshot dataset by running:
```bash
wget https://ftp.uniprot.org/pub/databases/uniprot/previous_major_releases/release-2023_02/uniref/uniref2023_02.tar.gz
```
The 2023 release for MGnify Proteins (MGnify 90) is found [here](https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2023_02) and you can run:
```bash
BASE="https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/2023_02"
wget -c "${BASE}/mgy_clusters.fa.gz"
```

JGI is a little more complicated because we need to first create an account and then specify the proper portals. From the JGI search UI you can export a report that includes the “Portal ID” column, which this script consumes. JGI’s help explains where to grab those IDs and shows the exact curl login + “get-directory” calls. Make sure you save your username and password in a .env file. You can do so like this:
```bash
JGI_USER=xxxxxx
JGI_PASS=xxxxxx

source .env
```
Make sure to add the .env to the .gitignore so it isn't public!

First we need to grab all public portals from 2023. You can get them from this [link](https://genome.jgi.doe.gov/portal/). Once you are here, go to advanced search, click "JGI portals" from "search within" and then "show all". This will allow you to download a csv that contains all of the project IDs alongside other metadata. Once you have this you can run these commands:
```bash
curl 'https://signon.jgi.doe.gov/signon/create' \
--data-urlencode "login=${JGI_USER}" \
--data-urlencode "password=${JGI_PASS}" \
-c cookies > /dev/null

pip install csvkit

csvcut -c "Portal ID" genome_project.csv > portals.txt

at portals.txt | grep -oE '[A-Za-z0-9_]+_FD' | sort -u > portals_clean.txt
```

Now you have the clean list of portal IDs.

Once you have a clean list of portal IDs you can run my download script:
```bash
./download_jgi.sh portals_clean.txt dir/you/want/to/save/to
```



## Licenses  <a name="licenses"></a>

The code and model weights of ESM3 and ESM C are available under a mixture of non-commercial and permissive commercial licenses. For complete license details, see [LICENSE.md](./LICENSE.md).

## Citations  <a name="citations"></a>
If you use ESM in your work, please cite one of the following:

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

#### ESM C
```
@misc{esm2024cambrian,
  author = {{ESM Team}},
  title = {ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning},
  year = {2024},
  publisher = {EvolutionaryScale Website},
  url = {https://evolutionaryscale.ai/blog/esm-cambrian},
  urldate = {2024-12-04}
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
