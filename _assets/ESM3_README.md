# ESM3 README

[ESM3](https://www.science.org/doi/10.1126/science.ads0018) is a frontier generative model for biology, able to jointly reason across three fundamental biological properties of proteins: sequence, structure, and function. These three data modalities are represented as tracks of discrete tokens at the input and output of ESM3. You can present the model with a combination of partial inputs across the tracks, and ESM3 will provide output predictions for all the tracks.

ESM3 is a *generative* masked language model. You can prompt it with partial sequence, structure, and function keywords, and iteratively sample masked positions until all positions are unmasked. This iterative sampling is what the `.generate()` function does.

<!--![ESM3 Diagram](_assets/esm3_diagram.png)-->
<img src="./esm3_diagram.png" alt="ESM3 Diagram" width="400" />

The ESM3 architecture is highly scalable due to its transformer backbone and all-to-all reasoning over discrete token sequences. At its largest scale, ESM3 was trained with 1.07e24 FLOPs on 2.78 billion proteins and 771 billion unique tokens, and has 98 billion parameters.

Learn more by reading the paper [(Hayes et al., 2024)](https://www.science.org/doi/10.1126/science.ads0018).

## [ESM3 Family](https://huggingface.co/collections/biohub/esm3-model-family)

The code for ESM3 is available from Github and weights for esm3-sm-open-v1 is available on [Hugging Face](https://huggingface.co/collections/biohub/esm3-model-family). Other weights are available when the model is accessed through Biohub.

| Model | Model Size | Release Date | Note |
| :---- | :---- | :---- | :---- |
| **Flagship Models** |  |  | Most users will be interested in using one of these models. |
| esm3-large-2024-03 | 98B | 2024-03 |  |
| esm3-medium-2024-08 | 7B | 2024-08 |  |
| esm3-small-2024-08 | 1.4B | 2024-08 |  |
| **Published Models** |  |  | These models were used to generate all of the results in the ESM3 paper and are provided to facilitate reproducibility. |
| esm3-large-2024-03 | 98B | 2024-03 |  |
| esm3-medium-2024-03 | 7B | 2024-03 |  |
| esm3-small-2024-03 | 1.4B | 2024-03 |  |

## Quickstart for ESM3
<a name="quickstart-esm3"></a>

### Running ESM3 Through Biohub

First install the python library using `pip`:

```
pip install esm@git+https://github.com/Biohub/esm.git@main
```

Then import the necessary libraries and instantiate your model. Use your token from the [Biohub platform](https://biohub.ai")

```py
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk import client
from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig, LogitsOutput

model: ESM3InferenceClient = esm.sdk.client("esm3-medium-2024-08", token="<your API token>")
```

### Running ESM3 Locally

The following code demonstrates how to run ESM3 locally and generate a simple sequence prompt. The weights are stored on [Hugging Face](https://huggingface.co/biohub/esm3-sm-open-v1).

First install the python library using `pip`:

```
pip install esm@git+https://github.com/Biohub/esm.git@main
```

Then import the necessary libraries for your model.

```py
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Will instruct you how to get an API key from huggingface hub, make one with "Read" permission.
login()

# This will download the model weights and instantiate the model on your machine.
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-sm-open-v1").to("cuda") # or "cpu"

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

## Tutorials for ESM3
<a name="tutorials-esm3"></a>

For tutorials on how to use ESM3, see our Tutorials [here](https://github.com/Biohub/esm/tree/main/cookbook/tutorials).


## Responsible Development
<a name="responsible-development"></a>

Biohub has established a safety team to assess the benefits and potential risks of our models and tools prior to release, and develop mitigations where necessary. To do this, we follow a structured approach that includes assessing both biosafety and biosecurity risks as well as existing, comparable open-source models and tools. We actively engage with the scientific community, stakeholders and domain experts to advance innovation as well as best practices for responsible development. Risk assessment was conducted for ESM3.

Please follow our [Acceptable Use Policy](https://biohub.org/acceptable-use-policy/) when using the model.

## Licenses
<a name="licenses"></a>

These models are available under the [MIT license](https://github.com/Biohub/esm/blob/main/LICENSE.md).

## Citations
<a name="citations"></a>

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
