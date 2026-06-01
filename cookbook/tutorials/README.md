# **ESM Tutorial Notebooks**

Tutorial notebooks are the best way to get hands-on with ESM models\! Use the notebooks to explore model capabilities, learn workflows that can be applied to your own data, and learn how to interpret model outputs.

**ESMC**

ESMC is a protein language model that embeds sequences into rich numerical representations. Use it for analyzing, classifying, comparing, and interpreting proteins.

| Notebook | Colab  Notebook | Description |
| :---- | :---- | :---- |
| Embedding sequences with ESMC | `embed.ipynb`<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biohub/esm/blob/main/cookbook/tutorials/embed.ipynb) | Embed protein sequences and explore how different transformer layers encode structural and functional information. |
| Zero-shot entropy and mutation analysis | `esmc_mutation_scoring.ipynb`<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biohub/esm/blob/main/cookbook/tutorials/esmc_mutation_scoring.ipynb) | Compute per-position entropy and log-likelihood ratios to identify constrained vs. mutation-tolerant sites.  |
| Layer sweep for enzyme function classification | `esmc_layer_sweep.ipynb`<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biohub/esm/blob/main/cookbook/tutorials/esmc_layer_sweep.ipynb) | Learn how to sweep all layers to find which one is best using enzyme classification as a task. |
| Fine-tuning ESMC | `esmc_finetune.ipynb`<br> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biohub/esm/blob/main/cookbook/tutorials/esmc_finetune.ipynb) |  Fine-tune a classification or regression head for your dataset on top of ESMC using Parameter Efficient Fine-tuning (PEFT)  |

## **Interpretable features through Sparse Autoencoders (SAEs)**

| Notebook | Colab  Notebook | Description |
| :---- | :---- | :---- |
| Understanding proteins with SAE features |`esmc_sae_feature_interpretation.ipynb`<br> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biohub/esm/blob/main/cookbook/tutorials/esmc_sae_feature_interpretation.ipynb) |Extract and visualize sparse autoencoder features, rank by peak activation and prevalence, and map activations onto 3D structure. |

## **ESMFold2**

ESMFold2 predicts 3D protein structure from sequence, including DNA/RNA and small molecules.

| Notebook | Colab Notebook | Description |
| :---- | :---- | :---- |
| Folding with ESMFold2 | `esmfold2.ipynb`<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biohub/esm/blob/main/cookbook/tutorials/esmfold2.ipynb) | Fold proteins in combination with DNA, RNA and small-molecule ligands. |
| Binder design | `binder_design.ipynb`<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biohub/esm/blob/main/cookbook/tutorials/binder_design.ipynb) | Design antibodies and minibinders with high hit rates. Implements the protocol featured in our paper, which produced binders exhibiting nanomolar affinity, target specificity, and functional activity in laboratory assays. |

## **ESM3**

ESM3 is a generative model that reasons jointly over protein sequence, structure, and function. Use it for designing new proteins or editing existing ones.

| Notebook | Colab  Notebook | Description |
| :---- | :---- | :---- |
| Understanding the ESMProtein class  | `esmprotein.ipynb`<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biohub/esm/blob/main/cookbook/tutorials/esmprotein.ipynb) | Get familiar with how ESM3 represents proteins.  |
| Generating proteins with ESM3  | `esm3_generate.ipynb`<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biohub/esm/blob/main/cookbook/tutorials/esm3_generate.ipynb) | Learn how to scaffold a functional motif, edit secondary structure, and guide design using solvent exposure. |
| Designing a novel GFP with ESM3  | `gfp_design.ipynb`<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biohub/esm/blob/main/cookbook/tutorials/gfp_design.ipynb) | Walk through the exact prompting strategy used to design a novel fluorescent protein with no close natural relatives. |
| Guided generation with ESM3  | `esm3_guided_generation.ipynb`<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/biohub/esm/blob/main/cookbook/tutorials/esm3_guided_generation.ipynb) | Add scoring functions into the generation process, such as structural quality, sequence constraints, or other properties. |
