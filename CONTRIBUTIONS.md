We welcome community contributions to help make this package better!

## Contributing

_to be written_

## Testing

Improving test coverage and automating the process is high priority but not done yet. For now, to test:

0. Ensure you are testing in a clean environment with Python 3.12 or newer,
as required by `pyproject.toml`. For example, using micromamba:

```bash
micromamba create -n esm -c conda-forge python=3.12 pip
micromamba activate esm

# in root level of repo
python -m pip install -e .
python -c 'from huggingface_hub import login; login()'


```

```bash

# Ensure package can correctly interact with forge.
# This will require an API key from forge.evolutionaryscale.ai
ESM_API_KEY=$ESM_API_KEY PYTHONPATH='.' python cookbook/snippets/esm3.py


```

1. Ensure the following scripts and notebooks run without errors. Comment out
notebook cells that install the published `esm` package so the candidate checkout
is tested. Hosted notebooks prompt for an API token; run those interactively
unless authenticated inputs have been configured for the `treon` run.

```bash
ESM_API_KEY=$ESM_API_KEY PYTHONPATH='.' python cookbook/snippets/esm3.py
python -m pip install treon
treon cookbook/tutorials/esmprotein.ipynb
treon cookbook/tutorials/embed.ipynb
treon cookbook/tutorials/gfp_design.ipynb
treon cookbook/tutorials/esm3_generate.ipynb

# requires a GPU
python cookbook/snippets/esm3.py
python cookbook/snippets/esmc.py
python cookbook/local/raw_forwards.py


```

Setting `ESM_API_KEY` alone does not fill a notebook's `getpass` prompt.
See [`cookbook/tutorials/README.md`](cookbook/tutorials/README.md) for the
current notebook descriptions and their setup instructions.
