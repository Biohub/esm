# Maximum-accuracy folding with the SDK

Use `ForgeBatchClient.fold_max_accuracy` to predict a molecular complex with a 4096-token input limit through
Biohub's Batch API. The service generates an MSA, runs five folds with different
seeds, and selects the best predicted structure. The service uses ESMFold2 Pro trained on data with a 2025 cutoff, `esmfold2-2026-05-cutoff-2025`.

For general batch operations, see the [Batch API guide](batch.md).

## Inputs and MSA settings

Provide a `StructurePredictionInput` containing protein, RNA, DNA, or ligand
entries. The service searches UniRef30 and ColabFold to generate its own multiple sequence
alignment (MSA), using these search settings:

| Setting | Value |
| --- | --- |
| `max_prefilter` | 100,000 |
| `max_accept` | 10,000 |
| `diff` | 3,000 |
| `expand_diff` | 3,000 |

These settings are managed by the service. Leave `ProteinInput.msa` unset. Use `FoldMaxAccuracyConfig` for output options:

| Option | Requested output |
| --- | --- |
| `include_pae` | Predicted Aligned Error (PAE) matrix. |
| `include_pair_chains_iptm` | Pair-chain interface predicted TM scores. |
| `include_embeddings` | Pooled pair embeddings. |

## Submit and save the handle

This example submits two proteins, waits for both,
and saves each selected structure to an mmCIF file named by task ID. Each request
describes one complex; to fold multiple chains together, put them in the same
`StructurePredictionInput.sequences` list.

```python
import json
import os
from pathlib import Path

from esm.sdk.api import FoldMaxAccuracyConfig, FoldMaxAccuracyRequest
from esm.sdk.batch import BatchError
from esm.sdk.forge import ForgeBatchClient
from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput

sequences = [
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
]
fold_inputs = [
    StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])
    for sequence in sequences
]
config = FoldMaxAccuracyConfig(include_pae=True)

with ForgeBatchClient(token=os.environ["ESM_API_KEY"]) as client:
    submission = client.fold_max_accuracy.submit(
        requests=[FoldMaxAccuracyRequest(all_atom_input=item) for item in fold_inputs],
        config=config,
    )
    # Optional: save the submission handle for later retrieval.
    Path("fold-submission.json").write_text(json.dumps(submission.to_dict()))
    print("Batch ID:", submission.batch_id)

    client.fold_max_accuracy.wait(submission)

    for task_id, result in client.fold_max_accuracy.iter_results(submission):
        if isinstance(result, BatchError):
            print(task_id, result.error_msg)
            continue
        Path(f"{task_id}.cif").write_text(result.complex.to_mmcif())
        print(task_id, "pTM:", result.ptm)
        # The requested PAE matrix is available as result.pae.
```

The shared `config` applies to every request. To customize one
item, pass `config=FoldMaxAccuracyConfig(...)` to its `FoldMaxAccuracyRequest`.
An item's config replaces the shared config entirely; fields are not merged.

In an asyncio application, use the `async` counterpart methods as shown in the
[asyncio example](batch.md#using-python-asyncio).
