# Batch API

Use the Biohub Batch API to process larger or longer-running workloads in the background. Some models/services may only be available over the Batch API, not Sync API. The Batch API offers inference at discounted rates, and should be used when immediate responses are not required.

`ForgeBatchClient` lets you submit many tasks, wait for completion, and retrieve decoded results. This guide covers submission, progress checks, result retrieval, expiry, cache control, cancellation, and Python asyncio usage.

## Supported services

| Service | Description | Guide |
| --- | --- | --- |
| `fold` | Protein structure prediction with ESMFold2 | Examples below |
| `fold_all_atom` | Molecular-complex prediction with ESMFold2 | [Input guide](../../README.md#running-esmfold2-through-the-biohub-platform) |
| `fold_max_accuracy` | Maximum-accuracy molecular-complex prediction | [SDK guide](fold_max_accuracy.md) |

Every fold service accepts inputs up to 4096 tokens, against 768 over the Sync API.

We will be launching more models/services soon!

## Submit, wait, and retrieve results

```python
import os

from esm.sdk.api import FoldRequest
from esm.sdk.forge import ForgeBatchClient

sequences = ["MKTAYIAKQRQISFVKSHFSRQ", "GSHMASMTGGQQMGRGS"]

with ForgeBatchClient(
    token=os.environ["ESM_API_KEY"], model="esmfold2-fast-2026-05"
) as client:
    submission = client.fold.submit(
        requests=[FoldRequest(sequence=sequence) for sequence in sequences]
    )
    print("Batch ID:", submission.batch_id)
    summary = client.fold.wait(submission)
    outputs = client.fold.results(submission)
```


`submit()` returns a `BatchSubmission` containing the batch ID. Pass that submission to `wait()` to poll until the tasks finish, then to `results()` to download and decode their outputs.

`fold.results()` returns one `ESMProtein` or `BatchError` per listed task, in server listing order. A failed item does not discard
other results. Inspect `error.task_status` to distinguish the outcome:

| `task_status` | Meaning |
| --- | --- |
| `"failed"` | Task execution failed. |
| `"cancelled"` | The task was cancelled, including expiry before it started. `error_msg` includes the server's reason when provided. |
| `"queued"` or `"in_progress"` | The result is not ready; wait and retrieve it later. |
| `None` | The error came from a request, decoding, or waiting rather than a task state. |


`wait()` periodically polls `/batch/list` until batch processing finishes, including failed or cancelled tasks. It raises an error if waiting times out. It returns the batch summary while `results()` retrieves the outputs.
`results()` fetches every page automatically and collects the outputs in a list.
Use `page_size` to control how many tasks are listed per request (default 100,
maximum 1000).

A failed task appears as a `BatchError` in the list alongside successful results.
If the batch cannot be fully retrieved, `results()` raises a `BatchError`; outputs
already retrieved are available in `error.partial_results` as `(task_id, output)`
pairs.

### Submission errors and retries

The SDK sends each submission once. A timeout or lost response may occur after
the server accepts the work, so submitting again can create a duplicate batch.
Automatic retries apply to status reads, batch listing, and result downloads.

`submit()` raises `BatchError` on failure; `run()` returns it. If the server
response included usable job IDs, `error.submission` retains them. Otherwise the
error has no recovery handle. Save the handle returned by a successful submit
before waiting for results.

### Optional: save a submission for later

To continue later, save the submission and restore it:

```python
import json
from pathlib import Path

from esm.sdk.batch import BatchSubmission

Path("fold-job.json").write_text(json.dumps(submission.to_dict()))
restored = BatchSubmission.from_dict(json.loads(Path("fold-job.json").read_text()))
```

Pass `restored` to `wait()` and `results()` on a new client authenticated as the
same user and pointed at the same service.

If you only kept the batch ID, build the submission directly. `endpoint` must
match the service you call.

```python
from esm.sdk.batch import BatchSubmission

submission = BatchSubmission(endpoint="fold", batch_id="<batch id>")
summary = client.fold.wait(submission)
outputs = client.fold.results(submission)
```

## Queue expiry and result retention

Both deadlines are measured from when the batch is submitted:

| Deadline | Behavior |
| --- | --- |
| 24 hours (`expires_at`) | Tasks that have not started are cancelled without charge. Tasks already running finish normally. |
| 31 days (`results_expire_at`) | Results are no longer available. Download and save outputs before this deadline. |

Fetching an expired result produces a `BatchError` with `error_code=410` and an expiry explanation in `error_msg`.

## Bypass cached results

By default, submissions allow cached inference results (`disable_cache=False`).
To bypass the cache, pass `disable_cache=True` to `submit()` or `run()`:

```python
submission = client.fold.submit(
    requests=[FoldRequest(sequence=sequence) for sequence in sequences],
    disable_cache=True,
)
```

## Process results incrementally

For large batches that consume too much memory in aggregate, use `iter_results()` in place of `results()` to process outputs
as they are retrieved. It follows pagination, downloads and decodes stored outputs,
and yields `(task_id, result)` pairs in server listing order. `results()` uses the
same order.

```python
from esm.sdk.batch import BatchError

for task_id, result in client.fold.iter_results(
    submission, page_size=100, max_concurrency=4
):
    if isinstance(result, BatchError):
        print(task_id, result.error_msg)
        continue
    result.to_pdb(f"{task_id}.pdb")
```

`max_concurrency` bounds both simultaneous downloads and prefetched decoded outputs. The next page is fetched after the current page
is consumed.

Iteration does not wait for unfinished tasks; call `wait()` first when you want all tasks to finish.

## List a batch

Use `list_batch()` for progress inspection, status filters, manual paging, or
selecting outputs to download. Use `iter_results()` to download and decode outputs
incrementally, or `results()` to collect every output in the submission.

```python
summary = client.list_batch(submission.batch_id)
page = client.list_batch(
    submission.batch_id,
    status=["done", "failed"],
    include_responses=True,
    limit=100,
)
while True:
    for item in page["items"]:
        print(item["task_id"], item["status"])
    if page["cursor"] is None:
        break
    page = client.list_batch(
        submission.batch_id,
        status=["done", "failed"],
        include_responses=True,
        limit=100,
        cursor=page["cursor"],
    )
```

## Run the whole lifecycle

`run()` owns the whole pipeline: it submits the work, waits for completion, and
downloads and decodes the results in one call.

```python
from esm.sdk.batch import BatchError

outputs = client.fold.run(
    requests=[FoldRequest(sequence=sequence) for sequence in sequences],
)
if isinstance(outputs, BatchError):
    print(outputs.error_msg)
```

On timeout or interruption, `wait()` stops waiting and leaves work running.
`run()` attempts to cancel queued work in both cases by default
(`cancel_on_timeout=True`). Set `cancel_on_timeout=False` to leave work running.

Manual cancellation uses exactly one kind of ID:

```python
client.cancel(task_id=task_id)  # An actual ID returned by list_batch() or iter_results().
client.cancel(batch_id=submission.batch_id)
```

## Using Python asyncio

Both client interfaces submit jobs for background processing. The examples above
use regular Python methods. In an asyncio application, use the `async_*` methods
to avoid blocking the event loop while making requests or polling for completion.

```python
async with ForgeBatchClient(
    token=os.environ["ESM_API_KEY"], model="esmfold2-fast-2026-05"
) as client:
    submission = await client.fold.async_submit(
        requests=[FoldRequest(sequence=sequence) for sequence in sequences],
    )
    await client.fold.async_wait(submission)
    outputs = await client.fold.async_results(submission)
```
