"""Public batch lifecycle contracts against a mocked HTTP server."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio

from esm.sdk import base_forge_client
from esm.sdk.api import (
    ESMProtein,
    ESMProteinError,
    FoldAllAtomRequest,
    FoldMaxAccuracyConfig,
    FoldMaxAccuracyRequest,
    FoldRequest,
)
from esm.sdk.batch import BatchError, BatchSubmission, BatchTimeoutError
from esm.sdk.forge import ForgeBatchClient
from esm.utils.constants.models import ESMFOLD2_CUTOFF_2025
from esm.utils.residue_constants import restype_1to3
from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput
from esm.utils.structure.molecular_complex import MolecularComplexResult


def _input(sequence: str) -> StructurePredictionInput:
    return StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=sequence)])


def _fold_result(sequence: str = "ACDEF") -> dict[str, Any]:
    return {
        "complex": {
            "id": sequence,
            "sequence": [restype_1to3[residue] for residue in sequence],
            "atom_positions": [[0.0, 0.0, 0.0] for _ in sequence],
            "atom_elements": ["C" for _ in sequence],
            "token_to_atoms": [[index, index + 1] for index in range(len(sequence))],
            "chain_id": [0 for _ in sequence],
            "plddt": [90.0 for _ in sequence],
            "metadata": {"entity_lookup": {1: "protein"}, "chain_lookup": {0: "A"}},
        }
    }


class _BatchAPI:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.replies: list[httpx.Response | Exception] = []
        self.clients: list[ForgeBatchClient] = []

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def client(self, model: str | None = None) -> ForgeBatchClient:
        client = ForgeBatchClient(
            token="test-token",
            model=model,
            min_retry_wait=0,
            max_retry_wait=0,
            max_retry_attempts=3,
        )
        transport = httpx.MockTransport(self.respond)
        client._batch_client._client = httpx.Client(transport=transport)
        client._batch_client._async_client = httpx.AsyncClient(transport=transport)
        self.clients.append(client)
        return client

    def data(self, value: dict[str, Any]) -> None:
        self.replies.append(
            httpx.Response(200, json={"status": "success", "data": value})
        )

    def accepted(self, count: int = 1) -> None:
        reply: dict[str, Any] = {"batch_id": "batch-A", "total_items": count}
        if count == 1:
            reply["task_id"] = "task-1"
        self.data(reply)

    def completed(self, task_id: str = "task-1") -> None:
        self.data({"status": "done", "response": f"https://results.example/{task_id}"})

    def download(self, sequence: str = "ACDEF") -> None:
        self.replies.append(httpx.Response(200, json=_fold_result(sequence)))

    def page(
        self,
        items: list[dict[str, Any]],
        cursor: str | None = None,
        total: int | None = None,
    ) -> None:
        self.data(
            {
                "status": "done",
                "items": items,
                "cursor": cursor,
                "total": len(items) if total is None else total,
            }
        )


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest_asyncio.fixture
async def api() -> AsyncIterator[_BatchAPI]:
    api = _BatchAPI()
    yield api
    for client in api.clients:
        client._batch_client.client.close()
        await client._batch_client.async_client.aclose()


async def _call(target: Any, asynchronous: bool, method: str, *args, **kwargs) -> Any:
    if asynchronous:
        return await getattr(target, f"async_{method}")(*args, **kwargs)
    return getattr(target, method)(*args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "count,request_shape,disable_cache",
    [(1, "single", False), (3, "list", True), (2, "positional_generator", False)],
)
async def test_submit_retains_only_explicit_ids(
    api, asynchronous, count, request_shape, disable_cache
):
    client = api.client(model="unrelated-client-model")
    sequences = ["ACDEF", "GHIKL", "MNPQR"][:count]
    api.accepted(count)
    requests = (
        FoldMaxAccuracyRequest(all_atom_input=_input(sequence))
        for sequence in sequences
    )
    kwargs = (
        {"requests": list(requests) if request_shape == "list" else requests}
        if request_shape != "single"
        else {"all_atom_input": _input(sequences[0])}
    )
    args = (kwargs.pop("requests"),) if request_shape == "positional_generator" else ()

    submission = await _call(
        client.fold_max_accuracy,
        asynchronous,
        "submit",
        *args,
        disable_cache=disable_cache,
        **kwargs,
    )

    assert submission.batch_id == "batch-A"
    saved = json.loads(json.dumps(submission.to_dict()))
    assert saved == {
        "endpoint": "fold_max_accuracy",
        "batch_id": "batch-A",
        "task_ids": list(submission.task_ids),
    }
    assert BatchSubmission.from_dict(saved) == submission
    assert submission.task_ids == (("task-1",) if count == 1 else ())
    assert len(api.requests) == 1
    assert api.requests[0].headers.get("X-Disable-Cache") == (
        "true" if disable_cache else None
    )
    request = json.loads(api.requests[0].content)
    assert request["endpoint"] == "fold_max_accuracy"
    assert set(request) == {"endpoint", "payload"}
    assert [
        payload["all_atom_input"]["sequences"][0]["sequence"]
        for payload in request["payload"]
    ] == sequences
    assert all(
        payload["model"] == ESMFOLD2_CUTOFF_2025 for payload in request["payload"]
    )
    if count == 1:
        assert submission.task_id == "task-1"
    else:
        with pytest.raises(ValueError, match="exactly one known task ID"):
            _ = submission.task_id


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_fold_max_accuracy_typed_requests_resolve_configs_independently(
    api, asynchronous
):
    client = api.client()
    item_config = FoldMaxAccuracyConfig(include_embeddings=True)
    requests = [
        FoldMaxAccuracyRequest(all_atom_input=_input("ACDEF")),
        FoldMaxAccuracyRequest(all_atom_input=_input("GHIKL"), config=item_config),
    ]
    config = FoldMaxAccuracyConfig(include_pae=True)
    api.accepted(len(requests))
    api.data({"status": "done", "counts": {"done": 2}, "total": 2})
    api.page(
        [
            {"task_id": str(index), "status": "done", "response": _fold_result()}
            for index in range(len(requests))
        ]
    )

    await _call(
        client.fold_max_accuracy,
        asynchronous,
        "run",
        requests=iter(requests),
        config=config,
    )

    payload = json.loads(api.requests[0].content)["payload"]
    assert [(item["include_pae"], item["include_embeddings"]) for item in payload] == [
        (True, False),
        (False, True),
    ]
    assert requests[0].config is None
    assert requests[1].config is item_config
    assert not api.replies


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "invalid_input", ["empty_generator", "ambiguous", "model_override", "dict"]
)
async def test_fold_max_accuracy_rejects_invalid_request_inputs_before_http(
    api, asynchronous, invalid_input
):
    client = api.client()
    typed = [FoldMaxAccuracyRequest(all_atom_input=_input("GHIKL"))]
    kwargs: dict[str, Any] = {
        "empty_generator": {
            "requests": (FoldMaxAccuracyRequest(all_atom_input=item) for item in [])
        },
        "ambiguous": {"all_atom_input": _input("ACDEF"), "requests": typed},
        "model_override": {"model_name": "esmfold2", "requests": typed},
        "dict": {"requests": [{"all_atom_input": _input("ACDEF")}]},
    }[invalid_input]

    expected_error = ValueError if invalid_input == "empty_generator" else TypeError
    expected_message = (
        "fold_max_accuracy selects its own model"
        if invalid_input == "model_override"
        else None
    )
    with pytest.raises(expected_error, match=expected_message):
        await _call(client.fold_max_accuracy, asynchronous, "run", **kwargs)

    assert not api.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_run_accepts_single_task_response(api, asynchronous):
    client = api.client()
    api.data({"message": "Batch request submitted", "task_id": "legacy-task"})
    api.completed("legacy-task")
    api.completed("legacy-task")
    api.download()

    result = await _call(
        client.fold_all_atom,
        asynchronous,
        "run",
        all_atom_input=_input("ACDEF"),
        model_name="esmfold2-test",
    )

    assert isinstance(result, MolecularComplexResult)
    assert [request.url.path for request in api.requests] == [
        "/api/v1/batch/submit",
        "/api/v1/batch/status",
        "/api/v1/batch/status",
        "/legacy-task",
    ]
    assert not api.replies


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "model",
    ["esmfold2-2026-05", "esmfold2-fast-2026-05", "esmfold2-2026-05-cutoff-2025"],
)
@pytest.mark.parametrize("endpoint", ["fold", "fold_all_atom"])
async def test_public_folding_submit_and_decode(api, asynchronous, model, endpoint):
    client = api.client(model=model)
    sequences = ["ACDEF", "GHIKL"]
    api.accepted(count=2)
    api.data({"status": "done", "counts": {"done": 2}, "total": 2})
    api.page(
        [
            {
                "task_id": f"batch-A-{index:05d}",
                "status": "done",
                "response_available": True,
            }
            for index in range(2)
        ]
    )
    for index, sequence in enumerate(sequences):
        api.completed(f"batch-A-{index:05d}")
        response = (
            {
                "sequence": sequence,
                "coordinates": [[[0.0, 0.0, 0.0]] * 37] * len(sequence),
            }
            if endpoint == "fold"
            else _fold_result(sequence)
        )
        api.replies.append(httpx.Response(200, json=response))
    requests = [
        FoldRequest(sequence=sequence)
        if endpoint == "fold"
        else FoldAllAtomRequest(all_atom_input=_input(sequence))
        for sequence in sequences
    ]
    results = await _call(
        getattr(client, endpoint),
        asynchronous,
        "run",
        requests=requests,
        max_concurrency=1,
    )
    assert isinstance(results, list)
    if endpoint == "fold":
        assert all(isinstance(result, ESMProtein) for result in results)
        assert [result.sequence for result in results] == sequences
        assert all(result.coordinates.shape == (5, 37, 3) for result in results)
    else:
        assert all(isinstance(result, MolecularComplexResult) for result in results)
        assert [result.complex.id for result in results] == sequences
    submitted = json.loads(api.requests[0].content)
    assert submitted["endpoint"] == endpoint
    assert [item["model"] for item in submitted["payload"]] == [model, model]
    assert (
        len(
            [
                request
                for request in api.requests
                if request.url.path.endswith("/submit")
            ]
        )
        == 1
    )
    assert not api.replies


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("method", ["submit", "run"])
@pytest.mark.parametrize(
    "failure, error_code",
    [
        (lambda: httpx.ReadTimeout("Server accepted the request, response lost"), 500),
        (lambda: httpx.Response(503, json={"message": "Unavailable"}), 503),
        (lambda: httpx.Response(200, content=b"invalid JSON"), 400),
    ],
    ids=["lost_response", "unavailable", "invalid_json"],
)
async def test_submission_failures_are_not_resent(
    api, asynchronous, method, failure, error_code
):
    client = api.client()
    api.replies.append(failure())
    api.accepted()
    options = {"all_atom_input": _input("ACDEF")}

    if method == "submit":
        with pytest.raises(BatchError) as raised:
            await _call(client.fold_max_accuracy, asynchronous, method, **options)
        error = raised.value
    else:
        error = await _call(client.fold_max_accuracy, asynchronous, method, **options)

    assert isinstance(error, BatchError)
    assert error.error_code == error_code
    assert error.submission is None
    assert error.task_status is None
    assert len(api.requests) == 1
    assert len(api.replies) == 1
    assert api.requests[0].url.path == "/api/v1/batch/submit"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "request_shape", ["single", "list_one", "positional_generator_many"]
)
async def test_run_composes_submit_wait_and_ordered_typed_results(
    api, asynchronous, request_shape
):
    client = api.client()
    sequences = ["ACDEF", "GHIKL"] if request_shape.endswith("many") else ["ACDEF"]
    api.accepted(len(sequences))
    task_ids = ["opaque-z", "opaque-a"][: len(sequences)]
    api.data(
        {"status": "done", "counts": {"done": len(sequences)}, "total": len(sequences)}
    )
    api.page(
        [
            {"task_id": task_id, "status": "done", "response": _fold_result(sequence)}
            for task_id, sequence in reversed(list(zip(task_ids, sequences)))
        ]
    )
    requests = (
        FoldMaxAccuracyRequest(all_atom_input=_input(sequence))
        for sequence in sequences
    )
    kwargs = (
        {"all_atom_input": _input(sequences[0])}
        if request_shape == "single"
        else {
            "requests": list(requests) if request_shape.startswith("list") else requests
        }
    )

    args = (
        (kwargs.pop("requests"),)
        if request_shape == "positional_generator_many"
        else ()
    )
    output = await _call(
        client.fold_max_accuracy,
        asynchronous,
        "run",
        *args,
        disable_cache=True,
        page_size=2,
        max_concurrency=1,
        **kwargs,
    )

    if request_shape == "single":
        assert isinstance(output, MolecularComplexResult)
        results = [output]
    else:
        assert isinstance(output, list)
        results = output
    assert [result.complex.id for result in results] == list(reversed(sequences))
    assert api.requests[0].headers.get("X-Disable-Cache") == "true"
    assert [request.url.path for request in api.requests] == [
        "/api/v1/batch/submit",
        "/api/v1/batch/list",
        "/api/v1/batch/list",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_timeout_without_cancellation_keeps_a_recoverable_handle(
    api, asynchronous
):
    client = api.client()
    api.accepted()
    api.data({"status": "in_progress", "counts": {"queued": 1}, "total": 1})

    timeout = await _call(
        client.fold_max_accuracy,
        asynchronous,
        "run",
        all_atom_input=_input("ACDEF"),
        timeout=0,
        cancel_on_timeout=False,
    )

    assert isinstance(timeout, BatchTimeoutError)
    assert timeout.error_code == 408
    assert timeout.task_status is None
    assert timeout.batch_id == "batch-A"
    assert timeout.task_ids == ("task-1",)
    assert timeout.submission is not None
    restored = BatchSubmission.from_dict(
        json.loads(json.dumps(timeout.submission.to_dict()))
    )
    api.page([{"task_id": "task-1", "status": "done", "response": _fold_result()}])
    results = await _call(
        api.client().fold_max_accuracy, asynchronous, "results", restored
    )
    assert results[0].complex.id == "ACDEF"
    assert sum(request.url.path.endswith("/submit") for request in api.requests) == 1
    assert not any(request.url.path.endswith("/cancel") for request in api.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_multi_task_timeout_cancels_and_retains_batch_only_handle(
    api, monkeypatch, asynchronous
):
    clock = _Clock()
    monkeypatch.setattr(base_forge_client, "time", clock)
    respond = api.respond

    def slow_pending_task(request: httpx.Request) -> httpx.Response:
        response = respond(request)
        if request.url.path.endswith("/list"):
            clock.now += 2
        return response

    monkeypatch.setattr(api, "respond", slow_pending_task)
    client = api.client()
    api.accepted(3)
    api.data(
        {
            "status": "in_progress",
            "counts": {"done": 1, "in_progress": 1, "queued": 1},
            "total": 3,
        }
    )
    cancellation = {"cancelled": 1, "already_running": 1}
    api.data(cancellation)

    timeout = await _call(
        client.fold_max_accuracy,
        asynchronous,
        "run",
        requests=[
            FoldMaxAccuracyRequest(all_atom_input=_input("ACDEF")),
            FoldMaxAccuracyRequest(all_atom_input=_input("GHIKL")),
            FoldMaxAccuracyRequest(all_atom_input=_input("MNPQR")),
        ],
        timeout=1,
    )

    assert isinstance(timeout, BatchTimeoutError)
    assert timeout.batch_id == "batch-A"
    assert timeout.task_ids == ()
    assert timeout.cancellation == cancellation
    assert timeout.cancellation_error is None
    assert [request.url.path for request in api.requests] == [
        "/api/v1/batch/submit",
        "/api/v1/batch/list",
        "/api/v1/batch/cancel",
    ]
    assert json.loads(api.requests[-1].content) == {"batch_id": "batch-A"}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("dispatch_status", ["in_progress", "cancelled", "failed"])
async def test_wait_polls_one_aggregate_per_cycle_until_no_active_tasks(
    api, monkeypatch, asynchronous, dispatch_status
):
    client = api.client()
    clock = _Clock()
    monkeypatch.setattr(base_forge_client, "time", clock)

    async def sleep(seconds: float) -> None:
        clock.sleep(seconds)

    monkeypatch.setattr(base_forge_client.asyncio, "sleep", sleep)
    submission = BatchSubmission(
        endpoint="fold_max_accuracy", batch_id="batch-A", task_ids=("a", "b", "c", "d")
    )
    api.data(
        {"status": dispatch_status, "counts": {"in_progress": 1, "done": 1}, "total": 4}
    )
    terminal = {
        "status": "done" if dispatch_status == "in_progress" else dispatch_status,
        "counts": {"done": 2, "failed": 1, "cancelled": 1}
        if dispatch_status == "in_progress"
        else {"done": 2},
        "total": 4,
    }
    api.data(terminal)

    assert (
        await _call(
            client, asynchronous, "wait", submission, timeout=5, poll_interval=1
        )
        == terminal
    )

    assert len(api.requests) == 2
    assert all(request.url.path.endswith("/list") for request in api.requests)
    assert all(
        json.loads(request.content)
        == {"batch_id": "batch-A", "include_responses": False}
        for request in api.requests
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("iterate", [False, True], ids=["collect", "iterate"])
async def test_results_paginate_and_only_refresh_stored_outputs(
    api, asynchronous, iterate
):
    client = api.client()
    task_ids = ("stored", "inline", "failed", "cancelled", "queued", "in_progress")
    submission = BatchSubmission(
        endpoint="fold_max_accuracy", batch_id="batch-A", task_ids=task_ids
    )
    api.page(
        [
            {"task_id": "inline", "status": "done", "response": _fold_result()},
            {"task_id": "failed", "status": "failed", "error": "CUDA out of memory"},
        ],
        cursor="page-2",
        total=6,
    )
    api.page(
        [
            {
                "task_id": "cancelled",
                "status": "cancelled",
                "error": "Task expired after waiting too long in the queue.",
            },
            {"task_id": "queued", "status": "queued"},
        ],
        cursor="page-3",
        total=6,
    )
    api.page(
        [
            {"task_id": "stored", "status": "done", "response_available": True},
            {"task_id": "in_progress", "status": "in_progress"},
        ],
        total=6,
    )
    api.completed("fresh-url")
    api.download()

    if iterate:
        if asynchronous:
            entries = [
                entry
                async for entry in client.fold_max_accuracy.async_iter_results(
                    submission, page_size=2
                )
            ]
        else:
            entries = list(
                client.fold_max_accuracy.iter_results(submission, page_size=2)
            )
        assert [task_id for task_id, _ in entries] == [
            "inline",
            "failed",
            "cancelled",
            "queued",
            "stored",
            "in_progress",
        ]
        results = [result for _, result in entries]
    else:
        results = await _call(
            client.fold_max_accuracy, asynchronous, "results", submission, page_size=2
        )

    assert [results[index].complex.id for index in (0, 4)] == ["ACDEF", "ACDEF"]
    failures = [results[index] for index in (1, 2, 3, 5)]
    assert [result.error_code for result in failures] == [500, 409, 409, 409]
    assert [result.task_status for result in failures] == [
        "failed",
        "cancelled",
        "queued",
        "in_progress",
    ]
    assert [result.task_id for result in failures] == [
        "failed",
        "cancelled",
        "queued",
        "in_progress",
    ]
    assert all(result.submission == submission for result in failures)
    assert "CUDA out of memory" in failures[0].error_msg
    assert "Task expired after waiting too long in the queue." in failures[1].error_msg
    list_requests = [
        json.loads(request.content)
        for request in api.requests
        if request.url.path.endswith("/list")
    ]
    assert list_requests == [
        {"batch_id": "batch-A", "include_responses": True, "limit": 2},
        {
            "batch_id": "batch-A",
            "include_responses": True,
            "limit": 2,
            "cursor": "page-2",
        },
        {
            "batch_id": "batch-A",
            "include_responses": True,
            "limit": 2,
            "cursor": "page-3",
        },
    ]
    assert [
        json.loads(request.content)
        for request in api.requests
        if request.url.path.endswith("/status")
    ] == [{"task_id": "stored"}]
    assert [
        str(request.url) for request in api.requests if request.method == "GET"
    ] == ["https://results.example/fresh-url"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_result_decode_error_is_not_a_task_failure(api, asynchronous):
    client = api.client()
    api.data({"status": "done", "response": {"complex": {}}})

    with pytest.raises(BatchError) as error:
        await _call(client.fold_max_accuracy, asynchronous, "result", "saved-task")

    assert error.value.error_code == 502
    assert error.value.task_status is None
    assert error.value.task_id == "saved-task"
    assert len(api.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_on_timeout", [None, False])
async def test_async_cancellation_reraises_with_submission(api, cancel_on_timeout):
    client = api.client()
    polling = asyncio.Event()
    await client._batch_client.async_client.aclose()

    async def respond(request: httpx.Request) -> httpx.Response:
        api.requests.append(request)
        if request.url.path.endswith("/submit"):
            data = {"batch_id": "batch-A", "task_id": "task-1", "total_items": 1}
        elif request.url.path.endswith("/list"):
            polling.set()
            await asyncio.Event().wait()
            raise AssertionError("Cancelled poll resumed")
        else:
            assert request.url.path.endswith("/cancel")
            data = {"cancelled": 1}
        return httpx.Response(200, json={"status": "success", "data": data})

    client._batch_client._async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    )
    options = (
        {} if cancel_on_timeout is None else {"cancel_on_timeout": cancel_on_timeout}
    )
    task = asyncio.create_task(
        client.fold_max_accuracy.async_run(all_atom_input=_input("ACDEF"), **options)
    )
    await asyncio.wait_for(polling.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await task

    assert getattr(cancelled.value, "submission").task_ids == ("task-1",)
    cancellations = [
        request for request in api.requests if request.url.path.endswith("/cancel")
    ]
    assert len(cancellations) == int(cancel_on_timeout is not False)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("method", ["results", "run"])
@pytest.mark.parametrize("failure", ["incomplete", "http"])
async def test_collect_failure_retains_mixed_partial_results(
    api, asynchronous, method, failure
):
    client = api.client()
    submission = BatchSubmission(endpoint="fold_max_accuracy", batch_id="batch-A")
    if method == "run":
        api.accepted(3)
        api.data({"status": "failed", "counts": {"done": 1, "failed": 1}, "total": 3})
    api.page(
        [
            {"task_id": "opaque-success", "status": "done", "response_available": True},
            {"task_id": "opaque-failed", "status": "failed", "error": "Model failed"},
        ],
        cursor="later" if failure == "http" else None,
        total=3,
    )
    api.completed("stored-output")
    api.download("GHIKL")
    if failure == "http":
        api.replies.append(httpx.Response(404, json={"message": "Batch not found"}))

    if method == "run":
        error = await _call(
            client.fold_max_accuracy,
            asynchronous,
            method,
            requests=[
                FoldMaxAccuracyRequest(all_atom_input=_input("ACDEF")) for _ in range(3)
            ],
            page_size=2,
            max_concurrency=1,
        )
    else:
        with pytest.raises(BatchError) as raised:
            await _call(
                client.fold_max_accuracy,
                asynchronous,
                method,
                submission,
                page_size=2,
                max_concurrency=1,
            )
        error = raised.value

    assert isinstance(error, BatchError)
    assert error.error_code == (404 if failure == "http" else 409)
    assert error.submission == submission
    assert error.task_id is None
    assert error.task_status is None
    assert [task_id for task_id, _ in error.partial_results] == [
        "opaque-success",
        "opaque-failed",
    ]
    successful, failed = [result for _, result in error.partial_results]
    assert isinstance(successful, MolecularComplexResult)
    assert successful.complex.id == "GHIKL"
    assert isinstance(failed, BatchError)
    assert failed.task_id == "opaque-failed"
    assert failed.task_status == "failed"
    assert failed.error_code == 500
    assert failed.submission == submission
    assert failed.partial_results == []
    if failure == "http":
        assert isinstance(error.__cause__, ESMProteinError)
        assert error.__cause__.error_code == 404
    assert sum(request.url.path.endswith("/status") for request in api.requests) == 1
    assert [
        str(request.url) for request in api.requests if request.method == "GET"
    ] == ["https://results.example/stored-output"]
    assert not any(request.url.path.endswith("/cancel") for request in api.requests)
    assert api.replies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("method", ["result", "results"])
async def test_expired_result_preserves_http_410_without_retries(
    api, asynchronous, method
):
    client = api.client()
    submission = BatchSubmission(endpoint="fold_max_accuracy", batch_id="batch-A")
    reason = "Results expired after the retention period."
    if method == "results":
        api.page([{"task_id": "task-1", "status": "done", "response_available": False}])
    api.replies.append(httpx.Response(410, json={"message": reason}))

    if method == "result":
        with pytest.raises(BatchError) as raised:
            await _call(client.fold_max_accuracy, asynchronous, method, "task-1")
        result = raised.value
    else:
        results = await _call(
            client.fold_max_accuracy, asynchronous, method, submission
        )
        assert len(results) == 1
        result = results[0]
        assert result.submission == submission

    assert isinstance(result, BatchError)
    assert result.error_code == 410
    assert result.task_id == "task-1"
    assert reason in result.error_msg
    assert [request.url.path for request in api.requests] == [
        *(["/api/v1/batch/list"] if method == "results" else []),
        "/api/v1/batch/status",
    ]
