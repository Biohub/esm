import asyncio
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Iterator,
)
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any, Generic, Literal, TypeVar, cast, overload
from urllib.parse import urljoin

import attr
import httpx

from esm import __version__
from esm.sdk.api import ESMProteinError
from esm.sdk.batch import BatchError, BatchSubmission, BatchTimeoutError
from esm.sdk.retry import retry_decorator
from esm.utils.decoding import assemble_message

POLL_BACKOFF_FACTOR = 2.0
POLLS_PER_INTERVAL = 3
DEFAULT_POLL_INTERVAL = 2
TRequest = TypeVar("TRequest", bound=attr.AttrsInstance)
TResponse = TypeVar("TResponse")


def _check_result_options(page_size: int, max_concurrency: int) -> None:
    if not 1 <= page_size <= 1000:
        raise ValueError("page_size must be between 1 and 1000")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")


def _check_polling(interval: int | None, ceiling: int | None) -> None:
    """Reject a pair one caller supplied that cannot both be honored."""
    if any(value is not None and value <= 0 for value in (interval, ceiling)):
        raise ValueError("Polling intervals must be positive")
    if interval is not None and ceiling is not None and ceiling < interval:
        raise ValueError(
            f"poll_max_interval ({ceiling}) is below poll_interval ({interval}); the "
            "backoff ceiling cannot be lower than the interval it grows from"
        )


def _resolve_polling(
    intervals: tuple[int | None, ...], ceilings: tuple[int | None, ...]
) -> tuple[int, int]:
    """Settle the poll interval and its backoff ceiling, most specific preference first."""
    interval = next((v for v in intervals if v is not None), DEFAULT_POLL_INTERVAL)
    ceiling = next((v for v in ceilings if v is not None), interval)
    return interval, max(interval, ceiling)


def _poll_intervals(initial: float, maximum: float | None = None) -> Iterator[float]:
    """Staircased exponential polling backoff.

    Poll at `initial` for `POLLS_PER_INTERVAL` times, then multiply the interval by
    `POLL_BACKOFF_FACTOR` and repeat, up to `maximum`.
    """
    ceiling = initial if maximum is None else maximum
    interval = min(initial, ceiling)
    while True:
        for _ in range(POLLS_PER_INTERVAL):
            yield interval
        interval = min(interval * POLL_BACKOFF_FACTOR, ceiling)


def _sdk_version() -> str:
    return __version__


def merge_headers(base: dict[str, str], overrides: dict[str, str]) -> dict[str, str]:
    """Merge headers so an override replaces a base key of any casing."""
    overridden = {name.lower() for name in overrides}
    merged = {k: v for k, v in base.items() if k.lower() not in overridden}
    merged.update(overrides)
    return merged


def build_biohub_component() -> str:
    """Build the ``X-Biohub-Component`` sent on Forge/Biohub Platform requests."""
    return f"biohub/forge-sdk-python/{_sdk_version()}"


class _BaseForgeInferenceClient:
    def __init__(
        self,
        model: str,
        url: str,
        token: str,
        request_timeout: int | None,
        min_retry_wait: int,
        max_retry_wait: int,
        max_retry_attempts: int,
    ):
        if token == "":
            raise RuntimeError(
                "Please provide a token to connect to Forge/Biohub Platform via token=YOUR_API_TOKEN_HERE"
            )
        self.model = model  # Name of the model to run.
        self.url = url
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Biohub-Component": build_biohub_component(),
        }
        self.request_timeout = request_timeout
        self.min_retry_wait = min_retry_wait
        self.max_retry_wait = max_retry_wait
        self.max_retry_attempts = max_retry_attempts

        self._async_client: httpx.AsyncClient | None = None
        self._client: httpx.Client | None = None

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient()
        return self._async_client

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client()
        return self._client

    def close(self):
        if self._client is not None:
            self._client.close()

    async def aclose(self):
        if self._async_client is not None:
            await self._async_client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def prepare_request(
        self,
        request: dict[str, Any],
        potential_sequence_of_concern: bool | None = None,
        return_bytes: bool = False,
        disable_cache: bool = False,
        headers: dict[str, str] = {},
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if potential_sequence_of_concern is not None:
            request["potential_sequence_of_concern"] = potential_sequence_of_concern

        headers = merge_headers(self.headers, headers)
        if return_bytes:
            headers["return-bytes"] = "true"
        if disable_cache:
            headers["X-Disable-Cache"] = "true"
        return request, headers

    def prepare_data(self, response, endpoint: str) -> dict[str, Any]:
        if not response.is_success:
            raise ESMProteinError(
                error_code=response.status_code,
                error_msg=f"Failure in {endpoint}: {response.text}",
            )
        data = assemble_message(response.headers, response)
        # Nextjs puts outputs dict under "data" key.
        # Lift it up for easier downstream processing.
        if "outputs" not in data and "data" in data:
            data = data["data"]

        # Print warning message if there is any.
        if "warning_messages" in data and data["warning_messages"] is not None:
            for msg in data["warning_messages"]:
                print("\033[31m", msg, "\033[0m")

        return data

    async def _async_post(
        self,
        endpoint,
        request,
        potential_sequence_of_concern: bool | None = None,
        params: dict[str, Any] = {},
        headers: dict[str, str] = {},
        disable_cache: bool = False,
        return_bytes: bool = False,
        timeout: int | None = None,
    ):
        try:
            request, headers = self.prepare_request(
                request,
                potential_sequence_of_concern,
                return_bytes,
                disable_cache,
                headers,
            )
            response = await self.async_client.post(
                url=urljoin(self.url, f"/api/v1/{endpoint}"),
                json=request,
                params=params,
                headers=headers,
                timeout=timeout if timeout is not None else self.request_timeout,
            )
            data = self.prepare_data(response, endpoint)
            return data
        except ESMProteinError as e:
            raise e
        except (TypeError, ValueError) as e:
            # Serializing the request or parsing the reply failed deterministically.
            # 500 would be retried
            raise ESMProteinError(
                error_code=400,
                error_msg=f"Failed to submit request to {endpoint}. Error: {str(e)}",
            )
        except Exception as e:
            raise ESMProteinError(
                error_code=500,
                error_msg=f"Failed to submit request to {endpoint}. Error: {str(e)}",
            )

    def _post(
        self,
        endpoint,
        request,
        potential_sequence_of_concern: bool | None = None,
        params: dict[str, Any] = {},
        headers: dict[str, str] = {},
        return_bytes: bool = False,
        disable_cache: bool = False,
        timeout: int | None = None,
    ):
        try:
            request, headers = self.prepare_request(
                request,
                potential_sequence_of_concern,
                return_bytes,
                disable_cache,
                headers,
            )
            response = self.client.post(
                url=urljoin(self.url, f"/api/v1/{endpoint}"),
                json=request,
                params=params,
                headers=headers,
                timeout=timeout if timeout is not None else self.request_timeout,
            )
            data = self.prepare_data(response, endpoint)
            return data
        except ESMProteinError as e:
            raise e
        except (TypeError, ValueError) as e:
            # Serializing the request or parsing the reply failed deterministically.
            # 500 would be retried
            raise ESMProteinError(
                error_code=400,
                error_msg=f"Failed to submit request to {endpoint}. Error: {str(e)}",
            )
        except Exception as e:
            raise ESMProteinError(
                error_code=500,
                error_msg=f"Failed to submit request to {endpoint}. Error: {str(e)}",
            )


class _BaseForgeBatchClient(_BaseForgeInferenceClient):
    """
    A Python client for the protein folding batch API.
    """

    def __init__(
        self,
        url: str = "https://biohub.ai",
        token: str = "",
        request_timeout: int | None = 120,
        min_retry_wait: int = 1,
        max_retry_wait: int = 10,
        max_retry_attempts: int = 3,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
        transfer_timeout: int | None = 60,
    ):
        super().__init__(
            model="",  # model is not used in batch client
            url=url,
            token=token,
            request_timeout=request_timeout,
            min_retry_wait=min_retry_wait,
            max_retry_wait=max_retry_wait,
            max_retry_attempts=max_retry_attempts,
        )
        _check_polling(poll_interval, poll_max_interval)
        self.poll_interval = poll_interval
        self.poll_max_interval = poll_max_interval
        # Separate (longer) timeout for the payload-sized transfers (submit upload +
        # S3 result download)
        self.transfer_timeout = transfer_timeout

    def submit(
        self, endpoint: str, payload: list[dict[str, Any]], disable_cache: bool = False
    ) -> BatchSubmission:
        try:
            response_data = self._post(
                "batch/submit",
                {"endpoint": endpoint, "payload": payload},
                disable_cache=disable_cache,
                timeout=self.transfer_timeout,
            )
        except ESMProteinError as error:
            raise BatchError.from_error(error) from error
        return self._submission_response(endpoint, len(payload), response_data)

    async def async_submit(
        self, endpoint: str, payload: list[dict[str, Any]], disable_cache: bool = False
    ) -> BatchSubmission:
        try:
            response_data = await self._async_post(
                "batch/submit",
                {"endpoint": endpoint, "payload": payload},
                disable_cache=disable_cache,
                timeout=self.transfer_timeout,
            )
        except ESMProteinError as error:
            raise BatchError.from_error(error) from error
        return self._submission_response(endpoint, len(payload), response_data)

    @staticmethod
    def _submission_response(
        endpoint: str, task_count: int, data: dict[str, Any]
    ) -> BatchSubmission:
        batch_id = data.get("batch_id")
        task_id = data.get("task_id")
        valid_batch_id = isinstance(batch_id, str) and bool(batch_id)
        valid_task_id = isinstance(task_id, str) and bool(task_id)
        total = data.get("total_items")
        valid_total = type(total) is int and total > 0
        submission = None
        if valid_task_id:
            submission = BatchSubmission(
                endpoint=endpoint,
                batch_id=batch_id if valid_batch_id else None,
                task_ids=(task_id,),
            )
        elif valid_batch_id:
            submission = BatchSubmission(endpoint=endpoint, batch_id=batch_id)

        # Legacy support
        if (
            task_count == 1
            and valid_task_id
            and "batch_id" not in data
            and "total_items" not in data
        ):
            return cast(BatchSubmission, submission)

        if not valid_batch_id or not valid_total or total != task_count:
            raise BatchError(
                502, f"Unexpected submission response: {data}", submission=submission
            )
        return cast(BatchSubmission, submission)

    @staticmethod
    def _cancel_request(task_id: str | None, batch_id: str | None) -> dict[str, str]:
        if bool(task_id) == bool(batch_id):
            raise ValueError("Provide exactly one of task_id or batch_id")
        if task_id:
            return {"task_id": task_id}
        return {"batch_id": cast(str, batch_id)}

    def cancel(
        self, task_id: str | None = None, *, batch_id: str | None = None
    ) -> dict[str, Any]:
        return self._post("batch/cancel", self._cancel_request(task_id, batch_id))

    async def async_cancel(
        self, task_id: str | None = None, *, batch_id: str | None = None
    ) -> dict[str, Any]:
        return await self._async_post(
            "batch/cancel", self._cancel_request(task_id, batch_id)
        )

    @retry_decorator
    def get_status(self, task_id: str) -> dict[str, Any]:
        return self._post("batch/status", {"task_id": task_id})

    @retry_decorator
    async def async_get_status(self, task_id: str) -> dict[str, Any]:
        return await self._async_post("batch/status", {"task_id": task_id})

    @staticmethod
    def _list_batch_request(
        batch_id: str,
        status: list[str] | None,
        include_responses: bool,
        limit: int | None,
        cursor: str | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "batch_id": batch_id,
            "include_responses": include_responses,
        }
        if status is not None:
            request["status"] = status
        if limit is not None:
            request["limit"] = limit
        if cursor is not None:
            request["cursor"] = cursor
        return request

    @retry_decorator
    def list_batch(
        self,
        batch_id: str,
        *,
        status: list[str] | None = None,
        include_responses: bool = False,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "batch/list",
            self._list_batch_request(
                batch_id, status, include_responses, limit, cursor
            ),
        )

    @retry_decorator
    async def async_list_batch(
        self,
        batch_id: str,
        *,
        status: list[str] | None = None,
        include_responses: bool = False,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return await self._async_post(
            "batch/list",
            self._list_batch_request(
                batch_id, status, include_responses, limit, cursor
            ),
        )

    def wait(
        self,
        submission: BatchSubmission,
        timeout: int,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Return a terminal batch summary, or task states for a task-only handle."""
        _check_polling(poll_interval, poll_max_interval)
        deadline = time.monotonic() + timeout
        intervals = _poll_intervals(
            *_resolve_polling(
                (poll_interval, self.poll_interval),
                (poll_max_interval, self.poll_max_interval),
            )
        )

        completed: dict[str, dict[str, Any]] = {}
        while True:
            if submission.batch_id is not None:
                try:
                    summary = self.list_batch(submission.batch_id)
                except ESMProteinError as error:
                    raise BatchError.from_error(error, submission) from error
                if self._batch_is_complete(summary):
                    return summary
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(next(intervals), remaining))
                continue
            polled = False
            for task_id in submission.task_ids:
                if task_id in completed:
                    continue
                if polled and time.monotonic() >= deadline:
                    raise BatchTimeoutError(submission, timeout)
                try:
                    response = self.get_status(task_id)
                except ESMProteinError as error:
                    raise BatchError.from_error(error, submission, task_id) from error
                if response["status"] in {"done", "failed", "cancelled"}:
                    completed[task_id] = response
                polled = True
            if len(completed) == len(submission.task_ids):
                return [completed[task_id] for task_id in submission.task_ids]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(next(intervals), remaining))

        raise BatchTimeoutError(submission, timeout)

    async def async_wait(
        self,
        submission: BatchSubmission,
        timeout: int,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        _check_polling(poll_interval, poll_max_interval)
        deadline = time.monotonic() + timeout
        intervals = _poll_intervals(
            *_resolve_polling(
                (poll_interval, self.poll_interval),
                (poll_max_interval, self.poll_max_interval),
            )
        )

        completed: dict[str, dict[str, Any]] = {}
        while True:
            if submission.batch_id is not None:
                try:
                    summary = await self.async_list_batch(submission.batch_id)
                except ESMProteinError as error:
                    raise BatchError.from_error(error, submission) from error
                if self._batch_is_complete(summary):
                    return summary
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(next(intervals), remaining))
                continue
            polled = False
            for task_id in submission.task_ids:
                if task_id in completed:
                    continue
                if polled and time.monotonic() >= deadline:
                    raise BatchTimeoutError(submission, timeout)
                try:
                    response = await self.async_get_status(task_id)
                except ESMProteinError as error:
                    raise BatchError.from_error(error, submission, task_id) from error
                if response["status"] in {"done", "failed", "cancelled"}:
                    completed[task_id] = response
                polled = True
            if len(completed) == len(submission.task_ids):
                return [completed[task_id] for task_id in submission.task_ids]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(next(intervals), remaining))

        raise BatchTimeoutError(submission, timeout)

    @staticmethod
    def _batch_is_complete(summary: dict[str, Any]) -> bool:
        counts = summary["counts"]
        if counts.get("queued", 0) + counts.get("in_progress", 0) > 0:
            return False
        # Cancelled/failed dispatch can leave tasks uncreated, but running work must finish.
        if summary["status"] in {"failed", "cancelled"}:
            return True
        terminal = sum(
            counts.get(state, 0) for state in ("done", "failed", "cancelled")
        )
        return summary["status"] == "done" and terminal == summary["total"]

    @staticmethod
    def _require_done(task_id: str, status: dict[str, Any]) -> None:
        state = status["status"]
        if state == "done":
            return
        if state == "failed":
            raise BatchError(
                500,
                f"Job {task_id} failed with error: '{status.get('error')}'.",
                task_id=task_id,
                task_status=state,
            )
        message = f"Job {task_id} is {state}."
        if state == "cancelled" and status.get("error"):
            message += f" {status['error']}"
        raise BatchError(409, message, task_id=task_id, task_status=state)

    def wait_for_completion(
        self,
        task_id: str,
        timeout: int,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
    ) -> dict[str, Any]:
        submission = BatchSubmission(endpoint="", task_ids=(task_id,))
        statuses = self.wait(submission, timeout, poll_interval, poll_max_interval)
        status = cast(list[dict[str, Any]], statuses)[0]
        self._require_done(task_id, status)
        return status

    async def async_wait_for_completion(
        self,
        task_id: str,
        timeout: int,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
    ) -> dict[str, Any]:
        submission = BatchSubmission(endpoint="", task_ids=(task_id,))
        statuses = await self.async_wait(
            submission, timeout, poll_interval, poll_max_interval
        )
        status = cast(list[dict[str, Any]], statuses)[0]
        self._require_done(task_id, status)
        return status

    @staticmethod
    def _check_listing_complete(
        submission: BatchSubmission, seen: set[str], summary: dict[str, Any]
    ) -> None:
        total = summary["total"]
        if len(seen) != total:
            raise BatchError(
                409,
                f"Batch {submission.batch_id} lists {len(seen)} of {total} tasks "
                f"(status: {summary['status']}); not all tasks are available.",
                submission=submission,
            )

    @staticmethod
    def _record_listed_tasks(
        submission: BatchSubmission, seen: set[str], items: list[dict[str, Any]]
    ) -> None:
        for item in items:
            task_id = item["task_id"]
            if task_id in seen:
                raise BatchError(
                    502, f"Batch listing repeated task {task_id}", submission
                )
            seen.add(task_id)

    def _result_pages(
        self, submission: BatchSubmission, page_size: int
    ) -> Iterator[list[dict[str, Any]]]:
        if submission.batch_id is None:
            for start in range(0, len(submission.task_ids), page_size):
                yield [
                    {"task_id": task_id}
                    for task_id in submission.task_ids[start : start + page_size]
                ]
            return
        cursor = None
        cursors: set[str] = set()
        seen: set[str] = set()
        while True:
            try:
                page = self.list_batch(
                    submission.batch_id,
                    include_responses=True,
                    limit=page_size,
                    cursor=cursor,
                )
            except ESMProteinError as error:
                raise BatchError.from_error(error, submission) from error
            self._record_listed_tasks(submission, seen, page["items"])
            yield page["items"]
            cursor = page["cursor"]
            if cursor is None:
                break
            if cursor in cursors:
                raise BatchError(
                    502, "Batch listing repeated a pagination cursor", submission
                )
            cursors.add(cursor)
        self._check_listing_complete(submission, seen, page)

    async def _async_result_pages(
        self, submission: BatchSubmission, page_size: int
    ) -> AsyncIterator[list[dict[str, Any]]]:
        if submission.batch_id is None:
            for page in self._result_pages(submission, page_size):
                yield page
            return
        cursor = None
        cursors: set[str] = set()
        seen: set[str] = set()
        while True:
            try:
                page = await self.async_list_batch(
                    submission.batch_id,
                    include_responses=True,
                    limit=page_size,
                    cursor=cursor,
                )
            except ESMProteinError as error:
                raise BatchError.from_error(error, submission) from error
            self._record_listed_tasks(submission, seen, page["items"])
            yield page["items"]
            cursor = page["cursor"]
            if cursor is None:
                break
            if cursor in cursors:
                raise BatchError(
                    502, "Batch listing repeated a pagination cursor", submission
                )
            cursors.add(cursor)
        self._check_listing_complete(submission, seen, page)

    def _retrieve_result(
        self,
        item: dict[str, Any],
        process_response: Callable[[dict[str, Any]], TResponse],
        submission: BatchSubmission | None = None,
    ) -> TResponse | BatchError:
        task_id = item["task_id"]
        try:
            if "status" not in item or (
                item["status"] == "done" and "response" not in item
            ):
                item = self.get_status(task_id)
            self._require_done(task_id, item)
            return process_response(item)
        except ESMProteinError as error:
            return BatchError.from_error(error, submission, task_id)
        except (KeyError, TypeError, ValueError) as error:
            return BatchError(
                502, f"Cannot decode task {task_id}: {error}", submission, task_id
            )

    async def _async_retrieve_result(
        self,
        item: dict[str, Any],
        process_response: Callable[[dict[str, Any]], Awaitable[TResponse]],
        submission: BatchSubmission | None = None,
    ) -> TResponse | BatchError:
        task_id = item["task_id"]
        try:
            if "status" not in item or (
                item["status"] == "done" and "response" not in item
            ):
                item = await self.async_get_status(task_id)
            self._require_done(task_id, item)
            return await process_response(item)
        except ESMProteinError as error:
            return BatchError.from_error(error, submission, task_id)
        except (KeyError, TypeError, ValueError) as error:
            return BatchError(
                502, f"Cannot decode task {task_id}: {error}", submission, task_id
            )

    def _iter_results(
        self,
        submission: BatchSubmission,
        process_response: Callable[[dict[str, Any]], TResponse],
        *,
        page_size: int,
        max_concurrency: int,
    ) -> Generator[tuple[str, TResponse | ESMProteinError], None, None]:
        # Initialize once before worker threads share the HTTP connection pool.
        _ = self.client
        with ThreadPoolExecutor(
            max_workers=min(max_concurrency, page_size)
        ) as executor:
            for items in self._result_pages(submission, page_size):
                # Bound prefetched outputs as well as simultaneous downloads.
                for start in range(0, len(items), max_concurrency):
                    chunk = items[start : start + max_concurrency]
                    pending = deque(
                        executor.submit(
                            copy_context().run,
                            self._retrieve_result,
                            item,
                            process_response,
                            submission,
                        )
                        for item in chunk
                    )
                    for item in chunk:
                        output = cast(
                            TResponse | ESMProteinError, pending.popleft().result()
                        )
                        yield item["task_id"], output
                        del output

    async def _async_iter_results(
        self,
        submission: BatchSubmission,
        process_response: Callable[[dict[str, Any]], Awaitable[TResponse]],
        *,
        page_size: int,
        max_concurrency: int,
    ) -> AsyncGenerator[tuple[str, TResponse | ESMProteinError], None]:
        async for items in self._async_result_pages(submission, page_size):
            for start in range(0, len(items), max_concurrency):
                chunk = items[start : start + max_concurrency]
                results: dict[int, TResponse | ESMProteinError] = {}

                async def retrieve_into(index: int, item: dict[str, Any]) -> None:
                    # Keep outputs outside Task results, which can outlive aclose().
                    results[index] = await self._async_retrieve_result(
                        item, process_response, submission
                    )

                tasks = [
                    asyncio.create_task(retrieve_into(index, item))
                    for index, item in enumerate(chunk)
                ]
                try:
                    await asyncio.gather(*tasks)
                    for index, item in enumerate(chunk):
                        yield item["task_id"], results.pop(index)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    results.clear()

    def _collect_results(
        self,
        submission: BatchSubmission,
        process_response: Callable[[dict[str, Any]], TResponse],
        *,
        page_size: int,
        max_concurrency: int,
    ) -> list[TResponse | ESMProteinError]:
        results: list[tuple[str, TResponse | ESMProteinError]] = []
        try:
            results.extend(
                self._iter_results(
                    submission,
                    process_response,
                    page_size=page_size,
                    max_concurrency=max_concurrency,
                )
            )
        except BatchError as error:
            error.partial_results = results
            raise
        return [output for _, output in results]

    async def _async_collect_results(
        self,
        submission: BatchSubmission,
        process_response: Callable[[dict[str, Any]], Awaitable[TResponse]],
        *,
        page_size: int,
        max_concurrency: int,
    ) -> list[TResponse | ESMProteinError]:
        results: list[tuple[str, TResponse | ESMProteinError]] = []
        try:
            async for result in self._async_iter_results(
                submission,
                process_response,
                page_size=page_size,
                max_concurrency=max_concurrency,
            ):
                results.append(result)
        except BatchError as error:
            error.partial_results = results
            raise
        return [output for _, output in results]

    def _read_response(
        self, status: dict[str, Any], *, return_bytes: bool = False
    ) -> Any:
        response = status["response"]
        if isinstance(response, str) and response.startswith(("https://", "http://")):
            return self.get_result_from_s3(response, return_bytes=return_bytes)
        return response

    async def _async_read_response(
        self, status: dict[str, Any], *, return_bytes: bool = False
    ) -> Any:
        response = status["response"]
        if isinstance(response, str) and response.startswith(("https://", "http://")):
            return await self.async_get_result_from_s3(
                response, return_bytes=return_bytes
            )
        return response

    @overload
    def get_result_from_s3(
        self, s3_url: str, return_bytes: Literal[False] = False
    ) -> dict[str, Any]: ...

    @overload
    def get_result_from_s3(self, s3_url: str, return_bytes: Literal[True]) -> bytes: ...

    @retry_decorator
    def get_result_from_s3(
        self, s3_url: str, return_bytes: bool = False
    ) -> dict[str, Any] | bytes:
        """Downloads the result JSON from a pre-signed S3 URL."""
        try:
            response = self.client.get(s3_url, timeout=self.transfer_timeout)
            response.raise_for_status()
            if return_bytes:
                return response.content
            else:
                return response.json()
        except Exception as e:
            raise ESMProteinError(
                error_code=500,
                error_msg=f"Failed to download result from S3 URL: {s3_url}. Error: {str(e)}",
            )

    @overload
    async def async_get_result_from_s3(
        self, s3_url: str, return_bytes: Literal[False] = False
    ) -> dict[str, Any]: ...

    @overload
    async def async_get_result_from_s3(
        self, s3_url: str, return_bytes: Literal[True]
    ) -> bytes: ...

    @retry_decorator
    async def async_get_result_from_s3(
        self, s3_url: str, return_bytes: bool = False
    ) -> dict[str, Any] | bytes:
        """Asynchronously downloads the result JSON from a pre-signed S3 URL."""
        try:
            response = await self.async_client.get(
                s3_url, timeout=self.transfer_timeout
            )
            response.raise_for_status()
            if return_bytes:
                return response.content
            else:
                return response.json()
        except Exception as e:
            raise ESMProteinError(
                error_code=500,
                error_msg=f"Failed to download result from S3 URL: {s3_url}. Error: {str(e)}",
            )


class EndpointHandler(ABC, Generic[TRequest, TResponse]):
    request_type: type[TRequest]
    poll_interval: int | None = None
    poll_max_interval: int = 30
    default_timeout: int = 30 * 60  # 30 minutes

    def __init__(self, batch_client: _BaseForgeBatchClient):
        self._batch_client = batch_client
        self.min_retry_wait = batch_client.min_retry_wait
        self.max_retry_wait = batch_client.max_retry_wait
        self.max_retry_attempts = batch_client.max_retry_attempts

    def _resolved_polling(
        self, poll_interval: int | None = None, poll_max_interval: int | None = None
    ) -> tuple[int, int]:
        _check_polling(poll_interval, poll_max_interval)
        return _resolve_polling(
            (poll_interval, self._batch_client.poll_interval, self.poll_interval),
            (
                poll_max_interval,
                self._batch_client.poll_max_interval,
                self.poll_max_interval,
            ),
        )

    @property
    @abstractmethod
    def endpoint_name(self) -> str:
        pass

    @abstractmethod
    def _prepare_request(
        self, request: TRequest, *, model_name: str | None = None
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def _process_response(self, response: dict) -> TResponse:
        pass

    @abstractmethod
    async def _async_process_response(self, response: dict) -> TResponse:
        pass

    def _prepare_requests(
        self,
        requests: Iterable[TRequest] | None,
        shared: dict[str, Any],
        *,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        if requests is None:
            # attrs rejects unknown and missing fields before any HTTP request.
            return [
                self._prepare_request(
                    self.request_type(**shared), model_name=model_name
                )
            ]
        requests = list(requests)
        if not requests:
            raise ValueError("requests must contain at least one task")
        name = self.request_type.__name__
        fields = attr.fields_dict(self.request_type)
        for key in shared:
            if key != "config" or key not in fields:
                raise TypeError(
                    f"{key}= cannot be shared across requests; set it on each {name}"
                )
        for index, request in enumerate(requests):
            if not isinstance(request, self.request_type):
                raise TypeError(
                    f"requests[{index}] must be a {name}, got {type(request).__name__}"
                )
        return [
            self._prepare_request(
                attr.evolve(
                    request,
                    **{k: v for k, v in shared.items() if getattr(request, k) is None},
                ),
                model_name=model_name,
            )
            for request in requests
        ]

    def submit(
        self,
        requests: Iterable[TRequest] | None = None,
        *,
        model_name: str | None = None,
        disable_cache: bool = False,
        **kwargs: Any,
    ) -> BatchSubmission:
        """Submit one task from request fields as keywords, or an iterable of requests.

        Pass the iterable positionally or as ``requests=``.
        Iterables, including generators, are materialized before validation.
        With ``requests``, only ``config`` supplies a default for requests whose
        config is ``None``. Other inputs must be set on each request.
        """
        payloads = self._prepare_requests(requests, kwargs, model_name=model_name)
        return self._batch_client.submit(
            self.endpoint_name, payloads, disable_cache=disable_cache
        )

    async def async_submit(
        self,
        requests: Iterable[TRequest] | None = None,
        *,
        model_name: str | None = None,
        disable_cache: bool = False,
        **kwargs: Any,
    ) -> BatchSubmission:
        payloads = self._prepare_requests(requests, kwargs, model_name=model_name)
        return await self._batch_client.async_submit(
            self.endpoint_name, payloads, disable_cache=disable_cache
        )

    def _check_submission(self, submission: BatchSubmission) -> None:
        if submission.endpoint != self.endpoint_name:
            raise ValueError(
                f"Expected a {self.endpoint_name} submission, got {submission.endpoint}"
            )

    def wait(
        self,
        submission: BatchSubmission,
        timeout: int | None = None,
        *,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Return a batch summary (task states without a batch ID); never cancel."""
        self._check_submission(submission)
        interval, ceiling = self._resolved_polling(poll_interval, poll_max_interval)
        return self._batch_client.wait(
            submission,
            self.default_timeout if timeout is None else timeout,
            poll_interval=interval,
            poll_max_interval=ceiling,
        )

    async def async_wait(
        self,
        submission: BatchSubmission,
        timeout: int | None = None,
        *,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        self._check_submission(submission)
        interval, ceiling = self._resolved_polling(poll_interval, poll_max_interval)
        return await self._batch_client.async_wait(
            submission,
            self.default_timeout if timeout is None else timeout,
            poll_interval=interval,
            poll_max_interval=ceiling,
        )

    def result(self, task_id: str) -> TResponse:
        """Retrieve a completed result; otherwise raise with its ``task_status``."""
        output = self._batch_client._retrieve_result(
            {"task_id": task_id}, self._process_response
        )
        if isinstance(output, BatchError):
            raise output
        return output

    async def async_result(self, task_id: str) -> TResponse:
        output = await self._batch_client._async_retrieve_result(
            {"task_id": task_id}, self._async_process_response
        )
        if isinstance(output, BatchError):
            raise output
        return output

    def iter_results(
        self,
        submission: BatchSubmission,
        *,
        page_size: int = 100,
        max_concurrency: int = 8,
    ) -> Generator[tuple[str, TResponse | ESMProteinError], None, None]:
        """Yield authoritative task IDs and decoded outputs in server listing order.

        Parameters
        ----------
        submission
            Handle returned by ``submit()``, containing the endpoint and job IDs.
        page_size
            Entries per listing request, between 1 and 1000.
        max_concurrency
            Maximum number of prefetched task outputs and simultaneous downloads.

        Notes
        -----
        Incomplete dispatch raises ``BatchError`` after available results are yielded.
        Task-only handles use the supplied task ID order.
        """
        self._check_submission(submission)
        _check_result_options(page_size, max_concurrency)
        return self._batch_client._iter_results(
            submission,
            self._process_response,
            page_size=page_size,
            max_concurrency=max_concurrency,
        )

    def async_iter_results(
        self,
        submission: BatchSubmission,
        *,
        page_size: int = 100,
        max_concurrency: int = 8,
    ) -> AsyncGenerator[tuple[str, TResponse | ESMProteinError], None]:
        """Asynchronous equivalent of :meth:`iter_results`; use ``async for``."""
        self._check_submission(submission)
        _check_result_options(page_size, max_concurrency)
        return self._batch_client._async_iter_results(
            submission,
            self._async_process_response,
            page_size=page_size,
            max_concurrency=max_concurrency,
        )

    def results(
        self,
        submission: BatchSubmission,
        *,
        page_size: int = 100,
        max_concurrency: int = 8,
    ) -> list[TResponse | ESMProteinError]:
        """Collect decoded results in server listing order, retaining per-task errors.

        Parameters
        ----------
        submission
            Handle returned by ``submit()``, containing the endpoint and batch or task IDs.
        page_size
            Entries per listing request, between 1 and 1000. All pages are read;
            this does not limit the number of returned results or their memory.
        max_concurrency
            Maximum number of simultaneous task result retrievals.

        Notes
        -----
        Inline results need no status read. Stored outputs use fresh status
        reads and downloads. Failed, cancelled, and unfinished tasks return
        errors with ``task_status`` set.
        Listing failures and incomplete dispatch raise ``BatchError`` with the
        submission and collected ``(task_id, output)`` pairs in ``partial_results``.
        Use :meth:`iter_results` to consume outputs without collecting the whole batch.
        """
        self._check_submission(submission)
        _check_result_options(page_size, max_concurrency)
        return self._batch_client._collect_results(
            submission,
            self._process_response,
            page_size=page_size,
            max_concurrency=max_concurrency,
        )

    async def async_results(
        self,
        submission: BatchSubmission,
        *,
        page_size: int = 100,
        max_concurrency: int = 8,
    ) -> list[TResponse | ESMProteinError]:
        """Asynchronous equivalent of :meth:`results`, with the same limits."""
        self._check_submission(submission)
        _check_result_options(page_size, max_concurrency)
        return await self._batch_client._async_collect_results(
            submission,
            self._async_process_response,
            page_size=page_size,
            max_concurrency=max_concurrency,
        )

    @staticmethod
    def _cancellation_target(submission: BatchSubmission) -> dict[str, str]:
        if submission.batch_id is not None:
            return {"batch_id": submission.batch_id}
        if len(submission.task_ids) == 1:
            return {"task_id": submission.task_id}
        raise ValueError("Cancelling multiple tasks requires their batch_id")

    def _cancel_after_error(
        self, submission: BatchSubmission, error: BatchError
    ) -> None:
        try:
            error.cancellation = self._batch_client.cancel(
                **self._cancellation_target(submission)
            )
        except ESMProteinError as cancellation_error:
            error.cancellation_error = cancellation_error
            warnings.warn(
                f"Could not cancel job: {cancellation_error.error_msg}",
                RuntimeWarning,
                stacklevel=3,
            )

    async def _async_cancel_after_error(
        self, submission: BatchSubmission, error: BatchError
    ) -> None:
        try:
            error.cancellation = await self._batch_client.async_cancel(
                **self._cancellation_target(submission)
            )
        except ESMProteinError as cancellation_error:
            error.cancellation_error = cancellation_error
            warnings.warn(
                f"Could not cancel job: {cancellation_error.error_msg}",
                RuntimeWarning,
                stacklevel=3,
            )

    @overload
    def run(
        self,
        requests: Iterable[TRequest],
        *,
        timeout: int | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> list[TResponse | ESMProteinError] | ESMProteinError: ...

    @overload
    def run(
        self,
        requests: None = None,
        *,
        timeout: int | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> TResponse | ESMProteinError: ...

    def run(
        self,
        requests: Iterable[TRequest] | None = None,
        *,
        timeout: int | None = None,
        model_name: str | None = None,
        disable_cache: bool = False,
        cancel_on_timeout: bool = True,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
        page_size: int = 100,
        max_concurrency: int = 8,
        **kwargs,
    ) -> TResponse | list[TResponse | ESMProteinError] | ESMProteinError:
        """Submit, wait, and decode tasks in server listing order.

        ``requests`` (positional or named) selects list output, even for one item; keywords
        combine with it as in :meth:`submit`.
        Invalid SDK arguments raise ``TypeError`` or ``ValueError`` before submission.
        Server submission and waiting failures retain accepted job IDs in
        ``BatchError.submission``.
        Retrieval failures return ``BatchError`` with collected ``(task_id, output)``
        pairs in ``partial_results``.
        Timeout or interruption attempts to cancel queued work by default;
        running tasks can still finish and be billed. Set ``cancel_on_timeout=False``
        to leave work running and recover through ``BatchError.submission``.
        Use ``submit()`` then ``wait()`` to save a handle before waiting;
        ``wait()`` never cancels work. ``page_size`` and ``max_concurrency``
        configure result retrieval, as on ``results()``.
        """
        timeout = self.default_timeout if timeout is None else timeout
        _check_result_options(page_size, max_concurrency)
        submission = None
        interval, max_interval = self._resolved_polling(
            poll_interval, poll_max_interval
        )
        try:
            submission = self.submit(
                requests=requests,
                disable_cache=disable_cache,
                model_name=model_name,
                **kwargs,
            )
            self.wait(
                submission,
                timeout,
                poll_interval=interval,
                poll_max_interval=max_interval,
            )
            outputs = self._batch_client._collect_results(
                submission,
                self._process_response,
                page_size=page_size,
                max_concurrency=max_concurrency,
            )
            return outputs if requests is not None else outputs[0]
        except KeyboardInterrupt as interruption:
            if submission is not None:
                interruption.add_note(
                    f"Submitted batch {submission.batch_id}; tasks {submission.task_ids}"
                )
                setattr(interruption, "submission", submission)
                if cancel_on_timeout:
                    error = BatchError(499, "Interrupted", submission)
                    self._cancel_after_error(submission, error)
                    setattr(interruption, "cancellation", error.cancellation)
                    setattr(
                        interruption, "cancellation_error", error.cancellation_error
                    )
            raise
        except ESMProteinError as original:
            error = BatchError.from_error(original, submission)
            if (
                cancel_on_timeout
                and submission is not None
                and isinstance(error, BatchTimeoutError)
            ):
                self._cancel_after_error(submission, error)
            return error

    @overload
    async def async_run(
        self,
        requests: Iterable[TRequest],
        *,
        timeout: int | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> list[TResponse | ESMProteinError] | ESMProteinError: ...

    @overload
    async def async_run(
        self,
        requests: None = None,
        *,
        timeout: int | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> TResponse | ESMProteinError: ...

    async def async_run(
        self,
        requests: Iterable[TRequest] | None = None,
        *,
        timeout: int | None = None,
        model_name: str | None = None,
        disable_cache: bool = False,
        cancel_on_timeout: bool = True,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
        page_size: int = 100,
        max_concurrency: int = 8,
        **kwargs,
    ) -> TResponse | list[TResponse | ESMProteinError] | ESMProteinError:
        """Async equivalent of ``run()``, including cancellation on timeout by default.

        Cancelling the awaiting coroutine also attempts to cancel queued work.
        Set ``cancel_on_timeout=False`` to leave work running.
        """
        timeout = self.default_timeout if timeout is None else timeout
        _check_result_options(page_size, max_concurrency)
        submission = None
        interval, max_interval = self._resolved_polling(
            poll_interval, poll_max_interval
        )
        try:
            submission = await self.async_submit(
                requests=requests,
                disable_cache=disable_cache,
                model_name=model_name,
                **kwargs,
            )
            await self.async_wait(
                submission,
                timeout,
                poll_interval=interval,
                poll_max_interval=max_interval,
            )
            outputs = await self._batch_client._async_collect_results(
                submission,
                self._async_process_response,
                page_size=page_size,
                max_concurrency=max_concurrency,
            )
            return outputs if requests is not None else outputs[0]
        except (KeyboardInterrupt, asyncio.CancelledError) as interruption:
            if submission is not None:
                interruption.add_note(
                    f"Submitted batch {submission.batch_id}; tasks {submission.task_ids}"
                )
                setattr(interruption, "submission", submission)
                if cancel_on_timeout:
                    error = BatchError(499, "Interrupted", submission)
                    await self._async_cancel_after_error(submission, error)
                    setattr(interruption, "cancellation", error.cancellation)
                    setattr(
                        interruption, "cancellation_error", error.cancellation_error
                    )
            raise
        except ESMProteinError as original:
            error = BatchError.from_error(original, submission)
            if (
                cancel_on_timeout
                and submission is not None
                and isinstance(error, BatchTimeoutError)
            ):
                await self._async_cancel_after_error(submission, error)
            return error
