import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import suppress
from typing import Any, Generic, Literal, TypeVar, overload
from urllib.parse import urljoin

import httpx

from esm.sdk.api import ESMProteinError
from esm.sdk.retry import retry_decorator
from esm.utils.decoding import assemble_message

POLL_BACKOFF_FACTOR = 2.0
POLLS_PER_INTERVAL = 3
DEFAULT_POLL_INTERVAL = 2


def _check_polling(interval: int | None, ceiling: int | None) -> None:
    """Reject a pair one caller supplied that cannot both be honored."""
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
        self.headers = {"Authorization": f"Bearer {self.token}"}
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
        if self.async_client is not None:
            await self.async_client.aclose()

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
        headers: dict[str, str] = {},
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if potential_sequence_of_concern is not None:
            request["potential_sequence_of_concern"] = potential_sequence_of_concern

        headers = {**self.headers, **headers}
        if return_bytes:
            headers["return-bytes"] = "true"
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
        return_bytes: bool = False,
        timeout: int | None = None,
    ):
        try:
            request, headers = self.prepare_request(
                request, potential_sequence_of_concern, return_bytes, headers
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
        timeout: int | None = None,
    ):
        try:
            request, headers = self.prepare_request(
                request, potential_sequence_of_concern, return_bytes, headers
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
        max_retry_attempts: int = 5,
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

    @retry_decorator
    def submit(self, endpoint: str, payload: list[dict[str, Any]]) -> str:
        response_data = self._post(
            "batch/submit",
            {"endpoint": endpoint, "payload": payload},
            timeout=self.transfer_timeout,
        )
        task_id = response_data.get("task_id")
        if not task_id:
            raise ESMProteinError(
                error_code=500, error_msg="API did not return a valid task_id."
            )
        return task_id

    @retry_decorator
    async def async_submit(self, endpoint: str, payload: list[dict[str, Any]]) -> str:
        response_data = await self._async_post(
            "batch/submit",
            {"endpoint": endpoint, "payload": payload},
            timeout=self.transfer_timeout,
        )
        task_id = response_data.get("task_id")
        if not task_id:
            raise ESMProteinError(
                error_code=500, error_msg="API did not return a valid task_id."
            )
        return task_id

    def cancel(self, task_id: str) -> dict[str, Any]:
        return self._post("batch/cancel", {"task_id": task_id})

    async def async_cancel(self, task_id: str) -> dict[str, Any]:
        return await self._async_post("batch/cancel", {"task_id": task_id})

    @retry_decorator
    def get_status(self, task_id: str) -> dict[str, Any]:
        return self._post("batch/status", {"task_id": task_id})

    @retry_decorator
    async def async_get_status(self, task_id: str) -> dict[str, Any]:
        return await self._async_post("batch/status", {"task_id": task_id})

    def wait_for_completion(
        self,
        task_id: str,
        timeout: int,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
    ) -> dict:
        _check_polling(poll_interval, poll_max_interval)
        deadline = time.monotonic() + timeout
        intervals = _poll_intervals(
            *_resolve_polling(
                (poll_interval, self.poll_interval),
                (poll_max_interval, self.poll_max_interval),
            )
        )

        while True:
            response = self.get_status(task_id)
            job_status = response.get("status")
            if job_status == "done":
                return response
            elif job_status == "cancelled":
                raise ESMProteinError(
                    error_code=500, error_msg=f"Job {task_id} cancelled."
                )
            elif job_status == "failed":
                raise ESMProteinError(
                    error_code=500,
                    error_msg=f"Job {task_id} failed with error: '{response.get('error')}'.",
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(next(intervals), remaining))

        raise ESMProteinError(
            error_code=500,
            error_msg=f"Job {task_id} timed out after {timeout} seconds.",
        )

    async def async_wait_for_completion(
        self,
        task_id: str,
        timeout: int,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
    ) -> dict:
        _check_polling(poll_interval, poll_max_interval)
        deadline = time.monotonic() + timeout
        intervals = _poll_intervals(
            *_resolve_polling(
                (poll_interval, self.poll_interval),
                (poll_max_interval, self.poll_max_interval),
            )
        )

        while True:
            response = await self.async_get_status(task_id)
            job_status = response.get("status")
            if job_status == "done":
                return response
            elif job_status == "cancelled":
                raise ESMProteinError(
                    error_code=500, error_msg=f"Job {task_id} cancelled."
                )
            elif job_status == "failed":
                raise ESMProteinError(
                    error_code=500,
                    error_msg=f"Job {task_id} failed with error: '{response.get('error')}'.",
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(next(intervals), remaining))

        raise ESMProteinError(
            error_code=500,
            error_msg=f"Job {task_id} timed out after {timeout} seconds.",
        )

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


TResponse = TypeVar("TResponse")


class EndpointHandler(ABC, Generic[TResponse]):
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
    def _prepare_request(self, **kwargs) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def _process_response(self, response: dict, **kwargs) -> TResponse:
        pass

    @abstractmethod
    async def _async_process_response(self, response: dict, **kwargs) -> TResponse:
        pass

    def run(
        self,
        timeout: int | None = None,
        cancel_on_timeout: bool = True,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
        **kwargs,
    ) -> TResponse | ESMProteinError:
        """
        Submit and execute a batch job, waiting for completion by polling the status of the job.
        Args:
            timeout: Maximum time to wait for job completion, in seconds.
            poll_interval: Seconds between status polls
            poll_max_interval: Ceiling the backoff may grow `poll_interval` to.
            cancel_on_timeout: If True, cancels the batch job if it times out or is interrupted.
            **kwargs: Arguments to pass to the batch job.
        Returns:
            The response from the batch job or an ESMProteinError if the job fails.
        """
        timeout = self.default_timeout if timeout is None else timeout
        task_id = None
        task_timed_out = False
        keyboard_interrupted = False
        interval, max_interval = self._resolved_polling(
            poll_interval, poll_max_interval
        )
        try:
            request = self._prepare_request(**kwargs)
            task_id = self._batch_client.submit(self.endpoint_name, request)
            response = self._batch_client.wait_for_completion(
                task_id, timeout, poll_interval=interval, poll_max_interval=max_interval
            )
            return self._process_response(response, **kwargs)
        except KeyboardInterrupt:
            keyboard_interrupted = True
            raise
        except ESMProteinError as e:
            if "timed out" in e.error_msg:
                task_timed_out = True
            return e
        finally:
            if (
                cancel_on_timeout
                and task_id
                and (task_timed_out or keyboard_interrupted)
            ):
                with suppress(
                    ESMProteinError
                ):  # Don't surface errors from canceling the task
                    with suppress(KeyboardInterrupt):
                        self._batch_client.cancel(task_id)

    async def async_run(
        self,
        timeout: int | None = None,
        cancel_on_timeout: bool = True,
        poll_interval: int | None = None,
        poll_max_interval: int | None = None,
        **kwargs,
    ) -> TResponse | ESMProteinError:
        timeout = self.default_timeout if timeout is None else timeout
        task_id = None
        task_timed_out = False
        keyboard_interrupted = False
        interval, max_interval = self._resolved_polling(
            poll_interval, poll_max_interval
        )
        try:
            request = self._prepare_request(**kwargs)
            task_id = await self._batch_client.async_submit(self.endpoint_name, request)
            response = await self._batch_client.async_wait_for_completion(
                task_id, timeout, poll_interval=interval, poll_max_interval=max_interval
            )
            return await self._async_process_response(response, **kwargs)
        except KeyboardInterrupt:
            keyboard_interrupted = True
            raise
        except ESMProteinError as e:
            if "timed out" in e.error_msg:
                task_timed_out = True
            return e
        finally:
            if (
                cancel_on_timeout
                and task_id
                and (task_timed_out or keyboard_interrupted)
            ):
                with suppress(
                    ESMProteinError
                ):  # Don't surface errors from canceling the task
                    with suppress(KeyboardInterrupt):
                        await self._batch_client.async_cancel(task_id)
