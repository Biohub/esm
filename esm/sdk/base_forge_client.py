import asyncio
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any, Generic, TypeVar
from urllib.parse import urljoin

import httpx

from esm.sdk.api import ESMProteinError
from esm.sdk.retry import retry_decorator
from esm.utils.decoding import assemble_message


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
                "Please provide a token to connect to Forge via token=YOUR_API_TOKEN_HERE"
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
        disable_cache: bool = False,
        headers: dict[str, str] = {},
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if potential_sequence_of_concern is not None:
            request["potential_sequence_of_concern"] = potential_sequence_of_concern

        headers = {**self.headers, **headers}
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
                timeout=self.request_timeout,
            )
            data = self.prepare_data(response, endpoint)
            return data
        except ESMProteinError as e:
            raise e
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
                timeout=self.request_timeout,
            )
            data = self.prepare_data(response, endpoint)
            return data
        except ESMProteinError as e:
            raise e
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
        url: str = "https://forge.evolutionaryscale.ai",
        token: str = "",
        request_timeout: int | None = None,
        min_retry_wait: int = 1,
        max_retry_wait: int = 10,
        max_retry_attempts: int = 5,
        poll_interval: int = 2,
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
        # How often to poll for status
        self.poll_interval = poll_interval

    @retry_decorator
    def submit(
        self, endpoint: str, payload: list[dict[str, Any]], disable_cache: bool = False
    ) -> str:
        response_data = self._post(
            "batch/submit",
            {"endpoint": endpoint, "payload": payload},
            disable_cache=disable_cache,
        )
        task_id = response_data.get("task_id")
        if not task_id:
            raise ESMProteinError(
                error_code=500, error_msg="API did not return a valid task_id."
            )
        return task_id

    @retry_decorator
    async def async_submit(
        self, endpoint: str, payload: list[dict[str, Any]], disable_cache: bool = False
    ) -> str:
        response_data = await self._async_post(
            "batch/submit",
            {"endpoint": endpoint, "payload": payload},
            disable_cache=disable_cache,
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

    def wait_for_completion(self, task_id: str, timeout: int) -> dict:
        start_time = time.time()

        while time.time() - start_time < timeout:
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
            time.sleep(self.poll_interval)

        raise ESMProteinError(
            error_code=500,
            error_msg=f"Job {task_id} timed out after {timeout} seconds.",
        )

    async def async_wait_for_completion(self, task_id: str, timeout: int) -> dict:
        start_time = time.time()

        while time.time() - start_time < timeout:
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
            await asyncio.sleep(self.poll_interval)

        raise ESMProteinError(
            error_code=500,
            error_msg=f"Job {task_id} timed out after {timeout} seconds.",
        )

    @retry_decorator
    def get_result_from_s3(self, s3_url: str) -> dict[str, Any]:
        """Downloads the result JSON from a pre-signed S3 URL."""
        try:
            response = self.client.get(s3_url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ESMProteinError(
                error_code=500,
                error_msg=f"Failed to download result from S3 URL: {s3_url}. Error: {str(e)}",
            )

    @retry_decorator
    async def async_get_result_from_s3(self, s3_url: str) -> dict[str, Any]:
        """Asynchronously downloads the result JSON from a pre-signed S3 URL."""
        try:
            response = await self.async_client.get(s3_url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ESMProteinError(
                error_code=500,
                error_msg=f"Failed to download result from S3 URL: {s3_url}. Error: {str(e)}",
            )


TResponse = TypeVar("TResponse")


class EndpointHandler(ABC, Generic[TResponse]):
    def __init__(self, batch_client: _BaseForgeBatchClient):
        self._batch_client = batch_client
        self.min_retry_wait = batch_client.min_retry_wait
        self.max_retry_wait = batch_client.max_retry_wait
        self.max_retry_attempts = batch_client.max_retry_attempts

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
        timeout: int = 300,
        disable_cache: bool = False,
        cancel_on_timeout: bool = True,
        **kwargs,
    ) -> TResponse | ESMProteinError:
        """
        Submit and execute a batch job, waiting for completion by polling the status of the job.
        Args:
            timeout: Maximum time to wait for job completion, in seconds.
            disable_cache: If True, bypasses any cached results and forces
            a fresh computation.
            cancel_on_timeout: If True, cancels the batch job if it times out or is interrupted.
            **kwargs: Arguments to pass to the batch job.
        Returns:
            The response from the batch job or an ESMProteinError if the job fails.
        """
        task_id = None
        task_timed_out = False
        keyboard_interrupted = False
        try:
            request = self._prepare_request(**kwargs)
            task_id = self._batch_client.submit(
                self.endpoint_name, request, disable_cache=disable_cache
            )
            response = self._batch_client.wait_for_completion(task_id, timeout)
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
        timeout: int = 300,
        disable_cache: bool = False,
        cancel_on_timeout: bool = True,
        **kwargs,
    ) -> TResponse | ESMProteinError:
        task_id = None
        task_timed_out = False
        keyboard_interrupted = False
        try:
            request = self._prepare_request(**kwargs)
            task_id = await self._batch_client.async_submit(
                self.endpoint_name, request, disable_cache=disable_cache
            )
            response = await self._batch_client.async_wait_for_completion(
                task_id, timeout
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
