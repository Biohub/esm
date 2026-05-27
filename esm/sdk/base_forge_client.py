from typing import Any
from urllib.parse import urljoin

import httpx

from esm.sdk.api import ESMProteinError
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
