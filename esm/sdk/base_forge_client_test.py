import httpx
import pytest

from esm.sdk.base_forge_client import _BaseForgeInferenceClient, _sdk_version

BIOHUB_COMPONENT = f"biohub/forge-sdk-python/{_sdk_version()}"


def _client() -> _BaseForgeInferenceClient:
    return _BaseForgeInferenceClient(
        model="esmc-300m-2024-12",
        url="https://biohub.ai",
        token="test-token",
        request_timeout=1,
        min_retry_wait=1,
        max_retry_wait=1,
        max_retry_attempts=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("user_agent", [None, "custom-transport/1.0"])
async def test_posts_component_and_preserves_transport_user_agent(
    asynchronous: bool, user_agent: str | None
) -> None:
    client = _client()
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"outputs": {}})

    transport = httpx.MockTransport(handle_request)
    transport_headers = {} if user_agent is None else {"User-Agent": user_agent}
    override = {"x-biohub-component": f"biohub/forge-sdk-python-test/{_sdk_version()}"}
    request_headers = [{}, override, {}]

    if asynchronous:
        async with httpx.AsyncClient(
            transport=transport, headers=transport_headers
        ) as http_client:
            client._async_client = http_client
            for headers in request_headers:
                assert await client._async_post("fold", {}, headers=headers) == {
                    "outputs": {}
                }
    else:
        with httpx.Client(
            transport=transport, headers=transport_headers
        ) as http_client:
            client._client = http_client
            for headers in request_headers:
                assert client._post("fold", {}, headers=headers) == {"outputs": {}}

    assert [request.headers.get_list("x-biohub-component") for request in requests] == [
        [BIOHUB_COMPONENT],
        [override["x-biohub-component"]],
        [BIOHUB_COMPONENT],
    ]
    for request in requests:
        assert request.headers["authorization"] == "Bearer test-token"
        assert request.headers.get_list("user-agent") == [
            user_agent or f"python-httpx/{httpx.__version__}"
        ]

    assert client.headers == {
        "Authorization": "Bearer test-token",
        "X-Biohub-Component": BIOHUB_COMPONENT,
    }
    assert override == {
        "x-biohub-component": f"biohub/forge-sdk-python-test/{_sdk_version()}"
    }
