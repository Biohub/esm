from typing import Any, Callable, Literal, TypedDict

from esm.sdk.api import ESM3InferenceClient


class ClientInitContainerMetadata(TypedDict):
    inference_option: Literal["Forge/Biohub Platform API", "Local"] | None


class ClientInitContainer:
    client_init_callback: Callable[[], ESM3InferenceClient] | None = None
    metadata: ClientInitContainerMetadata

    def __init__(self):
        self.metadata = ClientInitContainerMetadata(inference_option=None)

    def __call__(self, *args: Any, **kwds: Any) -> ESM3InferenceClient:
        if self.client_init_callback is None:
            raise ValueError("Client not initialized.")
        return self.client_init_callback()
