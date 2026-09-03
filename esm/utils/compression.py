"""Compressed wire form for a large field carried inside a request payload."""

import base64
import json
from typing import Any, Protocol

import zstd


class SupportsStateDict(Protocol):
    """The SDK's serialization convention, as used for the Forge/BHP wire form."""

    def state_dict(self, json_serializable: bool = False) -> dict[str, Any]: ...


def compress_state_dict(obj: SupportsStateDict) -> str:
    payload = json.dumps(obj.state_dict(json_serializable=True)).encode()
    return base64.b64encode(zstd.compress(payload)).decode()


def decompress_state_dict(blob: str) -> dict[str, Any]:
    return json.loads(zstd.decompress(base64.b64decode(blob)))
