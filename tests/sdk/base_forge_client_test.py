"""The batch client's polling backoff and per-endpoint wait defaults."""

import itertools

import pytest

from esm.sdk import base_forge_client
from esm.sdk.api import ESMProteinError
from esm.sdk.base_forge_client import (
    DEFAULT_POLL_INTERVAL,
    EndpointHandler,
    _poll_intervals,
)
from esm.sdk.forge import ForgeBatchClient


class _QuietHandler(EndpointHandler[dict]):
    """An endpoint that states no pacing of its own, so the layers above it decide."""

    @property
    def endpoint_name(self) -> str:
        return "quiet"

    def _prepare_request(self, **kwargs) -> list[dict]:
        return []

    def _process_response(self, response: dict, **kwargs) -> dict:
        return response

    async def _async_process_response(self, response: dict, **kwargs) -> dict:
        return response


@pytest.fixture
def client() -> ForgeBatchClient:
    return ForgeBatchClient(url="http://localhost", token="unused")


def quiet(client: ForgeBatchClient) -> _QuietHandler:
    return _QuietHandler(client._batch_client)


def test_the_staircase_holds_each_rung_then_doubles_up_to_the_ceiling():
    got = list(itertools.islice(_poll_intervals(2, 30), 15))
    assert got == [2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 30, 30, 30]


def test_an_endpoint_paces_itself_when_the_user_says_nothing(client):
    assert client._batch_client.poll_interval is None
    assert client.fold_max_accuracy._resolved_polling() == (10, 60)
    assert quiet(client)._resolved_polling() == (DEFAULT_POLL_INTERVAL, 30)


def test_the_users_interval_beats_the_endpoints_own():
    client = ForgeBatchClient(url="http://localhost", token="unused", poll_interval=45)
    assert client.fold_max_accuracy.poll_interval == 10, (
        "the endpoint still asks for 10"
    )
    assert client.fold_max_accuracy._resolved_polling() == (45, 60)
    assert quiet(client)._resolved_polling() == (45, 45)


def test_the_ceiling_never_clamps_the_interval_below_what_was_asked_for():
    """Asking for 90s against `fold_max_accuracy`'s ceiling of 60 has to mean 90."""
    client = ForgeBatchClient(url="http://localhost", token="unused", poll_interval=90)
    assert client.fold_max_accuracy._resolved_polling() == (90, 90)
    assert quiet(client)._resolved_polling() == (90, 90)


def test_a_ceiling_above_the_interval_is_what_the_backoff_grows_into():
    client = ForgeBatchClient(
        url="http://localhost", token="unused", poll_interval=90, poll_max_interval=120
    )
    assert client.fold_max_accuracy._resolved_polling() == (90, 120)


def test_a_user_ceiling_applies_without_touching_the_endpoints_interval():
    client = ForgeBatchClient(
        url="http://localhost", token="unused", poll_max_interval=15
    )
    assert client.fold_max_accuracy._resolved_polling() == (10, 15)


def test_a_per_call_interval_beats_every_standing_setting(client):
    """Named on `run` rather than left to `**kwargs`, which forwards to the payload."""
    configured = ForgeBatchClient(
        url="http://localhost", token="unused", poll_interval=45
    )
    assert client.fold_max_accuracy._resolved_polling(poll_interval=5) == (5, 60)
    assert configured.fold_max_accuracy._resolved_polling(poll_interval=90) == (90, 90)
    assert client.fold_max_accuracy._resolved_polling(
        poll_interval=90, poll_max_interval=100
    ) == (90, 100)


def test_a_ceiling_below_its_interval_from_the_same_caller_is_rejected(client):
    with pytest.raises(ValueError, match="poll_max_interval"):
        ForgeBatchClient(
            url="http://localhost",
            token="unused",
            poll_interval=200,
            poll_max_interval=120,
        )
    with pytest.raises(ValueError, match="below poll_interval"):
        client.fold_max_accuracy._resolved_polling(
            poll_interval=200, poll_max_interval=120
        )
    with pytest.raises(ValueError, match="below poll_interval"):
        client._batch_client.wait_for_completion(
            "task-9", timeout=10, poll_interval=200, poll_max_interval=120
        )


class _FakeClock:
    """A monotonic clock that only advances when something sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(base_forge_client.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(base_forge_client.time, "sleep", fake.sleep)
    return fake


def _client_returning(statuses: list[dict]) -> ForgeBatchClient:
    """A client whose `get_status` walks `statuses`, then repeats the last one."""
    client = ForgeBatchClient(url="http://localhost", token="unused")
    remaining = list(statuses)

    def get_status(task_id: str) -> dict:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    client._batch_client.get_status = get_status
    return client


def test_a_sleep_never_overshoots_the_deadline(clock):
    client = _client_returning([{"status": "in_progress"}])
    with pytest.raises(ESMProteinError, match="timed out"):
        client._batch_client.wait_for_completion(
            "task-1", timeout=10, poll_interval=60, poll_max_interval=60
        )
    assert clock.sleeps == [10]


def test_a_job_finishing_during_the_final_sleep_is_returned(clock):
    """Waking past the deadline must not report a timeout and cancel a finished job."""
    client = _client_returning([{"status": "in_progress"}, {"status": "done"}])
    response = client._batch_client.wait_for_completion(
        "task-2", timeout=5, poll_interval=5, poll_max_interval=5
    )
    assert response == {"status": "done"}
    assert clock.sleeps == [5]


def test_a_terminal_status_is_returned_without_sleeping(clock):
    client = _client_returning([{"status": "done"}])
    assert client._batch_client.wait_for_completion("task-3", timeout=600) == {
        "status": "done"
    }
    assert clock.sleeps == []


def test_a_failed_job_raises_with_the_server_reason(clock):
    client = _client_returning([{"status": "failed", "error": "CUDA out of memory"}])
    with pytest.raises(ESMProteinError, match="CUDA out of memory"):
        client._batch_client.wait_for_completion("task-4", timeout=600)


def _stub_submission(
    monkeypatch: pytest.MonkeyPatch, handler: EndpointHandler, task_id: str
) -> None:
    monkeypatch.setattr(handler, "_prepare_request", lambda **kwargs: [])
    monkeypatch.setattr(handler._batch_client, "submit", lambda *a, **kw: task_id)


def test_the_users_interval_reaches_the_wait_loop(clock, monkeypatch):
    client = _client_returning([{"status": "in_progress"}])
    client._batch_client.poll_interval = 45
    handler = client.fold_max_accuracy
    _stub_submission(monkeypatch, handler, "task-7")

    result = handler.run(timeout=90, cancel_on_timeout=False)

    assert isinstance(result, ESMProteinError)
    assert clock.sleeps == [45, 45]


def test_a_per_call_interval_reaches_the_wait_loop(clock, monkeypatch):
    client = _client_returning([{"status": "in_progress"}])
    client._batch_client.poll_interval = 45
    handler = client.fold_max_accuracy
    _stub_submission(monkeypatch, handler, "task-8")

    result = handler.run(timeout=60, cancel_on_timeout=False, poll_interval=20)

    assert isinstance(result, ESMProteinError)
    assert clock.sleeps == [20, 20, 20]
