"""Durable identifiers and errors for the Batch API."""

from dataclasses import dataclass
from typing import Any

from esm.sdk.api import ESMProteinError


@dataclass(frozen=True)
class BatchSubmission:
    """A durable handle; task IDs contain only identifiers returned by the server."""

    endpoint: str
    task_ids: tuple[str, ...] = ()
    batch_id: str | None = None

    def __post_init__(self) -> None:
        if not self.batch_id and not self.task_ids:
            raise ValueError("A submission must contain a batch ID or task IDs")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("Task IDs must be unique")

    @property
    def task_id(self) -> str:
        """Return the task ID of a single-task submission."""
        if len(self.task_ids) != 1:
            raise ValueError(
                "This submission does not contain exactly one known task ID; "
                "use list_batch() to discover batch tasks"
            )
        return self.task_ids[0]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable identifiers and decoding context."""
        return {
            "endpoint": self.endpoint,
            "batch_id": self.batch_id,
            "task_ids": list(self.task_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchSubmission":
        """Restore a submission saved by :meth:`to_dict`."""
        return cls(
            endpoint=data["endpoint"],
            batch_id=data["batch_id"],
            task_ids=tuple(data["task_ids"]),
        )


class BatchError(ESMProteinError):
    """An error retaining the accepted submission and any collected partial results.

    ``partial_results`` contains ``(task_id, output)`` pairs collected before
    ``results()`` or ``run()`` failed. Outputs include per-task errors.
    Streaming methods leave this list empty; callers retain the yielded outputs.
    """

    def __init__(
        self,
        error_code: int,
        error_msg: str,
        submission: BatchSubmission | None = None,
        task_id: str | None = None,
        *,
        task_status: str | None = None,
    ) -> None:
        super().__init__(error_code, error_msg)
        self.submission = submission
        self.task_id = task_id
        self.task_status = task_status
        self.partial_results: list[tuple[str, Any]] = []
        self.cancellation: dict[str, Any] | None = None
        self.cancellation_error: ESMProteinError | None = None

    def __reduce__(self) -> tuple[Any, ...]:
        # Timeout constructors differ from Exception.args; attrs also omits our metadata.
        return (
            _restore_batch_error,
            (type(self), self.error_code, self.error_msg, self.__dict__),
        )

    @property
    def batch_id(self) -> str | None:
        return self.submission.batch_id if self.submission else None

    @property
    def task_ids(self) -> tuple[str, ...]:
        return self.submission.task_ids if self.submission else ()

    @classmethod
    def from_error(
        cls,
        error: ESMProteinError,
        submission: BatchSubmission | None = None,
        task_id: str | None = None,
    ) -> "BatchError":
        if isinstance(error, BatchError):
            if submission is not None:
                error.submission = submission
            if task_id is not None:
                error.task_id = task_id
            return error
        return cls(error.error_code, error.error_msg, submission, task_id)


class BatchTimeoutError(BatchError):
    """The caller stopped waiting; the saved submission remains recoverable."""

    def __init__(self, submission: BatchSubmission, timeout: float) -> None:
        identifier = submission.batch_id or ", ".join(submission.task_ids)
        super().__init__(
            408,
            f"Job {identifier} timed out after {timeout} seconds.",
            submission=submission,
            task_id=submission.task_ids[0] if len(submission.task_ids) == 1 else None,
        )


def _restore_batch_error(
    error_type: type[BatchError], error_code: int, error_msg: str, state: dict[str, Any]
) -> BatchError:
    error = error_type.__new__(error_type)
    ESMProteinError.__init__(error, error_code, error_msg)
    error.__dict__.update(state)
    return error
