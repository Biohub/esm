import pytest

from esm.sdk import batch_executor, parallel_executor
from esm.utils.forge_context_manager import (
    ForgeBatchExecutor,
    ForgeParallelExecutor,
)


def _double(x: int) -> int:
    return 2 * x


def test_parallel_executor_runs_every_input():
    with parallel_executor(show_progress=False) as executor:
        results = executor.execute_batch(user_func=_double, x=[1, 2, 3])
    assert results == [2, 4, 6]


def test_parallel_executor_returns_the_executor_class():
    with parallel_executor(show_progress=False) as executor:
        assert isinstance(executor, ForgeParallelExecutor)


def test_batch_executor_warns_and_still_works():
    with pytest.warns(DeprecationWarning, match="use parallel_executor instead"):
        ctx = batch_executor(show_progress=False)
    with ctx as executor:
        results = executor.execute_batch(user_func=_double, x=[1, 2, 3])
    assert results == [2, 4, 6]


def test_forge_batch_executor_alias_points_at_the_new_class():
    assert ForgeBatchExecutor is ForgeParallelExecutor
