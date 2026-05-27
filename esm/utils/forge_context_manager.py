import threading
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import copy_context
from typing import Any, Callable, List

from tqdm import tqdm

from esm.sdk.api import ESMProteinError
from esm.sdk.retry import retry_if_specific_error, skip_retries_var

TQDM_BAR_FORMAT = (
    "{desc:<12}{percentage:3.0f}%|{bar:24}| {n_fmt}/{total_fmt} "
    "[Elapsed: {elapsed} | Remaining: {remaining}] {postfix}"
)


class AIMDRateLimiter:
    """Rate limiter with AIMD (Additive Increase/Multiplicative Decrease) control."""

    def __init__(
        self,
        initial_concurrency: int = 32,
        min_concurrency: int = 1,
        max_concurrency: int = 64,
        step_up: int = 1,
    ):
        self.concurrency = initial_concurrency
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.step_up = step_up
        self._lock = threading.Lock()

    def adjust_concurrency(self, error_seen: bool) -> int:
        """Update concurrency based on if an error is seen."""
        with self._lock:
            if error_seen:
                self.concurrency = max(self.min_concurrency, self.concurrency // 2)
            else:
                self.concurrency = min(
                    self.max_concurrency, self.concurrency + self.step_up
                )
        return self.concurrency


class ForgeBatchExecutor:
    """Context manager for managing concurrent calls with rate limiting.

    Args:
        max_attempts: Maximum attempts per task before failing.
        max_workers: Maximum number of concurrent workers. Default is 64.
        show_progress: Whether to display a tqdm progress bar. Default ``True``.
    """

    def __init__(
        self, max_attempts: int = 10, max_workers: int = 64, show_progress: bool = True
    ):
        if max_workers > 64:
            raise ValueError("max_workers must be less than 64")
        self.rate_limiter = AIMDRateLimiter(max_concurrency=max_workers)
        self.max_attempts = max_attempts
        self.show_progress = show_progress
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._skip_retries_token = None

    def __enter__(self):
        self._skip_retries_token = skip_retries_var.set(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._skip_retries_token is not None:
            skip_retries_var.reset(self._skip_retries_token)
        if self._executor:
            self._executor.shutdown(wait=True)

    def _validate_inputs(self, args: Any, kwargs: Any) -> int:
        """Validate input lengths and return the number of tasks."""
        input_lengths = [len(v) for v in kwargs.values() if isinstance(v, list)]
        input_lengths.extend([len(v) for v in args if isinstance(v, list)])

        num_inputs = max(input_lengths) if input_lengths else 1

        if input_lengths and len(set(input_lengths)) > 1:
            raise ValueError("All list-valued arguments must have the same length")

        return num_inputs

    def execute_batch(
        self, user_func: Callable, *args: Any, **kwargs: Any
    ) -> List[Any]:
        """Call the endpoint with batched inputs, managing concurrency and retries."""
        num_tasks = self._validate_inputs(args, kwargs)
        # Initialize task queue with (task_index, attempt) tuples.
        task_queue = deque([(i, 1) for i in range(num_tasks)])
        results = [None] * num_tasks
        running_futures = {}

        success_count = 0
        fail_count = 0
        retry_count = 0

        with tqdm(
            total=num_tasks,
            desc="Processing",
            bar_format=TQDM_BAR_FORMAT,
            unit="task",
            disable=not self.show_progress,
        ) as pbar:
            while task_queue or running_futures:
                current_limit = self.rate_limiter.concurrency
                while task_queue and len(running_futures) < current_limit:
                    idx, attempt = task_queue.popleft()
                    call_kwargs = {
                        k: v[idx] if isinstance(v, list) else v
                        for k, v in kwargs.items()
                    }
                    call_args = [v[idx] if isinstance(v, list) else v for v in args]
                    ctx = copy_context()
                    future = self._executor.submit(
                        ctx.run, user_func, *call_args, **call_kwargs
                    )
                    running_futures[future] = (idx, attempt)

                done, _ = wait(
                    running_futures.keys(), return_when=FIRST_COMPLETED, timeout=1
                )
                error_seen = False
                for future in done:
                    idx, attempt = running_futures.pop(future)
                    try:
                        result = future.result()
                        if isinstance(result, ESMProteinError):
                            raise result
                        results[idx] = result
                        success_count += 1
                        pbar.update(1)
                    except Exception as e:
                        if retry_if_specific_error(e) and attempt < self.max_attempts:
                            task_queue.append((idx, attempt + 1))
                            # Only scale concurrency if hit rate limit errors.
                            if isinstance(e, ESMProteinError) and e.error_code == 429:
                                error_seen = True
                            retry_count += 1
                            pbar.update(0)
                        else:
                            results[idx] = e  # type: ignore
                            fail_count += 1
                            pbar.update(1)

                if len(done) > 0:
                    self.rate_limiter.adjust_concurrency(error_seen)
                pbar.set_postfix_str(
                    f"Success={success_count} Fail={fail_count} Retry={retry_count}"
                )

        return results
