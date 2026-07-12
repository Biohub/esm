"""Tests for retry.py"""

from esm.sdk.api import ESMProteinError
from esm.sdk.retry import retry_if_specific_error


def test_retry_if_specific_error_retries_transient_statuses():
    # Rate-limit and transient server errors are retried. 503 Service
    # Unavailable must be retried alongside its 500/502/504 siblings.
    for code in [429, 500, 502, 503, 504]:
        err = ESMProteinError(error_code=code, error_msg="transient")
        assert retry_if_specific_error(err), f"HTTP {code} should be retryable"


def test_retry_if_specific_error_skips_non_transient_and_non_esm():
    # A non-transient client error is not retried.
    assert not retry_if_specific_error(
        ESMProteinError(error_code=404, error_msg="not found")
    )
    # Exceptions that are not ESMProteinError are not retried.
    assert not retry_if_specific_error(ValueError("boom"))
