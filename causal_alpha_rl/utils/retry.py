"""Retry helpers for transient failures."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def retry_call(
    func: Callable[[], T],
    *,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    transient_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return func()
        except transient_exceptions as exc:  # pragma: no cover - exercised in integration flows
            last_error = exc
            if attempt == retries:
                break
            time.sleep(backoff_seconds * attempt)
    assert last_error is not None
    raise last_error

