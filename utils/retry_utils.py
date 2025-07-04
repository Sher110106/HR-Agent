"""Shared retry utilities with exponential back-off.

Used by multiple components (column analysis, execution retries, etc.) to avoid
code duplication and drift across different retry implementations.
"""
from __future__ import annotations

import time
import logging
from typing import Callable, Type, Tuple

logger = logging.getLogger(__name__)


def perform_with_retries(
    func: Callable[..., Tuple[str, str]],
    *args,
    max_retries: int = 2,
    base_delay: float = 1.0,
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    **kwargs,
):
    """Execute *func* with retries and exponential back-off.

    Parameters
    ----------
    func
        Callable to execute. Must return whatever the caller expects.
    *args, **kwargs
        Passed straight to *func*.
    max_retries
        How many **additional** attempts after the first call.
    base_delay
        Initial sleep in seconds before the first retry. Delay doubles each retry.
    retry_exceptions
        Tuple of exception types that trigger a retry.
    """
    attempt = 0
    delay = base_delay
    while True:
        try:
            return func(*args, **kwargs)
        except retry_exceptions as exc:
            if attempt >= max_retries:
                logger.error("❌ All %s retries exhausted for %s: %s", max_retries, func.__name__, exc)
                raise
            attempt += 1
            logger.warning("⚠️  %s failed (attempt %s/%s): %s – retrying in %.1fs", func.__name__, attempt, max_retries, exc, delay)
            time.sleep(delay)
            delay *= 2  # exponential back-off 