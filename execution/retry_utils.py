"""
Retry utilities with exponential backoff for resilient API calls
"""

import time
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type

from .logging_config import get_logger

logger = get_logger()


def retry_with_exponential_backoff(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_exponential_backoff(max_attempts=3, initial_wait=2.0)
        def call_api():
            return api.call()
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        # Last attempt failed, raise the exception
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {str(e)}"
                        )
                        raise

                    # Calculate wait time with exponential backoff
                    wait_time = min(
                        initial_wait * (exponential_base**attempt), max_wait
                    )

                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )

                    time.sleep(wait_time)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


class RetryableError(Exception):
    """Base exception for errors that should trigger a retry"""

    pass


class NonRetryableError(Exception):
    """Base exception for errors that should NOT trigger a retry"""

    pass
