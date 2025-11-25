"""Retry logic with exponential backoff for tool execution."""

import asyncio
import logging
from typing import Callable, TypeVar, Optional
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    retry_on: tuple = (Exception,),
    max_delay: float = 30.0,
):
    """
    Decorator for automatic retry with exponential backoff.

    PATTERN: Decorator pattern for retry logic
    CRITICAL: Must handle async functions
    GOTCHA: Backoff delay is capped at max_delay

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier
        retry_on: Tuple of exception types to retry on
        max_delay: Maximum delay between retries (seconds)

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_error: Optional[Exception] = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)

                except retry_on as e:
                    last_error = e

                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff
                        delay = min(backoff_factor**attempt, max_delay)

                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )

                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed for {func.__name__}: {e}"
                        )

            # All retries exhausted
            if last_error:
                raise last_error

            raise Exception(f"Unexpected error in retry logic for {func.__name__}")

        return wrapper

    return decorator


class RetryManager:
    """
    Retry manager with configurable retry logic.

    PATTERN: Manager class for retry handling
    CRITICAL: Supports async operations
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        max_delay: float = 30.0,
    ):
        """
        Initialize retry manager.

        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            max_delay: Maximum delay between retries (seconds)
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.logger = logging.getLogger(__name__)

    async def execute_with_retry(
        self, func: Callable[..., T], *args, retry_on: tuple = (Exception,), **kwargs
    ) -> T:
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments
            retry_on: Tuple of exception types to retry on
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries fail
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                result = await func(*args, **kwargs)
                return result

            except retry_on as e:
                last_error = e

                if attempt < self.max_retries - 1:
                    # Calculate delay with exponential backoff
                    delay = min(self.backoff_factor**attempt, self.max_delay)

                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    await asyncio.sleep(delay)
                    continue

        # All retries exhausted
        if last_error:
            self.logger.error(f"All {self.max_retries} attempts failed: {last_error}")
            raise last_error

        raise Exception("Unexpected error in retry logic")

    def should_retry(self, error: Exception, retry_on: tuple) -> bool:
        """
        Check if error should trigger a retry.

        Args:
            error: Exception that occurred
            retry_on: Tuple of exception types to retry on

        Returns:
            True if should retry
        """
        return isinstance(error, retry_on)

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate retry delay for given attempt.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        return min(self.backoff_factor**attempt, self.max_delay)
