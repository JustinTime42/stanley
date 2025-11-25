"""Tests for retry logic."""

import pytest
import asyncio

from src.tools.retry import RetryManager, with_retry


class RetryCounter:
    """Helper class to count retry attempts."""

    def __init__(self, fail_count: int = 2):
        self.attempts = 0
        self.fail_count = fail_count

    async def failing_function(self):
        """Function that fails a certain number of times."""
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise Exception(f"Attempt {self.attempts} failed")
        return f"Success on attempt {self.attempts}"


@pytest.mark.asyncio
async def test_retry_manager_success():
    """Test retry manager with eventual success."""
    manager = RetryManager(max_retries=3, backoff_factor=0.1)
    counter = RetryCounter(fail_count=2)

    result = await manager.execute_with_retry(
        counter.failing_function,
        retry_on=(Exception,),
    )

    assert result == "Success on attempt 3"
    assert counter.attempts == 3


@pytest.mark.asyncio
async def test_retry_manager_all_failures():
    """Test retry manager when all attempts fail."""
    manager = RetryManager(max_retries=2, backoff_factor=0.1)
    counter = RetryCounter(fail_count=10)  # Fail more times than retries

    with pytest.raises(Exception) as exc_info:
        await manager.execute_with_retry(
            counter.failing_function,
            retry_on=(Exception,),
        )

    assert "Attempt" in str(exc_info.value)
    assert counter.attempts == 2


@pytest.mark.asyncio
async def test_retry_decorator_success():
    """Test retry decorator with eventual success."""
    counter = RetryCounter(fail_count=1)

    @with_retry(max_retries=3, backoff_factor=0.1)
    async def decorated_function():
        return await counter.failing_function()

    result = await decorated_function()
    assert result == "Success on attempt 2"
    assert counter.attempts == 2


@pytest.mark.asyncio
async def test_retry_delay_calculation():
    """Test exponential backoff delay calculation."""
    manager = RetryManager(max_retries=5, backoff_factor=2.0, max_delay=10.0)

    # Test delay calculations
    assert manager.calculate_delay(0) == 1.0  # 2^0
    assert manager.calculate_delay(1) == 2.0  # 2^1
    assert manager.calculate_delay(2) == 4.0  # 2^2
    assert manager.calculate_delay(3) == 8.0  # 2^3
    assert manager.calculate_delay(4) == 10.0  # 2^4 = 16, capped at max_delay
