"""Tests for fallback chain manager."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.llm.fallback import FallbackChainManager
from src.llm.base import BaseLLM, RateLimitError, APIError
from src.models.llm_models import (
    LLMRequest,
    LLMResponse,
    ModelConfig,
    ModelProvider,
    ModelCapability,
)


class MockProvider(BaseLLM):
    """Mock LLM provider for testing."""

    def __init__(self, name: str, should_fail: bool = False, fail_count: int = 0):
        """Initialize mock provider."""
        config = ModelConfig(
            provider=ModelProvider.OLLAMA,
            model_name=name,
            context_window=8192,
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            capabilities=[ModelCapability.GENERAL],
            is_local=True,
        )
        super().__init__(config)
        self.should_fail = should_fail
        self.fail_count = fail_count
        self.attempt_count = 0

    async def agenerate(self, messages, max_tokens=None, temperature=0.7, **kwargs):
        """Mock generation."""
        import asyncio

        self.attempt_count += 1

        if self.should_fail:
            if self.fail_count > 0 and self.attempt_count <= self.fail_count:
                raise APIError(f"Mock failure on attempt {self.attempt_count}")
            elif self.fail_count == 0:
                raise APIError("Mock always fails")

        # Add small delay to simulate actual LLM call and ensure measurable latency
        await asyncio.sleep(0.001)  # 1ms delay

        return f"Response from {self.config.model_name}"

    async def astream(self, messages, max_tokens=None, temperature=0.7, **kwargs):
        """Mock streaming."""
        yield "chunk1"
        yield "chunk2"

    def get_num_tokens(self, text: str) -> int:
        """Mock token counting."""
        return len(text.split())


class TestFallbackChainManager:
    """Test suite for FallbackChainManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = FallbackChainManager(
            max_retries=3,
            base_delay=0.1,  # Fast for testing
            max_delay=1.0,
        )

    @pytest.mark.asyncio
    async def test_successful_first_provider(self):
        """Test successful execution with first provider."""
        provider = MockProvider("test-model")

        request = LLMRequest(
            messages=[{"role": "user", "content": "Test"}],
            agent_role="test",
            task_description="Test task",
        )

        response = await self.manager.execute_with_fallback(
            request=request,
            providers=[provider],
        )

        assert response.content == "Response from test-model"
        assert response.model_used == "test-model"
        assert response.fallback_used is False

    @pytest.mark.asyncio
    async def test_fallback_to_second_provider(self):
        """Test fallback to second provider when first fails."""
        provider1 = MockProvider("model1", should_fail=True)
        provider2 = MockProvider("model2", should_fail=False)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Test"}],
            agent_role="test",
            task_description="Test task",
        )

        response = await self.manager.execute_with_fallback(
            request=request,
            providers=[provider1, provider2],
        )

        assert response.content == "Response from model2"
        assert response.model_used == "model2"
        assert response.fallback_used is True

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test retry logic on rate limit errors."""
        # Create provider (no failure flags - mock wrapper handles failure logic)
        provider = MockProvider("test-model", should_fail=False)

        # Mock RateLimitError
        original_generate = provider.agenerate
        call_count = [0]

        async def mock_generate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RateLimitError("Rate limited")
            return await original_generate(*args, **kwargs)

        provider.agenerate = mock_generate

        request = LLMRequest(
            messages=[{"role": "user", "content": "Test"}],
            agent_role="test",
            task_description="Test task",
        )

        response = await self.manager.execute_with_fallback(
            request=request,
            providers=[provider],
        )

        assert call_count[0] == 3  # Failed twice, succeeded third time
        assert response.content == "Response from test-model"

    @pytest.mark.asyncio
    async def test_all_providers_fail(self):
        """Test that exception is raised when all providers fail."""
        provider1 = MockProvider("model1", should_fail=True)
        provider2 = MockProvider("model2", should_fail=True)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Test"}],
            agent_role="test",
            task_description="Test task",
        )

        with pytest.raises(Exception) as exc_info:
            await self.manager.execute_with_fallback(
                request=request,
                providers=[provider1, provider2],
            )

        assert "All 2 providers failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_streaming_with_fallback(self):
        """Test streaming with fallback."""
        provider1 = MockProvider("model1", should_fail=True)
        provider2 = MockProvider("model2", should_fail=False)

        request = LLMRequest(
            messages=[{"role": "user", "content": "Test"}],
            agent_role="test",
            task_description="Test task",
        )

        chunks = []
        async for chunk in self.manager.execute_streaming_with_fallback(
            request=request,
            providers=[provider1, provider2],
        ):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks == ["chunk1", "chunk2"]

    def test_create_fallback_chain(self):
        """Test fallback chain creation."""
        provider1 = MockProvider("model1")
        provider2 = MockProvider("model2")
        provider3 = MockProvider("model3")

        chain = self.manager.create_fallback_chain(
            primary=provider1,
            fallbacks=[provider2, provider3],
        )

        assert len(chain) == 3
        assert chain[0].config.model_name == "model1"
        assert chain[1].config.model_name == "model2"
        assert chain[2].config.model_name == "model3"

    def test_success_rate_estimation(self):
        """Test success rate estimation."""
        provider1 = MockProvider("model1")
        provider2 = MockProvider("model2")

        # Test with default success rates
        success_rate = self.manager.estimate_success_rate(
            providers=[provider1, provider2]
        )

        # With 2 providers at 95% success each + retries
        # Should be very high
        assert success_rate > 0.95

        # Test with custom success rates
        custom_rates = {
            "model1": 0.7,
            "model2": 0.8,
        }

        success_rate = self.manager.estimate_success_rate(
            providers=[provider1, provider2],
            provider_success_rates=custom_rates,
        )

        assert 0.9 < success_rate < 1.0

    @pytest.mark.asyncio
    async def test_no_providers_error(self):
        """Test error when no providers given."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Test"}],
            agent_role="test",
            task_description="Test task",
        )

        with pytest.raises(ValueError, match="No providers available"):
            await self.manager.execute_with_fallback(
                request=request,
                providers=[],
            )

    @pytest.mark.asyncio
    async def test_latency_tracking(self):
        """Test that latency is tracked in response."""
        provider = MockProvider("test-model")

        request = LLMRequest(
            messages=[{"role": "user", "content": "Test"}],
            agent_role="test",
            task_description="Test task",
        )

        response = await self.manager.execute_with_fallback(
            request=request,
            providers=[provider],
        )

        # Latency should be tracked
        assert response.latency_ms > 0
        assert isinstance(response.latency_ms, int)

    @pytest.mark.asyncio
    async def test_cost_calculation(self):
        """Test that cost is calculated correctly."""
        provider = MockProvider("test-model")

        request = LLMRequest(
            messages=[{"role": "user", "content": "This is a test message"}],
            agent_role="test",
            task_description="Test task",
        )

        response = await self.manager.execute_with_fallback(
            request=request,
            providers=[provider],
        )

        # Cost should be 0 for local model
        assert response.total_cost == 0.0
        assert response.input_tokens > 0
        assert response.output_tokens > 0
