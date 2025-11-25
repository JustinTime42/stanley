"""Tests for LLM response cache."""

import pytest
from datetime import datetime
from src.llm.cache import LLMResponseCache
from src.models.llm_models import LLMResponse, ModelProvider


class TestLLMResponseCache:
    """Test suite for LLMResponseCache."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache = LLMResponseCache(max_size=10, default_ttl=3600)

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss returns None."""
        result = await self.cache.get("nonexistent_key")
        assert result is None
        assert self.cache.hits == 0
        assert self.cache.misses == 1

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test setting and getting cached responses."""
        response = LLMResponse(
            content="Test response",
            model_used="test-model",
            provider=ModelProvider.OPENAI,
            input_tokens=10,
            output_tokens=20,
            total_cost=0.001,
            latency_ms=100,
        )

        cache_key = "test_key_123"

        # Set response
        await self.cache.set(cache_key, response)

        # Get response
        cached_response = await self.cache.get(cache_key)

        assert cached_response is not None
        assert cached_response.content == "Test response"
        assert cached_response.cache_hit is True
        assert self.cache.hits == 1

    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        for i in range(12):  # Max size is 10
            response = LLMResponse(
                content=f"Response {i}",
                model_used="test",
                provider=ModelProvider.OPENAI,
                input_tokens=10,
                output_tokens=10,
                total_cost=0.001,
                latency_ms=100,
            )
            await self.cache.set(f"key_{i}", response)

        # Cache should not exceed max size
        assert len(self.cache.cache) == 10
        assert self.cache.evictions == 2

    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.cache.get_stats()

        assert "size" in stats
        assert "max_size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        # Initially 0
        assert self.cache.get_hit_rate() == 0.0

        # Add some hits and misses
        self.cache.hits = 7
        self.cache.misses = 3

        # 70% hit rate
        assert self.cache.get_hit_rate() == 0.7

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test clearing cache."""
        # Add some entries
        for i in range(5):
            response = LLMResponse(
                content=f"Response {i}",
                model_used="test",
                provider=ModelProvider.OPENAI,
                input_tokens=10,
                output_tokens=10,
                total_cost=0.001,
                latency_ms=100,
            )
            await self.cache.set(f"key_{i}", response)

        assert len(self.cache.cache) == 5

        # Clear cache
        self.cache.clear()

        assert len(self.cache.cache) == 0

    @pytest.mark.asyncio
    async def test_cost_savings_estimation(self):
        """Test cost savings estimation."""
        response = LLMResponse(
            content="Test",
            model_used="test",
            provider=ModelProvider.OPENAI,
            input_tokens=100,
            output_tokens=100,
            total_cost=0.01,
            latency_ms=100,
        )

        cache_key = "test_savings"
        await self.cache.set(cache_key, response)

        # Access multiple times
        for _ in range(5):
            await self.cache.get(cache_key)

        # Should show cost savings
        savings = self.cache.estimate_cost_savings()
        assert savings > 0
