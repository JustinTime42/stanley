"""Tests for LLM orchestrator service."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.services.llm_service import LLMOrchestrator
from src.config.llm_config import LLMConfig
from src.models.llm_models import LLMRequest, ModelProvider


class TestLLMOrchestrator:
    """Test suite for LLMOrchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create config with minimal setup
        self.config = LLMConfig()
        self.orchestrator = LLMOrchestrator(self.config)

    def test_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.config is not None
        assert self.orchestrator.analyzer is not None
        assert self.orchestrator.router is not None
        assert self.orchestrator.fallback_manager is not None

        # Should have providers initialized
        assert len(self.orchestrator.providers) > 0

    def test_get_available_models(self):
        """Test getting available models."""
        models = self.orchestrator.get_available_models()

        assert isinstance(models, list)
        assert len(models) > 0

        # Should include qwen2.5-coder:14b
        assert "qwen2.5-coder:14b" in models

    def test_get_cache_stats_without_cache(self):
        """Test cache stats when cache is disabled."""
        # Create orchestrator without cache
        config = LLMConfig()
        config.enable_cache = False
        orchestrator = LLMOrchestrator(config)

        stats = orchestrator.get_cache_stats()
        assert stats["enabled"] is False

    def test_get_cache_stats_with_cache(self):
        """Test cache stats when cache is enabled."""
        stats = self.orchestrator.get_cache_stats()

        assert stats["enabled"] is True
        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "cost_savings" in stats

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup method."""
        # Should not raise errors
        await self.orchestrator.cleanup()

    def test_provider_initialization_without_api_keys(self):
        """Test that providers handle missing API keys gracefully."""
        # Create config without API keys
        config = LLMConfig()
        config.openai_api_key = None
        config.openrouter_api_key = None

        orchestrator = LLMOrchestrator(config)

        # Should still initialize with Ollama provider
        models = orchestrator.get_available_models()
        assert "qwen2.5-coder:14b" in models


class TestLLMOrchestratorRouting:
    """Test routing behavior of orchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = LLMConfig()
        self.orchestrator = LLMOrchestrator(self.config)

    def test_routing_decision_for_simple_task(self):
        """Test that simple tasks are routed appropriately."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Say hello"}],
            agent_role="test",
            task_description="Simple greeting task",
        )

        decision = self.orchestrator.router.route_request(request)

        # Should select a model
        assert decision.selected_model is not None
        assert decision.routing_reason is not None

        # Should have selected some model (local or cloud based on routing strategy)
        assert decision.selected_model.model_name is not None

    def test_routing_decision_with_complexity_override(self):
        """Test routing with explicit complexity override."""
        from src.models.llm_models import TaskComplexity

        request = LLMRequest(
            messages=[{"role": "user", "content": "Design architecture"}],
            agent_role="architect",
            task_description="Complex architectural design",
            complexity_override=TaskComplexity.COMPLEX,
        )

        decision = self.orchestrator.router.route_request(request)

        # Should select high-performance model for complex tasks
        assert decision.selected_model.performance_score >= 0.8

    def test_fallback_chain_creation(self):
        """Test that fallback chains are created properly."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Test"}],
            agent_role="test",
            task_description="Test",
        )

        decision = self.orchestrator.router.route_request(request)
        chain = self.orchestrator._build_fallback_chain(decision)

        # Should have at least primary model
        assert len(chain) >= 1

        # First should be selected model
        assert chain[0].config.model_name == decision.selected_model.model_name


class TestLLMOrchestratorCaching:
    """Test caching behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = LLMConfig()
        self.config.enable_cache = True
        self.orchestrator = LLMOrchestrator(self.config)

    def test_cache_enabled(self):
        """Test that cache is enabled when configured."""
        assert self.orchestrator.cache is not None

    def test_cache_disabled(self):
        """Test that cache can be disabled."""
        config = LLMConfig()
        config.enable_cache = False
        orchestrator = LLMOrchestrator(config)

        assert orchestrator.cache is None
