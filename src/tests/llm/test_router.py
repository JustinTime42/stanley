"""Tests for model router."""

import pytest
from src.llm.router import ModelRouter
from src.llm.analyzer import TaskComplexityAnalyzer
from src.models.llm_models import (
    LLMRequest,
    ModelConfig,
    ModelProvider,
    ModelCapability,
    TaskComplexity,
)
from src.models.routing_models import TaskAnalysis


class TestModelRouter:
    """Test suite for ModelRouter."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock model configs
        self.models = [
            ModelConfig(
                provider=ModelProvider.OLLAMA,
                model_name="llama3.2",
                context_window=8192,
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                capabilities=[ModelCapability.GENERAL, ModelCapability.CODE_GENERATION],
                performance_score=0.6,
                is_local=True,
            ),
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o-mini",
                context_window=128000,
                cost_per_1k_input=0.00015,
                cost_per_1k_output=0.0006,
                capabilities=[
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.PLANNING,
                    ModelCapability.DEBUGGING,
                ],
                performance_score=0.85,
                is_local=False,
                supports_functions=True,
            ),
            ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o",
                context_window=128000,
                cost_per_1k_input=0.0025,
                cost_per_1k_output=0.01,
                capabilities=[
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.PLANNING,
                    ModelCapability.DEBUGGING,
                ],
                performance_score=0.95,
                is_local=False,
                supports_functions=True,
            ),
        ]

        self.router = ModelRouter(
            available_models=self.models,
            prefer_local=True,
            routing_strategy="cost_optimized",
        )

    def test_simple_task_routes_to_local(self):
        """Test that simple tasks route to local models."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Fix a typo"}],
            agent_role="implementer",
            task_description="Fix typo in code",
        )

        decision = self.router.route_request(request)

        # Should route to local model for simple task
        assert decision.selected_model.is_local
        assert decision.estimated_cost == 0.0

    def test_complex_task_routes_to_premium(self):
        """Test that complex tasks route to premium models."""
        task_analysis = TaskAnalysis(
            task_id="test_123",
            complexity=TaskComplexity.COMPLEX,
            estimated_tokens=1000,
            required_capabilities=[ModelCapability.PLANNING],
            confidence=0.9,
            reasoning="Complex architectural task",
        )

        request = LLMRequest(
            messages=[{"role": "user", "content": "Design system architecture"}],
            agent_role="architect",
            task_description="Design comprehensive system",
        )

        decision = self.router.route_request(request, task_analysis)

        # Should route to high-performance model
        assert decision.selected_model.performance_score >= 0.8

    def test_fallback_models_provided(self):
        """Test that fallback models are included in routing decision."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Write a function"}],
            agent_role="implementer",
            task_description="Implement feature",
        )

        decision = self.router.route_request(request)

        # Should have fallback models
        assert len(decision.fallback_models) > 0

    def test_cache_key_generation(self):
        """Test that cache keys are generated."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Test"}],
            agent_role="test",
            task_description="Test",
            use_cache=True,
        )

        decision = self.router.route_request(request)

        assert decision.cache_key is not None
        assert isinstance(decision.cache_key, str)
        assert len(decision.cache_key) > 0

    def test_cost_estimation(self):
        """Test that costs are estimated correctly."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Test"}],
            agent_role="test",
            task_description="Test",
        )

        decision = self.router.route_request(request)

        assert decision.estimated_cost >= 0
        assert isinstance(decision.estimated_cost, float)

    def test_routing_reason_generated(self):
        """Test that routing reasons are provided."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Test"}],
            agent_role="test",
            task_description="Test",
        )

        decision = self.router.route_request(request)

        assert decision.routing_reason
        assert isinstance(decision.routing_reason, str)
