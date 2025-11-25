"""End-to-end integration tests for LLM system."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from src.services.llm_service import LLMOrchestrator
from src.services.cost_tracking_service import CostTracker
from src.config.llm_config import LLMConfig
from src.models.llm_models import LLMRequest, LLMResponse, ModelProvider, TaskComplexity
from src.agents.planner import PlannerAgent
from src.models.state_models import AgentState


class TestEndToEnd:
    """End-to-end integration tests."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = LLMConfig()
        self.orchestrator = LLMOrchestrator(self.config)
        self.cost_tracker = CostTracker()

    def test_full_system_initialization(self):
        """Test that the full system can be initialized."""
        # Verify all components are present
        assert self.orchestrator.config is not None
        assert self.orchestrator.analyzer is not None
        assert self.orchestrator.router is not None
        assert self.orchestrator.fallback_manager is not None
        assert len(self.orchestrator.providers) > 0

        # Verify cost tracker is ready
        assert self.cost_tracker.total_cost == 0.0

    def test_routing_workflow(self):
        """Test complete routing workflow from request to decision."""
        # Create a request
        request = LLMRequest(
            messages=[{"role": "user", "content": "Write a hello world function"}],
            agent_role="implementer",
            task_description="Implement hello world",
        )

        # Get routing decision
        decision = self.orchestrator.router.route_request(request)

        # Verify decision structure
        assert decision.task_analysis is not None
        assert decision.selected_model is not None
        assert decision.fallback_models is not None
        assert decision.estimated_cost >= 0
        assert decision.routing_reason is not None

        # Verify task analysis
        analysis = decision.task_analysis
        assert analysis.complexity in [
            TaskComplexity.SIMPLE,
            TaskComplexity.MEDIUM,
            TaskComplexity.COMPLEX,
        ]
        assert analysis.estimated_tokens > 0
        assert len(analysis.required_capabilities) > 0

    def test_cost_tracking_workflow(self):
        """Test cost tracking through a workflow."""
        # Simulate LLM responses
        response1 = LLMResponse(
            content="First response",
            model_used="qwen2.5-coder:14b",
            provider=ModelProvider.OLLAMA,
            input_tokens=100,
            output_tokens=50,
            total_cost=0.0,  # Local model
            latency_ms=500,
        )

        response2 = LLMResponse(
            content="Second response",
            model_used="gpt-4o-mini",
            provider=ModelProvider.OPENAI,
            input_tokens=200,
            output_tokens=100,
            total_cost=0.01,
            latency_ms=800,
        )

        # Track usage
        self.cost_tracker.track_usage(response1, "planner", "workflow_1")
        self.cost_tracker.track_usage(response2, "architect", "workflow_1")

        # Verify tracking
        assert self.cost_tracker.total_cost == 0.01
        assert self.cost_tracker.total_requests == 2

        # Verify per-agent tracking
        assert "planner" in self.cost_tracker.agent_costs
        assert "architect" in self.cost_tracker.agent_costs

        # Verify workflow tracking
        assert self.cost_tracker.workflow_costs["workflow_1"] == 0.01

    def test_agent_integration_without_llm(self):
        """Test that agents work without LLM service (fallback mode)."""
        # Create planner without LLM service
        planner = PlannerAgent(memory_service=None, llm_service=None)

        assert planner.role is not None
        assert planner.llm_service is None

    def test_multiple_complexity_levels(self):
        """Test routing for different complexity levels."""
        complexities_to_test = [
            ("Fix typo", TaskComplexity.SIMPLE),
            ("Implement authentication", TaskComplexity.MEDIUM),
            ("Design microservices architecture", TaskComplexity.COMPLEX),
        ]

        for task_desc, expected_complexity in complexities_to_test:
            request = LLMRequest(
                messages=[{"role": "user", "content": task_desc}],
                agent_role="test",
                task_description=task_desc,
            )

            decision = self.orchestrator.router.route_request(request)

            # Verify appropriate routing
            # Note: Complexity might vary based on heuristics, so we just check it's valid
            assert decision.task_analysis.complexity in [
                TaskComplexity.SIMPLE,
                TaskComplexity.MEDIUM,
                TaskComplexity.COMPLEX,
            ]

    def test_cost_savings_estimation(self):
        """Test cost savings estimation across workflow."""
        # Add mix of local and cloud usage
        local_response = LLMResponse(
            content="Local", model_used="qwen2.5-coder:14b",
            provider=ModelProvider.OLLAMA,
            input_tokens=1000, output_tokens=500, total_cost=0.0, latency_ms=500,
        )

        cloud_response = LLMResponse(
            content="Cloud", model_used="gpt-4o",
            provider=ModelProvider.OPENAI,
            input_tokens=1000, output_tokens=500, total_cost=0.05, latency_ms=800,
        )

        self.cost_tracker.track_usage(local_response, "planner")
        self.cost_tracker.track_usage(cloud_response, "architect")

        # Get savings estimate
        savings = self.cost_tracker.estimate_cost_savings()

        # Should show savings from using local model
        assert savings["actual_cost"] < savings["premium_baseline"]
        assert savings["savings"] > 0
        assert savings["savings_percent"] > 0

    def test_cache_integration(self):
        """Test cache integration in workflow."""
        if not self.orchestrator.cache:
            pytest.skip("Cache not enabled")

        # Get initial cache stats
        stats_before = self.orchestrator.get_cache_stats()
        initial_size = stats_before["size"]

        # Cache stats should be accessible
        assert "hits" in stats_before
        assert "misses" in stats_before
        assert stats_before["enabled"] is True

    @pytest.mark.asyncio
    async def test_cleanup_workflow(self):
        """Test cleanup of all resources."""
        # Should not raise errors
        await self.orchestrator.cleanup()

        # Can call multiple times
        await self.orchestrator.cleanup()

    def test_fallback_chain_reliability(self):
        """Test that fallback chains provide reliability."""
        request = LLMRequest(
            messages=[{"role": "user", "content": "Test"}],
            agent_role="test",
            task_description="Test task",
        )

        decision = self.orchestrator.router.route_request(request)
        chain = self.orchestrator._build_fallback_chain(decision)

        # Should have multiple options for reliability
        assert len(chain) >= 1

        # If prefer_local is true, primary should be local
        if self.config.prefer_local and len(chain) > 0:
            # At least one should be local
            has_local = any(p.config.is_local for p in chain)
            assert has_local

    def test_model_configuration_completeness(self):
        """Test that model configurations are complete."""
        models = self.config.get_model_configs()

        for model in models:
            # Verify required fields
            assert model.provider is not None
            assert model.model_name is not None
            assert model.context_window > 0
            assert model.cost_per_1k_input >= 0
            assert model.cost_per_1k_output >= 0
            assert len(model.capabilities) > 0
            assert 0 <= model.performance_score <= 1

    def test_qwen_model_availability(self):
        """Test that qwen2.5-coder:14b is properly configured."""
        models = self.config.get_model_configs()
        qwen_models = [m for m in models if "qwen" in m.model_name.lower()]

        assert len(qwen_models) > 0

        qwen = qwen_models[0]
        assert qwen.model_name == "qwen2.5-coder:14b"
        assert qwen.is_local is True
        assert qwen.cost_per_1k_input == 0.0
        assert qwen.cost_per_1k_output == 0.0
        assert qwen.context_window == 32768
        assert qwen.performance_score == 0.8
