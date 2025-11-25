"""Tests for cost tracking service."""

import pytest
from datetime import datetime, timedelta
from src.services.cost_tracking_service import CostTracker
from src.models.llm_models import LLMResponse, ModelProvider


class TestCostTracker:
    """Test suite for CostTracker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = CostTracker()

    def test_initial_state(self):
        """Test initial state of tracker."""
        assert self.tracker.total_cost == 0.0
        assert self.tracker.total_tokens == 0
        assert self.tracker.total_requests == 0
        assert len(self.tracker.model_metrics) == 0

    def test_track_usage(self):
        """Test tracking LLM usage."""
        response = LLMResponse(
            content="Test response",
            model_used="gpt-4o-mini",
            provider=ModelProvider.OPENAI,
            input_tokens=100,
            output_tokens=50,
            total_cost=0.01,
            latency_ms=500,
        )

        self.tracker.track_usage(
            response=response,
            agent_role="planner",
            workflow_id="workflow_123",
        )

        # Check global metrics
        assert self.tracker.total_cost == 0.01
        assert self.tracker.total_tokens == 150
        assert self.tracker.total_requests == 1

        # Check agent metrics
        assert self.tracker.agent_costs["planner"] == 0.01
        assert self.tracker.agent_tokens["planner"] == 150
        assert self.tracker.agent_requests["planner"] == 1

        # Check workflow metrics
        assert self.tracker.workflow_costs["workflow_123"] == 0.01

    def test_multiple_tracking(self):
        """Test tracking multiple requests."""
        for i in range(5):
            response = LLMResponse(
                content=f"Response {i}",
                model_used="gpt-4o-mini",
                provider=ModelProvider.OPENAI,
                input_tokens=100,
                output_tokens=50,
                total_cost=0.01,
                latency_ms=500,
            )
            self.tracker.track_usage(response, "planner")

        assert self.tracker.total_cost == 0.05
        assert self.tracker.total_tokens == 750
        assert self.tracker.total_requests == 5

    def test_model_metrics_tracking(self):
        """Test that model-specific metrics are tracked."""
        response = LLMResponse(
            content="Test",
            model_used="gpt-4o-mini",
            provider=ModelProvider.OPENAI,
            input_tokens=100,
            output_tokens=50,
            total_cost=0.01,
            latency_ms=500,
        )

        self.tracker.track_usage(response, "planner")

        # Check model metrics
        assert "gpt-4o-mini" in self.tracker.model_metrics
        metrics = self.tracker.model_metrics["gpt-4o-mini"]

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.total_tokens == 150
        assert metrics.total_cost == 0.01
        assert metrics.average_latency_ms == 500
        assert metrics.success_rate == 1.0

    def test_fallback_tracking(self):
        """Test tracking fallback usage."""
        response = LLMResponse(
            content="Test",
            model_used="gpt-4o",
            provider=ModelProvider.OPENAI,
            input_tokens=100,
            output_tokens=50,
            total_cost=0.05,
            latency_ms=500,
            fallback_used=True,
        )

        self.tracker.track_usage(response, "planner")

        metrics = self.tracker.model_metrics["gpt-4o"]
        assert metrics.failed_requests == 1  # Counted as failure since fallback was used

    def test_get_cost_report_global(self):
        """Test getting global cost report."""
        # Add some usage
        for _ in range(3):
            response = LLMResponse(
                content="Test",
                model_used="test",
                provider=ModelProvider.OPENAI,
                input_tokens=100,
                output_tokens=50,
                total_cost=0.01,
                latency_ms=500,
            )
            self.tracker.track_usage(response, "planner")

        report = self.tracker.get_cost_report(scope="global")

        assert report["total_cost"] == 0.03
        assert report["total_tokens"] == 450
        assert report["total_requests"] == 3
        assert report["average_cost_per_request"] == 0.01

    def test_get_cost_report_agent(self):
        """Test getting agent-specific cost report."""
        # Add usage for different agents
        response1 = LLMResponse(
            content="Test", model_used="test", provider=ModelProvider.OPENAI,
            input_tokens=100, output_tokens=50, total_cost=0.01, latency_ms=500,
        )
        response2 = LLMResponse(
            content="Test", model_used="test", provider=ModelProvider.OPENAI,
            input_tokens=200, output_tokens=100, total_cost=0.02, latency_ms=500,
        )

        self.tracker.track_usage(response1, "planner")
        self.tracker.track_usage(response2, "architect")

        report = self.tracker.get_cost_report(scope="agent", identifier="planner")

        assert report["agent"] == "planner"
        assert report["cost"] == 0.01
        assert report["tokens"] == 150
        assert report["requests"] == 1

    def test_get_top_costs(self):
        """Test getting top cost items."""
        # Add usage for multiple agents
        agents = ["planner", "architect", "implementer"]
        costs = [0.05, 0.03, 0.01]

        for agent, cost in zip(agents, costs):
            response = LLMResponse(
                content="Test", model_used="test", provider=ModelProvider.OPENAI,
                input_tokens=100, output_tokens=50, total_cost=cost, latency_ms=500,
            )
            self.tracker.track_usage(response, agent)

        top_costs = self.tracker.get_top_costs(scope="agent", limit=2)

        assert len(top_costs) == 2
        assert top_costs[0]["agent"] == "planner"
        assert top_costs[0]["cost"] == 0.05
        assert top_costs[1]["agent"] == "architect"

    def test_estimate_cost_savings(self):
        """Test cost savings estimation."""
        # Add some local model usage (free)
        local_response = LLMResponse(
            content="Test", model_used="qwen2.5-coder:14b",
            provider=ModelProvider.OLLAMA,
            input_tokens=1000, output_tokens=500, total_cost=0.0, latency_ms=500,
        )
        self.tracker.track_usage(local_response, "planner")

        # Add some cloud model usage
        cloud_response = LLMResponse(
            content="Test", model_used="gpt-4o",
            provider=ModelProvider.OPENAI,
            input_tokens=1000, output_tokens=500, total_cost=0.05, latency_ms=500,
        )
        self.tracker.track_usage(cloud_response, "architect")

        savings = self.tracker.estimate_cost_savings()

        # Should show savings from using local model
        assert savings["actual_cost"] == 0.05
        assert savings["premium_baseline"] > 0.05
        assert savings["savings"] > 0
        assert savings["savings_percent"] > 0

    def test_check_cost_limits(self):
        """Test cost limit checking."""
        # Test request limit
        result = self.tracker.check_cost_limits(
            request_cost=0.15,
            max_request_cost=0.10,
        )
        assert result["allowed"] is False
        assert len(result["violations"]) > 0

        # Test within limits
        result = self.tracker.check_cost_limits(
            request_cost=0.05,
            max_request_cost=0.10,
        )
        assert result["allowed"] is True
        assert len(result["violations"]) == 0

    def test_check_workflow_cost_limit(self):
        """Test workflow cost limit checking."""
        # Add some workflow cost
        response = LLMResponse(
            content="Test", model_used="test", provider=ModelProvider.OPENAI,
            input_tokens=100, output_tokens=50, total_cost=0.50, latency_ms=500,
        )
        self.tracker.track_usage(response, "planner", workflow_id="wf_123")

        # Check if adding more would exceed limit
        result = self.tracker.check_cost_limits(
            request_cost=0.60,
            workflow_id="wf_123",
            max_workflow_cost=1.00,
        )
        assert result["allowed"] is False

    def test_check_daily_cost_limit(self):
        """Test daily cost limit checking."""
        # Add some daily cost
        for _ in range(10):
            response = LLMResponse(
                content="Test", model_used="test", provider=ModelProvider.OPENAI,
                input_tokens=100, output_tokens=50, total_cost=1.0, latency_ms=500,
            )
            self.tracker.track_usage(response, "planner")

        # Should have 10.0 daily cost
        result = self.tracker.check_cost_limits(
            request_cost=45.0,
            max_daily_cost=50.0,
        )
        assert result["allowed"] is False

    def test_request_history(self):
        """Test request history tracking."""
        response = LLMResponse(
            content="Test", model_used="test", provider=ModelProvider.OPENAI,
            input_tokens=100, output_tokens=50, total_cost=0.01, latency_ms=500,
        )
        self.tracker.track_usage(response, "planner", workflow_id="wf_123")

        history = self.tracker.get_recent_requests(limit=10)

        assert len(history) == 1
        assert history[0]["model"] == "test"
        assert history[0]["agent"] == "planner"
        assert history[0]["workflow_id"] == "wf_123"
        assert history[0]["cost"] == 0.01

    def test_history_limit(self):
        """Test that history is limited."""
        # Add more than max_history requests
        for i in range(1500):
            response = LLMResponse(
                content="Test", model_used="test", provider=ModelProvider.OPENAI,
                input_tokens=10, output_tokens=10, total_cost=0.001, latency_ms=100,
            )
            self.tracker.track_usage(response, "test")

        # Should only keep last 1000
        assert len(self.tracker.request_history) == 1000
