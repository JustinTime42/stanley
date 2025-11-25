"""Tests for task complexity analyzer."""

import pytest
from src.llm.analyzer import TaskComplexityAnalyzer
from src.models.llm_models import TaskComplexity, ModelCapability


class TestTaskComplexityAnalyzer:
    """Test suite for TaskComplexityAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TaskComplexityAnalyzer()

    def test_simple_task_classification(self):
        """Test that simple tasks are classified correctly."""
        result = self.analyzer.analyze_task(
            task_description="Fix a typo in the README file",
            agent_role="implementer",
        )

        # Should be simple or medium (heuristics may vary)
        assert result.complexity in [TaskComplexity.SIMPLE, TaskComplexity.MEDIUM]
        assert result.confidence > 0.5
        assert len(result.required_capabilities) > 0

    def test_complex_task_classification(self):
        """Test that complex tasks are classified correctly."""
        result = self.analyzer.analyze_task(
            task_description=(
                "Design a comprehensive microservices architecture "
                "with advanced security features and scalable infrastructure"
            ),
            agent_role="architect",
        )

        assert result.complexity == TaskComplexity.COMPLEX
        assert result.confidence > 0.5

    def test_token_estimation(self):
        """Test token estimation."""
        short_task = "Simple task"
        long_task = "Very detailed task " * 100

        short_result = self.analyzer.analyze_task(
            task_description=short_task,
            agent_role="general",
        )

        long_result = self.analyzer.analyze_task(
            task_description=long_task,
            agent_role="general",
        )

        assert long_result.estimated_tokens > short_result.estimated_tokens

    def test_capability_identification(self):
        """Test capability identification from task description."""
        code_task = self.analyzer.analyze_task(
            task_description="Implement a new authentication function",
            agent_role="implementer",
        )

        assert ModelCapability.CODE_GENERATION in code_task.required_capabilities

        debug_task = self.analyzer.analyze_task(
            task_description="Debug the login error in the authentication module",
            agent_role="debugger",
        )

        assert ModelCapability.DEBUGGING in debug_task.required_capabilities

    def test_confidence_scoring(self):
        """Test that confidence scores are reasonable."""
        result = self.analyzer.analyze_task(
            task_description="Implement user authentication with OAuth2",
            agent_role="implementer",
        )

        assert 0.5 <= result.confidence <= 1.0

    def test_reasoning_generation(self):
        """Test that reasoning is generated."""
        result = self.analyzer.analyze_task(
            task_description="Optimize database query performance",
            agent_role="implementer",
        )

        assert result.reasoning
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0
