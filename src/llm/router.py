"""Intelligent model routing engine for cost-optimized LLM selection."""

import logging
import hashlib
import json
from typing import List, Optional
from ..models.llm_models import (
    TaskComplexity,
    ModelCapability,
    ModelConfig,
    LLMRequest,
)
from ..models.routing_models import TaskAnalysis, RoutingDecision
from .analyzer import TaskComplexityAnalyzer

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Routes LLM requests to the most appropriate model based on complexity and cost.

    PATTERN: Cost-optimized selection with capability matching
    CRITICAL: Decision must complete in <100ms
    GOTCHA: Balance cost vs performance based on task complexity
    """

    def __init__(
        self,
        available_models: List[ModelConfig],
        analyzer: Optional[TaskComplexityAnalyzer] = None,
        prefer_local: bool = True,
        routing_strategy: str = "cost_optimized",
    ):
        """
        Initialize model router.

        Args:
            available_models: List of available model configurations
            analyzer: Task complexity analyzer (creates default if None)
            prefer_local: Prefer local models when possible
            routing_strategy: Routing strategy (cost_optimized, performance, balanced)
        """
        self.available_models = available_models
        self.analyzer = analyzer or TaskComplexityAnalyzer()
        self.prefer_local = prefer_local
        self.routing_strategy = routing_strategy
        self.logger = logging.getLogger(__name__)

    def route_request(
        self,
        request: LLMRequest,
        task_analysis: Optional[TaskAnalysis] = None,
    ) -> RoutingDecision:
        """
        Route request to the most appropriate model.

        PATTERN: Multi-stage filtering and ranking
        CRITICAL: Must complete in <100ms

        Args:
            request: LLM request
            task_analysis: Pre-computed task analysis (analyzes if None)

        Returns:
            Routing decision with selected model and fallbacks
        """
        # Analyze task if not provided
        if not task_analysis:
            task_analysis = self.analyzer.analyze_task(
                task_description=request.task_description,
                agent_role=request.agent_role,
                message_history=request.messages,
            )

        # Override complexity if specified
        if request.complexity_override:
            task_analysis.complexity = request.complexity_override

        # Stage 1: Filter by capability
        capable_models = self._filter_by_capability(
            request.required_capability or (
                task_analysis.required_capabilities[0]
                if task_analysis.required_capabilities
                else ModelCapability.GENERAL
            )
        )

        if not capable_models:
            self.logger.warning("No capable models found, using all models")
            capable_models = self.available_models

        # Stage 2: Filter by complexity tier
        appropriate_models = self._filter_by_complexity(
            capable_models,
            task_analysis.complexity,
        )

        if not appropriate_models:
            self.logger.warning("No appropriate models, using capable models")
            appropriate_models = capable_models

        # Stage 3: Filter by requirements
        if task_analysis.requires_functions:
            func_models = [m for m in appropriate_models if m.supports_functions]
            if func_models:
                appropriate_models = func_models

        # Stage 4: Sort by routing strategy
        sorted_models = self._sort_by_strategy(
            appropriate_models,
            task_analysis,
        )

        # Select primary and fallbacks
        selected_model = sorted_models[0]
        fallback_models = sorted_models[1:4]  # Top 3 alternatives

        # Calculate estimated cost
        estimated_cost = self._estimate_cost(selected_model, task_analysis)

        # Calculate estimated latency
        estimated_latency = self._estimate_latency(selected_model)

        # Generate routing reason
        routing_reason = self._generate_routing_reason(
            selected_model,
            task_analysis,
        )

        # Generate cache key
        cache_key = self._generate_cache_key(request) if request.use_cache else None

        return RoutingDecision(
            task_analysis=task_analysis,
            selected_model=selected_model,
            fallback_models=fallback_models,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            routing_reason=routing_reason,
            cache_key=cache_key,
        )

    def _filter_by_capability(
        self,
        required_capability: ModelCapability,
    ) -> List[ModelConfig]:
        """
        Filter models by required capability.

        Args:
            required_capability: Required capability

        Returns:
            Models supporting the capability
        """
        return [
            model for model in self.available_models
            if required_capability in model.capabilities
        ]

    def _filter_by_complexity(
        self,
        models: List[ModelConfig],
        complexity: TaskComplexity,
    ) -> List[ModelConfig]:
        """
        Filter models appropriate for task complexity.

        PATTERN: Simple tasks can use any model, complex tasks need high-performance
        GOTCHA: Don't route complex tasks to weak models

        Args:
            models: Models to filter
            complexity: Task complexity

        Returns:
            Appropriate models
        """
        if complexity == TaskComplexity.SIMPLE:
            # Simple tasks can use any model, prefer local/cheap
            return models

        elif complexity == TaskComplexity.MEDIUM:
            # Medium tasks need decent models (performance > 0.6)
            return [
                m for m in models
                if m.performance_score >= 0.6
            ]

        else:  # COMPLEX
            # Complex tasks need premium models (performance > 0.8)
            return [
                m for m in models
                if m.performance_score >= 0.8
            ]

    def _sort_by_strategy(
        self,
        models: List[ModelConfig],
        task_analysis: TaskAnalysis,
    ) -> List[ModelConfig]:
        """
        Sort models by routing strategy.

        PATTERN: Different strategies optimize for different goals
        CRITICAL: cost_optimized is default for cost savings

        Args:
            models: Models to sort
            task_analysis: Task analysis

        Returns:
            Sorted models
        """
        if self.routing_strategy == "cost_optimized":
            # Sort by cost, prefer local
            return sorted(
                models,
                key=lambda m: (
                    not m.is_local,  # Local first
                    m.cost_per_1k_output,  # Then by cost
                )
            )

        elif self.routing_strategy == "performance":
            # Sort by performance score
            return sorted(
                models,
                key=lambda m: -m.performance_score,
            )

        else:  # balanced
            # Balance cost and performance
            return sorted(
                models,
                key=lambda m: (
                    not m.is_local,  # Local first
                    -m.performance_score / max(m.cost_per_1k_output, 0.001),
                )
            )

    def _estimate_cost(
        self,
        model: ModelConfig,
        task_analysis: TaskAnalysis,
    ) -> float:
        """
        Estimate cost for the request.

        PATTERN: Token-based cost estimation
        CRITICAL: Account for both input and output tokens

        Args:
            model: Selected model
            task_analysis: Task analysis with token estimate

        Returns:
            Estimated cost in USD
        """
        # Estimate input tokens (from task analysis)
        input_tokens = task_analysis.estimated_tokens

        # Estimate output tokens (heuristic: 50% of input for most tasks)
        output_tokens = int(input_tokens * 0.5)

        # Calculate cost
        input_cost = (input_tokens / 1000) * model.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model.cost_per_1k_output

        return input_cost + output_cost

    def _estimate_latency(self, model: ModelConfig) -> int:
        """
        Estimate response latency.

        PATTERN: Local models faster, cloud models vary
        GOTCHA: Actual latency depends on network, load, etc.

        Args:
            model: Model configuration

        Returns:
            Estimated latency in milliseconds
        """
        if model.is_local:
            # Local models: 500-2000ms depending on model size
            base_latency = 1000
        else:
            # Cloud models: 1000-3000ms depending on provider
            base_latency = 2000

        # Adjust by performance score (better models often faster)
        latency = int(base_latency / model.performance_score)

        return latency

    def _generate_routing_reason(
        self,
        model: ModelConfig,
        task_analysis: TaskAnalysis,
    ) -> str:
        """
        Generate human-readable routing reason.

        Args:
            model: Selected model
            task_analysis: Task analysis

        Returns:
            Routing reason string
        """
        reasons = []

        # Complexity match
        complexity_str = (
            task_analysis.complexity.value
            if hasattr(task_analysis.complexity, 'value')
            else task_analysis.complexity
        )
        reasons.append(
            f"{complexity_str} task routed to "
            f"{'local' if model.is_local else 'cloud'} model"
        )

        # Cost consideration
        if model.is_local:
            reasons.append("zero cost")
        else:
            reasons.append(f"est. ${self._estimate_cost(model, task_analysis):.4f}")

        # Performance
        reasons.append(f"performance: {model.performance_score:.2f}")

        return "; ".join(reasons)

    def _generate_cache_key(self, request: LLMRequest) -> str:
        """
        Generate cache key for the request.

        PATTERN: Hash of normalized request parameters
        GOTCHA: Include temperature to avoid wrong cached responses

        Args:
            request: LLM request

        Returns:
            Cache key (SHA256 hash)
        """
        # Normalize messages
        normalized_messages = json.dumps(
            request.messages,
            sort_keys=True,
        )

        # Create hash input
        cache_input = (
            f"{normalized_messages}:"
            f"{request.required_capability}:"
            f"{request.temperature}"
        )

        # Generate SHA256 hash
        return hashlib.sha256(cache_input.encode()).hexdigest()
