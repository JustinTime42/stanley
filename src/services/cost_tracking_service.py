"""Cost tracking service for monitoring LLM API usage and costs."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from ..models.llm_models import LLMResponse, PerformanceMetrics, ModelProvider

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Tracks LLM usage costs and performance metrics.

    PATTERN: Real-time tracking with aggregation
    CRITICAL: Track per agent, per workflow, and globally
    GOTCHA: Reset daily stats at midnight
    """

    def __init__(self):
        """Initialize cost tracker."""
        self.logger = logging.getLogger(__name__)

        # Global metrics
        self.total_cost = 0.0
        self.total_tokens = 0
        self.total_requests = 0

        # Per-model metrics
        self.model_metrics: Dict[str, PerformanceMetrics] = {}

        # Per-agent tracking
        self.agent_costs: Dict[str, float] = defaultdict(float)
        self.agent_tokens: Dict[str, int] = defaultdict(int)
        self.agent_requests: Dict[str, int] = defaultdict(int)

        # Per-workflow tracking
        self.workflow_costs: Dict[str, float] = defaultdict(float)

        # Daily tracking
        self.daily_cost = 0.0
        self.daily_reset_time = datetime.now().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ) + timedelta(days=1)

        # Request history (last 1000 for analysis)
        self.request_history: List[Dict[str, Any]] = []
        self.max_history = 1000

    def track_usage(
        self,
        response: LLMResponse,
        agent_role: str,
        workflow_id: Optional[str] = None,
    ) -> None:
        """
        Track LLM usage from a response.

        PATTERN: Update all relevant metrics atomically
        CRITICAL: Called after every LLM request

        Args:
            response: LLM response with cost/token info
            agent_role: Agent that made the request
            workflow_id: Optional workflow identifier
        """
        # Check if we need daily reset
        self._check_daily_reset()

        # Update global metrics
        self.total_cost += response.total_cost
        self.total_tokens += response.input_tokens + response.output_tokens
        self.total_requests += 1

        # Update daily metrics
        self.daily_cost += response.total_cost

        # Update agent metrics
        self.agent_costs[agent_role] += response.total_cost
        self.agent_tokens[agent_role] += (
            response.input_tokens + response.output_tokens
        )
        self.agent_requests[agent_role] += 1

        # Update workflow metrics
        if workflow_id:
            self.workflow_costs[workflow_id] += response.total_cost

        # Update model metrics
        self._update_model_metrics(response)

        # Add to history
        self._add_to_history(response, agent_role, workflow_id)

        self.logger.debug(
            f"Tracked: ${response.total_cost:.4f} for {agent_role} "
            f"using {response.model_used}"
        )

    def _update_model_metrics(self, response: LLMResponse) -> None:
        """
        Update performance metrics for a model.

        Args:
            response: LLM response
        """
        model_name = response.model_used

        # Handle provider enum or string
        provider = (
            response.provider
            if isinstance(response.provider, ModelProvider)
            else ModelProvider(response.provider)
        )

        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = PerformanceMetrics(
                model_name=model_name,
                provider=provider,
            )

        metrics = self.model_metrics[model_name]

        # Update counts
        metrics.total_requests += 1
        if not response.fallback_used:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1

        # Update tokens and cost
        metrics.total_tokens += response.input_tokens + response.output_tokens
        metrics.total_cost += response.total_cost

        # Update latency (running average)
        total_latency = metrics.average_latency_ms * (metrics.total_requests - 1)
        total_latency += response.latency_ms
        metrics.average_latency_ms = total_latency / metrics.total_requests

        # Update success rate
        metrics.success_rate = (
            metrics.successful_requests / metrics.total_requests
        )

        # Update timestamp
        metrics.last_updated = datetime.now()

    def _add_to_history(
        self,
        response: LLMResponse,
        agent_role: str,
        workflow_id: Optional[str],
    ) -> None:
        """
        Add request to history.

        Args:
            response: LLM response
            agent_role: Agent role
            workflow_id: Workflow ID
        """
        # Handle provider enum or string
        provider_value = (
            response.provider.value
            if hasattr(response.provider, 'value')
            else response.provider
        )

        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": response.model_used,
            "provider": provider_value,
            "agent": agent_role,
            "workflow_id": workflow_id,
            "cost": response.total_cost,
            "tokens": response.input_tokens + response.output_tokens,
            "latency_ms": response.latency_ms,
            "cache_hit": response.cache_hit,
            "fallback_used": response.fallback_used,
        }

        self.request_history.append(entry)

        # Trim history if too large
        if len(self.request_history) > self.max_history:
            self.request_history = self.request_history[-self.max_history:]

    def _check_daily_reset(self) -> None:
        """Reset daily metrics if needed."""
        if datetime.now() >= self.daily_reset_time:
            self.logger.info(
                f"Daily reset - Yesterday's cost: ${self.daily_cost:.2f}"
            )
            self.daily_cost = 0.0
            self.daily_reset_time += timedelta(days=1)

    def get_cost_report(
        self,
        scope: str = "global",
        identifier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get cost report for a scope.

        Args:
            scope: Report scope (global, agent, workflow, model)
            identifier: Identifier for agent/workflow/model

        Returns:
            Cost report dictionary
        """
        if scope == "global":
            return {
                "total_cost": self.total_cost,
                "daily_cost": self.daily_cost,
                "total_tokens": self.total_tokens,
                "total_requests": self.total_requests,
                "average_cost_per_request": (
                    self.total_cost / self.total_requests
                    if self.total_requests > 0
                    else 0
                ),
                "models_used": len(self.model_metrics),
            }

        elif scope == "agent":
            if not identifier:
                # Return all agents
                return {
                    agent: {
                        "cost": cost,
                        "tokens": self.agent_tokens[agent],
                        "requests": self.agent_requests[agent],
                    }
                    for agent, cost in self.agent_costs.items()
                }
            else:
                # Return specific agent
                return {
                    "agent": identifier,
                    "cost": self.agent_costs.get(identifier, 0.0),
                    "tokens": self.agent_tokens.get(identifier, 0),
                    "requests": self.agent_requests.get(identifier, 0),
                }

        elif scope == "workflow":
            if not identifier:
                return dict(self.workflow_costs)
            else:
                return {
                    "workflow_id": identifier,
                    "cost": self.workflow_costs.get(identifier, 0.0),
                }

        elif scope == "model":
            if not identifier:
                # Return all models
                return {
                    name: metrics.model_dump()
                    for name, metrics in self.model_metrics.items()
                }
            else:
                # Return specific model
                metrics = self.model_metrics.get(identifier)
                if metrics:
                    return metrics.model_dump()
                return {}

        else:
            raise ValueError(f"Unknown scope: {scope}")

    def get_top_costs(
        self,
        scope: str = "agent",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get top cost items.

        Args:
            scope: Scope (agent, workflow, model)
            limit: Number of results

        Returns:
            List of top cost items
        """
        if scope == "agent":
            items = sorted(
                self.agent_costs.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:limit]
            return [
                {
                    "agent": agent,
                    "cost": cost,
                    "requests": self.agent_requests[agent],
                }
                for agent, cost in items
            ]

        elif scope == "workflow":
            items = sorted(
                self.workflow_costs.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:limit]
            return [
                {"workflow_id": wf_id, "cost": cost}
                for wf_id, cost in items
            ]

        elif scope == "model":
            items = sorted(
                self.model_metrics.items(),
                key=lambda x: x[1].total_cost,
                reverse=True,
            )[:limit]
            return [
                {
                    "model": name,
                    "cost": metrics.total_cost,
                    "requests": metrics.total_requests,
                    "success_rate": metrics.success_rate,
                }
                for name, metrics in items
            ]

        else:
            raise ValueError(f"Unknown scope: {scope}")

    def estimate_cost_savings(self) -> Dict[str, Any]:
        """
        Estimate cost savings from routing vs all-premium baseline.

        PATTERN: Compare actual costs to hypothetical premium-only costs
        CRITICAL: Shows ROI of routing system

        Returns:
            Cost savings analysis
        """
        # Find most expensive model (premium baseline)
        if not self.model_metrics:
            return {
                "actual_cost": 0.0,
                "premium_baseline": 0.0,
                "savings": 0.0,
                "savings_percent": 0.0,
            }

        # Handle provider enum or string
        openai_provider = ModelProvider.OPENAI if hasattr(ModelProvider, 'OPENAI') else "openai"
        anthropic_provider = ModelProvider.ANTHROPIC if hasattr(ModelProvider, 'ANTHROPIC') else "anthropic"

        # Use average cost per token of premium cloud models (OpenAI, Anthropic)
        premium_models = [
            m for m in self.model_metrics.values()
            if str(m.provider).lower() in ["openai", "anthropic"]
        ]

        if not premium_models:
            # Use most expensive model as baseline
            max_cost_per_token = max(
                (m.total_cost / m.total_tokens if m.total_tokens > 0 else 0)
                for m in self.model_metrics.values()
            )
        else:
            # Average cost per token of premium models
            total_cost = sum(m.total_cost for m in premium_models)
            total_tokens = sum(m.total_tokens for m in premium_models)
            max_cost_per_token = (
                total_cost / total_tokens if total_tokens > 0 else 0
            )

        # Calculate hypothetical premium cost
        premium_baseline = self.total_tokens * max_cost_per_token

        # Calculate savings
        savings = premium_baseline - self.total_cost
        savings_percent = (
            (savings / premium_baseline * 100)
            if premium_baseline > 0
            else 0
        )

        return {
            "actual_cost": self.total_cost,
            "premium_baseline": premium_baseline,
            "savings": savings,
            "savings_percent": savings_percent,
            "total_tokens": self.total_tokens,
        }

    def check_cost_limits(
        self,
        request_cost: float,
        workflow_id: Optional[str] = None,
        max_request_cost: float = 0.10,
        max_workflow_cost: float = 1.00,
        max_daily_cost: float = 50.00,
    ) -> Dict[str, Any]:
        """
        Check if request would exceed cost limits.

        PATTERN: Pre-flight cost validation
        CRITICAL: Prevent runaway costs

        Args:
            request_cost: Estimated request cost
            workflow_id: Optional workflow ID
            max_request_cost: Max cost per request
            max_workflow_cost: Max cost per workflow
            max_daily_cost: Max daily cost

        Returns:
            Dictionary with limit check results
        """
        violations = []

        # Check request limit
        if request_cost > max_request_cost:
            violations.append(
                f"Request cost ${request_cost:.4f} exceeds "
                f"limit ${max_request_cost:.4f}"
            )

        # Check workflow limit
        if workflow_id:
            workflow_cost = self.workflow_costs.get(workflow_id, 0.0)
            if workflow_cost + request_cost > max_workflow_cost:
                violations.append(
                    f"Workflow cost ${workflow_cost + request_cost:.4f} "
                    f"would exceed limit ${max_workflow_cost:.4f}"
                )

        # Check daily limit
        if self.daily_cost + request_cost > max_daily_cost:
            violations.append(
                f"Daily cost ${self.daily_cost + request_cost:.2f} "
                f"would exceed limit ${max_daily_cost:.2f}"
            )

        return {
            "allowed": len(violations) == 0,
            "violations": violations,
            "request_cost": request_cost,
            "daily_cost": self.daily_cost,
            "workflow_cost": (
                self.workflow_costs.get(workflow_id, 0.0)
                if workflow_id
                else None
            ),
        }

    def get_recent_requests(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent request history.

        Args:
            limit: Number of recent requests

        Returns:
            List of recent requests
        """
        return self.request_history[-limit:]
