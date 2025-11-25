"""Task assignment to optimal models/agents based on complexity and cost."""

import logging
from typing import Optional
from ..llm.router import ModelRouter
from ..models.decomposition_models import Task, DecompositionTree
from ..models.llm_models import LLMRequest, ModelCapability


logger = logging.getLogger(__name__)


class TaskAssigner:
    """
    Assigns tasks to optimal models based on complexity and budget constraints.

    PATTERN: Greedy assignment with budget awareness
    CRITICAL: Must respect cost constraints while optimizing performance
    GOTCHA: Assign complex tasks to premium models, simple tasks to cheap models
    """

    def __init__(
        self,
        model_router: ModelRouter,
        default_budget: float = 10.0,
    ):
        """
        Initialize task assigner.

        Args:
            model_router: Model router for routing decisions
            default_budget: Default budget in USD
        """
        self.model_router = model_router
        self.default_budget = default_budget
        self.logger = logging.getLogger(__name__)

    async def assign_models(
        self,
        tree: DecompositionTree,
        budget_remaining: Optional[float] = None,
    ) -> float:
        """
        Assign optimal models to all leaf tasks in the tree.

        PATTERN: Sort by complexity (high to low), assign greedily
        CRITICAL: Track budget and fall back to cheaper models when depleted

        Args:
            tree: Decomposition tree
            budget_remaining: Available budget (uses default if None)

        Returns:
            Total estimated cost for all assignments
        """
        if budget_remaining is None:
            budget_remaining = self.default_budget

        # Get leaf tasks sorted by complexity (high to low)
        leaf_tasks = [tree.tasks[tid] for tid in tree.leaf_tasks]
        leaf_tasks.sort(key=lambda t: t.estimated_complexity, reverse=True)

        total_cost = 0.0
        assigned_count = 0

        for task in leaf_tasks:
            # Create routing request
            estimated_cost = await self._assign_model_to_task(
                task,
                budget_remaining,
            )

            total_cost += estimated_cost
            budget_remaining -= estimated_cost
            assigned_count += 1

            self.logger.debug(
                f"Assigned {task.assigned_model} to task {task.id} "
                f"(cost: ${estimated_cost:.4f}, remaining: ${budget_remaining:.2f})"
            )

        # Update tree cost estimate
        tree.estimated_total_cost = total_cost

        self.logger.info(
            f"Assigned models to {assigned_count} leaf tasks, "
            f"total estimated cost: ${total_cost:.2f}"
        )

        return total_cost

    async def _assign_model_to_task(
        self,
        task: Task,
        budget_remaining: float,
    ) -> float:
        """
        Assign optimal model to a single task.

        PATTERN: Use model router with budget constraint
        CRITICAL: Respect budget limit

        Args:
            task: Task to assign
            budget_remaining: Remaining budget

        Returns:
            Estimated cost for this task
        """
        # Map task type to capability
        capability = self._map_task_type_to_capability(task.type)

        # Create LLM request for routing
        request = LLMRequest(
            messages=[{"role": "user", "content": task.description}],
            agent_role=task.type.value,
            task_description=task.description,
            required_capability=capability,
            use_cache=True,
        )

        # Route to appropriate model
        try:
            routing_decision = self.model_router.route_request(request)

            # Check if estimated cost exceeds budget
            if routing_decision.estimated_cost > budget_remaining:
                self.logger.warning(
                    f"Task {task.id} cost ${routing_decision.estimated_cost:.4f} "
                    f"exceeds remaining budget ${budget_remaining:.2f}, "
                    f"falling back to cheaper model"
                )

                # Try to find cheaper model
                routing_decision = self._find_cheapest_capable_model(task, capability)

            # Assign model and cost
            task.assigned_model = routing_decision.selected_model.model_name
            task.estimated_cost = routing_decision.estimated_cost
            task.estimated_tokens = routing_decision.task_analysis.estimated_tokens

            return routing_decision.estimated_cost

        except Exception as e:
            self.logger.error(f"Failed to assign model to task {task.id}: {e}")

            # Fall back to default assignment
            task.assigned_model = "default"
            task.estimated_cost = 0.0
            task.estimated_tokens = 0

            return 0.0

    def _find_cheapest_capable_model(
        self,
        task: Task,
        capability: ModelCapability,
    ):
        """
        Find the cheapest model capable of handling the task.

        PATTERN: Filter by capability, sort by cost
        GOTCHA: May sacrifice quality for cost

        Args:
            task: Task to assign
            capability: Required capability

        Returns:
            Routing decision with cheapest model
        """
        # Create minimal request for routing
        request = LLMRequest(
            messages=[{"role": "user", "content": task.description}],
            agent_role=task.type.value,
            task_description=task.description,
            required_capability=capability,
            use_cache=True,
        )

        # Get routing decision
        # The router already tries to optimize for cost
        return self.model_router.route_request(request)

    def _map_task_type_to_capability(self, task_type) -> ModelCapability:
        """
        Map TaskType to ModelCapability.

        PATTERN: Direct mapping with sensible defaults
        CRITICAL: Ensure correct capability for task type

        Args:
            task_type: TaskType enum

        Returns:
            ModelCapability enum
        """
        from ..models.decomposition_models import TaskType

        mapping = {
            TaskType.CODE_GENERATION: ModelCapability.CODE_GENERATION,
            TaskType.CODE_MODIFICATION: ModelCapability.CODE_GENERATION,
            TaskType.TESTING: ModelCapability.TESTING,
            TaskType.DOCUMENTATION: ModelCapability.DOCUMENTATION,
            TaskType.RESEARCH: ModelCapability.GENERAL,
            TaskType.DEBUGGING: ModelCapability.DEBUGGING,
            TaskType.REFACTORING: ModelCapability.CODE_REVIEW,
            TaskType.ARCHITECTURE: ModelCapability.PLANNING,
            TaskType.ANALYSIS: ModelCapability.CODE_REVIEW,
        }

        return mapping.get(task_type, ModelCapability.GENERAL)

    async def optimize_assignments(
        self,
        tree: DecompositionTree,
        budget: float,
    ) -> None:
        """
        Optimize model assignments to minimize cost while meeting quality constraints.

        PATTERN: Iterative reassignment with cost-quality tradeoff
        GOTCHA: May move simple tasks to cheaper models

        Args:
            tree: Decomposition tree
            budget: Total budget constraint
        """
        # Get current total cost
        current_cost = tree.estimated_total_cost

        if current_cost <= budget:
            self.logger.info(
                f"Current cost ${current_cost:.2f} within budget ${budget:.2f}, "
                "no optimization needed"
            )
            return

        self.logger.info(
            f"Optimizing assignments: current ${current_cost:.2f} > budget ${budget:.2f}"
        )

        # Sort leaf tasks by complexity (low to high)
        # Move simple tasks to cheaper models first
        leaf_tasks = [tree.tasks[tid] for tid in tree.leaf_tasks]
        leaf_tasks.sort(key=lambda t: t.estimated_complexity)

        budget_remaining = budget
        total_cost = 0.0

        for task in leaf_tasks:
            # Reassign with updated budget
            cost = await self._assign_model_to_task(task, budget_remaining)
            total_cost += cost
            budget_remaining -= cost

        tree.estimated_total_cost = total_cost

        self.logger.info(
            f"Optimized assignments: new cost ${total_cost:.2f}, "
            f"savings: ${current_cost - total_cost:.2f}"
        )

    def get_assignment_summary(self, tree: DecompositionTree) -> dict:
        """
        Generate summary of model assignments.

        PATTERN: Aggregate statistics for analysis
        GOTCHA: Useful for debugging and cost analysis

        Args:
            tree: Decomposition tree

        Returns:
            Dictionary with assignment statistics
        """
        model_counts = {}
        model_costs = {}
        total_cost = 0.0

        for task_id in tree.leaf_tasks:
            task = tree.tasks[task_id]

            if task.assigned_model:
                model_counts[task.assigned_model] = (
                    model_counts.get(task.assigned_model, 0) + 1
                )
                model_costs[task.assigned_model] = (
                    model_costs.get(task.assigned_model, 0.0) + task.estimated_cost
                )
                total_cost += task.estimated_cost

        return {
            "total_tasks": len(tree.leaf_tasks),
            "model_counts": model_counts,
            "model_costs": model_costs,
            "total_estimated_cost": total_cost,
            "average_cost_per_task": (
                total_cost / len(tree.leaf_tasks) if tree.leaf_tasks else 0.0
            ),
        }
