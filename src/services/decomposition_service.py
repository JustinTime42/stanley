"""High-level decomposition orchestration service."""

import logging
from typing import Optional
from ..decomposition.fractal_decomposer import FractalDecomposer
from ..decomposition.task_assigner import TaskAssigner
from ..decomposition.progress_tracker import ProgressTracker
from ..decomposition.dependency_manager import DependencyManager
from ..llm.router import ModelRouter
from ..models.decomposition_models import (
    DecompositionRequest,
    DecompositionResult,
    DecompositionTree,
    TaskStatus,
    ProgressUpdate,
)


logger = logging.getLogger(__name__)


class DecompositionOrchestrator:
    """
    High-level orchestration service for task decomposition.

    PATTERN: Facade pattern coordinating all decomposition components
    CRITICAL: Single entry point for decomposition operations
    GOTCHA: Maintains consistency across components
    """

    def __init__(
        self,
        model_router: Optional[ModelRouter] = None,
        redis_client=None,
        max_depth: int = 10,
        complexity_threshold: float = 0.2,
        max_subtasks_per_level: int = 5,
    ):
        """
        Initialize decomposition orchestrator.

        Args:
            model_router: Model router for task assignment
            redis_client: Optional Redis client for distributed operations
            max_depth: Maximum decomposition depth
            complexity_threshold: Minimum complexity to decompose
            max_subtasks_per_level: Maximum subtasks per level
        """
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.decomposer = FractalDecomposer(
            max_depth=max_depth,
            complexity_threshold=complexity_threshold,
            max_subtasks_per_level=max_subtasks_per_level,
        )

        self.progress_tracker = ProgressTracker(redis_client=redis_client)
        self.dependency_manager = DependencyManager()

        # Initialize task assigner if router provided
        self.task_assigner = None
        if model_router:
            self.task_assigner = TaskAssigner(
                model_router=model_router,
                default_budget=10.0,
            )

        self.logger.info("Decomposition orchestrator initialized")

    async def decompose_task(
        self,
        request: DecompositionRequest,
    ) -> DecompositionResult:
        """
        Decompose a task into a tree of subtasks.

        PATTERN: Decompose -> Assign models -> Return result
        CRITICAL: Main entry point for decomposition

        Args:
            request: Decomposition request

        Returns:
            DecompositionResult with tree and execution plan
        """
        self.logger.info(
            f"Decomposing task: '{request.task_description[:50]}...' "
            f"(max_depth: {request.max_depth})"
        )

        # Decompose task
        result = await self.decomposer.decompose(request)

        # Assign models if requested and available
        if (
            request.target_model_routing
            and self.task_assigner
            and request.estimate_costs
        ):
            try:
                estimated_cost = await self.task_assigner.assign_models(
                    tree=result.tree,
                    budget_remaining=None,  # Use default budget
                )

                self.logger.info(
                    f"Model assignment complete: ${estimated_cost:.2f} estimated cost"
                )

            except Exception as e:
                self.logger.error(f"Model assignment failed: {e}")
                result.warnings.append(f"Model assignment failed: {str(e)}")

        self.logger.info(
            f"Decomposition complete: {result.tree.total_tasks} tasks, "
            f"{len(result.tree.leaf_tasks)} leaf tasks, "
            f"{len(result.execution_plan)} execution batches"
        )

        return result

    async def track_progress(
        self,
        tree: DecompositionTree,
        progress_update: ProgressUpdate,
    ) -> None:
        """
        Update progress for a task in the tree.

        PATTERN: Update task -> Propagate to parents
        CRITICAL: Maintain progress consistency

        Args:
            tree: Decomposition tree
            progress_update: Progress update
        """
        await self.progress_tracker.update_progress(
            tree=tree,
            task_id=progress_update.task_id,
            progress=progress_update.progress,
            status=progress_update.status,
            message=progress_update.message,
        )

    async def mark_task_failed(
        self,
        tree: DecompositionTree,
        task_id: str,
        error_message: str,
    ) -> None:
        """
        Mark a task as failed.

        Args:
            tree: Decomposition tree
            task_id: Task ID
            error_message: Error message
        """
        await self.progress_tracker.mark_task_failed(
            tree=tree,
            task_id=task_id,
            error_message=error_message,
        )

    def get_ready_tasks(
        self,
        tree: DecompositionTree,
    ) -> list[str]:
        """
        Get tasks ready for execution based on dependencies.

        PATTERN: Filter by completion status and dependencies
        CRITICAL: Only return tasks with all dependencies met

        Args:
            tree: Decomposition tree

        Returns:
            List of task IDs ready for execution
        """
        # Build dependency graph from tree
        self.dependency_manager.build_from_tree(tree)

        # Get completed task IDs
        completed_task_ids = {
            task_id
            for task_id, task in tree.tasks.items()
            if task.status == TaskStatus.COMPLETED
        }

        # Get ready tasks
        ready_task_ids = self.dependency_manager.get_ready_tasks(completed_task_ids)

        # Filter to only leaf tasks that are READY or PENDING
        ready_leaf_tasks = [
            task_id
            for task_id in ready_task_ids
            if task_id in tree.leaf_tasks
            and tree.tasks[task_id].status in [TaskStatus.READY, TaskStatus.PENDING]
        ]

        self.logger.debug(f"Found {len(ready_leaf_tasks)} ready tasks")

        return ready_leaf_tasks

    def get_progress_summary(self, tree: DecompositionTree) -> dict:
        """
        Get progress summary for the tree.

        Args:
            tree: Decomposition tree

        Returns:
            Progress summary dictionary
        """
        return self.progress_tracker.get_progress_summary(tree)

    def visualize_tree(
        self,
        tree: DecompositionTree,
        max_depth: int = 3,
    ) -> str:
        """
        Generate ASCII visualization of the decomposition tree.

        Args:
            tree: Decomposition tree
            max_depth: Maximum depth to visualize

        Returns:
            String visualization
        """
        return self.progress_tracker.visualize_progress(tree, max_depth)

    def get_execution_plan(self, tree: DecompositionTree) -> list[list[str]]:
        """
        Get execution plan with parallel batches.

        PATTERN: Build dependency graph -> Get batches
        CRITICAL: Filter to leaf tasks only

        Args:
            tree: Decomposition tree

        Returns:
            List of execution batches
        """
        # Build dependency graph
        self.dependency_manager.build_from_tree(tree)

        # Get all batches
        all_batches = self.dependency_manager.get_execution_batches()

        # Filter to only leaf tasks
        leaf_task_set = set(tree.leaf_tasks)
        leaf_batches = [
            [task_id for task_id in batch if task_id in leaf_task_set]
            for batch in all_batches
        ]

        # Remove empty batches
        leaf_batches = [batch for batch in leaf_batches if batch]

        return leaf_batches

    async def optimize_cost(
        self,
        tree: DecompositionTree,
        budget: float,
    ) -> None:
        """
        Optimize model assignments to fit within budget.

        Args:
            tree: Decomposition tree
            budget: Maximum budget in USD
        """
        if not self.task_assigner:
            self.logger.warning("Task assigner not available, cannot optimize cost")
            return

        await self.task_assigner.optimize_assignments(tree, budget)

    def get_assignment_summary(self, tree: DecompositionTree) -> dict:
        """
        Get summary of model assignments.

        Args:
            tree: Decomposition tree

        Returns:
            Assignment summary dictionary
        """
        if not self.task_assigner:
            return {"error": "Task assigner not available"}

        return self.task_assigner.get_assignment_summary(tree)

    async def reset_task(
        self,
        tree: DecompositionTree,
        task_id: str,
    ) -> None:
        """
        Reset a task for retry.

        Args:
            tree: Decomposition tree
            task_id: Task ID to reset
        """
        await self.progress_tracker.reset_task_progress(tree, task_id)
