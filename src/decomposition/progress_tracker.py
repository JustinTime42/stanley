"""Hierarchical progress tracking with bottom-up aggregation."""

import logging
from datetime import datetime
from typing import Optional
from ..models.decomposition_models import (
    TaskStatus,
    DecompositionTree,
)


logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Tracks progress hierarchically with bottom-up aggregation.

    PATTERN: Update leaf task -> propagate to parents recursively
    CRITICAL: Progress updates should be atomic to prevent race conditions
    GOTCHA: Use Redis transactions for distributed scenarios
    """

    def __init__(self, redis_client=None):
        """
        Initialize progress tracker.

        Args:
            redis_client: Optional Redis client for atomic updates
        """
        self.redis = redis_client
        self.logger = logging.getLogger(__name__)

    async def update_progress(
        self,
        tree: DecompositionTree,
        task_id: str,
        progress: float,
        status: Optional[TaskStatus] = None,
        message: Optional[str] = None,
    ) -> None:
        """
        Update task progress and propagate to parents.

        PATTERN: Bottom-up aggregation through task hierarchy
        CRITICAL: Ensure atomic update if using Redis

        Args:
            tree: Decomposition tree
            task_id: Task ID to update
            progress: New progress value (0-100)
            status: Optional new status
            message: Optional progress message
        """
        if task_id not in tree.tasks:
            self.logger.error(f"Task {task_id} not found in tree")
            return

        task = tree.tasks[task_id]

        # Update task progress
        old_progress = task.progress
        task.progress = min(100.0, max(0.0, progress))

        # Update status if provided
        if status:
            task.status = status

        # Update timing
        if progress >= 100 and not task.end_time:
            task.end_time = datetime.now()
            task.status = TaskStatus.COMPLETED

            # Calculate execution time
            if task.start_time:
                delta = task.end_time - task.start_time
                task.execution_time_ms = int(delta.total_seconds() * 1000)

        elif progress > 0 and not task.start_time:
            task.start_time = datetime.now()
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.IN_PROGRESS

        self.logger.debug(
            f"Updated task {task_id} progress: {old_progress:.1f}% -> {progress:.1f}% "
            f"(status: {task.status.value})"
        )

        # Propagate to parent
        if task.parent_id:
            await self._update_parent_progress(tree, task.parent_id)

        # Update tree-level statistics
        self._update_tree_stats(tree)

        # Update tree timestamp
        tree.updated_at = datetime.now()

    async def _update_parent_progress(
        self,
        tree: DecompositionTree,
        parent_id: str,
    ) -> None:
        """
        Calculate and update parent progress from children.

        PATTERN: Weighted average based on complexity
        CRITICAL: Recursively propagate to ancestors

        Args:
            tree: Decomposition tree
            parent_id: Parent task ID
        """
        if parent_id not in tree.tasks:
            return

        parent = tree.tasks[parent_id]

        # Get all children
        children = [task for task in tree.tasks.values() if task.parent_id == parent_id]

        if not children:
            return

        # Calculate weighted progress
        total_weight = sum(c.estimated_complexity for c in children)

        if total_weight == 0:
            # Simple average if no complexity weights
            parent.progress = sum(c.progress for c in children) / len(children)
        else:
            # Weighted average based on complexity
            weighted_sum = sum(c.progress * c.estimated_complexity for c in children)
            parent.progress = weighted_sum / total_weight

        # Update subtask counts
        parent.subtask_count = len(children)
        parent.completed_subtask_count = sum(
            1 for c in children if c.status == TaskStatus.COMPLETED
        )

        # Update parent status based on children
        if all(c.status == TaskStatus.COMPLETED for c in children):
            parent.status = TaskStatus.COMPLETED
            if not parent.end_time:
                parent.end_time = datetime.now()
        elif any(c.status == TaskStatus.FAILED for c in children):
            parent.status = TaskStatus.FAILED
        elif any(c.status == TaskStatus.IN_PROGRESS for c in children):
            parent.status = TaskStatus.IN_PROGRESS
        elif any(c.status == TaskStatus.BLOCKED for c in children):
            parent.status = TaskStatus.BLOCKED

        self.logger.debug(
            f"Updated parent {parent_id} progress: {parent.progress:.1f}% "
            f"({parent.completed_subtask_count}/{parent.subtask_count} subtasks)"
        )

        # Recursively update parent's parent
        if parent.parent_id:
            await self._update_parent_progress(tree, parent.parent_id)

    def _update_tree_stats(self, tree: DecompositionTree) -> None:
        """
        Update tree-level statistics.

        PATTERN: Aggregate across all tasks
        GOTCHA: Cache these values for performance

        Args:
            tree: Decomposition tree
        """
        # Count tasks by status
        tree.completed_tasks = sum(
            1 for t in tree.tasks.values() if t.status == TaskStatus.COMPLETED
        )

        tree.failed_tasks = sum(
            1 for t in tree.tasks.values() if t.status == TaskStatus.FAILED
        )

        tree.blocked_tasks = sum(
            1 for t in tree.tasks.values() if t.status == TaskStatus.BLOCKED
        )

        # Calculate overall progress from root task
        if tree.root_task_id in tree.tasks:
            tree.overall_progress = tree.tasks[tree.root_task_id].progress

        # Calculate actual cost
        tree.actual_total_cost = sum(
            t.estimated_cost
            for t in tree.tasks.values()
            if t.status == TaskStatus.COMPLETED
        )

    async def mark_task_failed(
        self,
        tree: DecompositionTree,
        task_id: str,
        error_message: str,
    ) -> None:
        """
        Mark a task as failed with error message.

        PATTERN: Update status and propagate to parents
        CRITICAL: Failed tasks may block dependents

        Args:
            tree: Decomposition tree
            task_id: Task ID
            error_message: Error message
        """
        if task_id not in tree.tasks:
            self.logger.error(f"Task {task_id} not found in tree")
            return

        task = tree.tasks[task_id]
        task.status = TaskStatus.FAILED
        task.error_message = error_message
        task.end_time = datetime.now()

        self.logger.warning(f"Task {task_id} failed: {error_message}")

        # Propagate failure to parent
        if task.parent_id:
            await self._update_parent_progress(tree, task.parent_id)

        self._update_tree_stats(tree)

    async def reset_task_progress(
        self,
        tree: DecompositionTree,
        task_id: str,
    ) -> None:
        """
        Reset task progress for retry.

        PATTERN: Clear progress, timing, and status
        GOTCHA: Increment retry count

        Args:
            tree: Decomposition tree
            task_id: Task ID to reset
        """
        if task_id not in tree.tasks:
            self.logger.error(f"Task {task_id} not found in tree")
            return

        task = tree.tasks[task_id]
        task.progress = 0.0
        task.status = TaskStatus.PENDING
        task.start_time = None
        task.end_time = None
        task.execution_time_ms = None
        task.error_message = None
        task.retry_count += 1

        self.logger.info(f"Reset task {task_id} for retry (attempt {task.retry_count})")

        # Update parent
        if task.parent_id:
            await self._update_parent_progress(tree, task.parent_id)

    def get_progress_summary(self, tree: DecompositionTree) -> dict:
        """
        Get progress summary for the tree.

        PATTERN: Aggregate statistics for reporting
        GOTCHA: Useful for dashboards and monitoring

        Args:
            tree: Decomposition tree

        Returns:
            Dictionary with progress statistics
        """
        total_tasks = len(tree.tasks)
        leaf_tasks = len(tree.leaf_tasks)

        # Calculate task distribution
        status_counts = {}
        for task in tree.tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # Calculate average progress for leaf tasks
        if tree.leaf_tasks:
            leaf_progress_sum = sum(tree.tasks[tid].progress for tid in tree.leaf_tasks)
            avg_leaf_progress = leaf_progress_sum / len(tree.leaf_tasks)
        else:
            avg_leaf_progress = 0.0

        # Calculate estimated time remaining
        completed_count = tree.completed_tasks
        if completed_count > 0 and tree.root_task_id in tree.tasks:
            root_task = tree.tasks[tree.root_task_id]
            if root_task.start_time:
                elapsed = (datetime.now() - root_task.start_time).total_seconds()
                completion_rate = completed_count / elapsed  # tasks per second
                remaining_tasks = total_tasks - completed_count

                if completion_rate > 0:
                    estimated_remaining_seconds = remaining_tasks / completion_rate
                else:
                    estimated_remaining_seconds = 0
            else:
                estimated_remaining_seconds = 0
        else:
            estimated_remaining_seconds = 0

        return {
            "total_tasks": total_tasks,
            "leaf_tasks": leaf_tasks,
            "overall_progress": tree.overall_progress,
            "completed_tasks": tree.completed_tasks,
            "failed_tasks": tree.failed_tasks,
            "blocked_tasks": tree.blocked_tasks,
            "status_distribution": status_counts,
            "average_leaf_progress": avg_leaf_progress,
            "estimated_remaining_seconds": int(estimated_remaining_seconds),
            "actual_cost": tree.actual_total_cost,
            "estimated_cost": tree.estimated_total_cost,
        }

    def visualize_progress(self, tree: DecompositionTree, max_depth: int = 3) -> str:
        """
        Generate ASCII visualization of progress tree.

        PATTERN: Recursive tree traversal with progress bars
        GOTCHA: Limit depth to avoid overwhelming output

        Args:
            tree: Decomposition tree
            max_depth: Maximum depth to visualize

        Returns:
            String visualization
        """
        lines = ["Progress Tree:"]
        lines.append("=" * 60)

        def visualize_task(task_id: str, depth: int, prefix: str = "") -> None:
            """Recursively visualize task and children."""
            if depth > max_depth:
                return

            if task_id not in tree.tasks:
                return

            task = tree.tasks[task_id]

            # Create progress bar
            bar_width = 20
            filled = int((task.progress / 100) * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)

            # Format line
            indent = "  " * depth
            status_icon = {
                TaskStatus.PENDING: "⏸",
                TaskStatus.IN_PROGRESS: "▶",
                TaskStatus.COMPLETED: "✓",
                TaskStatus.FAILED: "✗",
                TaskStatus.BLOCKED: "🚫",
                TaskStatus.READY: "⏵",
            }.get(task.status, "?")

            line = (
                f"{indent}{prefix}{status_icon} {task.name[:30]:<30} "
                f"[{bar}] {task.progress:5.1f}%"
            )
            lines.append(line)

            # Visualize children
            children = [t for t in tree.tasks.values() if t.parent_id == task_id]

            for i, child in enumerate(children):
                is_last = i == len(children) - 1
                child_prefix = "└─ " if is_last else "├─ "
                visualize_task(child.id, depth + 1, child_prefix)

        # Start with root
        visualize_task(tree.root_task_id, 0)

        return "\n".join(lines)
