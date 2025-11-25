"""Testing task decomposition strategy."""

import logging
from typing import List
from ..base import BaseDecomposer
from ...models.decomposition_models import Task, TaskType


logger = logging.getLogger(__name__)


class TestingDecompositionStrategy(BaseDecomposer):
    """
    Decomposition strategy for testing tasks.

    PATTERN: Break down by test types, coverage areas
    CRITICAL: Align with code structure
    """

    def __init__(self, max_subtasks: int = 5, min_complexity: float = 0.2):
        """Initialize testing decomposition strategy."""
        super().__init__(
            task_type=TaskType.TESTING,
            max_subtasks=max_subtasks,
            min_complexity=min_complexity,
        )

    async def decompose(self, task: Task) -> List[Task]:
        """
        Decompose testing task into subtasks.

        Args:
            task: Task to decompose

        Returns:
            List of subtasks
        """
        subtasks = [
            self._create_subtask(
                name="Create unit tests",
                description=f"Create unit tests for: {task.description}",
                parent_task=task,
            ),
            self._create_subtask(
                name="Create integration tests",
                description=f"Create integration tests for: {task.description}",
                parent_task=task,
            ),
            self._create_subtask(
                name="Add test fixtures and mocks",
                description=f"Add test fixtures and mocks for: {task.description}",
                parent_task=task,
            ),
        ]

        if await self.validate_subtasks(subtasks):
            return subtasks

        return [task]

    async def estimate_complexity(
        self,
        task: Task,
        parent_complexity: float = None,
    ) -> float:
        """
        Estimate complexity for testing task.

        Args:
            task: Task to estimate
            parent_complexity: Parent complexity

        Returns:
            Complexity score (0-1)
        """
        # Testing tasks are generally medium-low complexity
        base_complexity = 0.4

        if parent_complexity is not None:
            base_complexity = (base_complexity * 0.7) + (parent_complexity * 0.3)

        return max(0.0, min(1.0, base_complexity))
