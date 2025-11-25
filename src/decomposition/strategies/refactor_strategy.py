"""Refactoring task decomposition strategy."""

import logging
from typing import List
from ..base import BaseDecomposer
from ...models.decomposition_models import Task, TaskType


logger = logging.getLogger(__name__)


class RefactorDecompositionStrategy(BaseDecomposer):
    """
    Decomposition strategy for refactoring tasks.

    PATTERN: Break down by code areas, refactoring steps
    CRITICAL: Ensure backward compatibility
    """

    def __init__(self, max_subtasks: int = 5, min_complexity: float = 0.3):
        """Initialize refactoring decomposition strategy."""
        super().__init__(
            task_type=TaskType.REFACTORING,
            max_subtasks=max_subtasks,
            min_complexity=min_complexity,
        )

    async def decompose(self, task: Task) -> List[Task]:
        """
        Decompose refactoring task into subtasks.

        Args:
            task: Task to decompose

        Returns:
            List of subtasks
        """
        subtasks = [
            self._create_subtask(
                name="Analyze current code structure",
                description=f"Analyze current structure for: {task.description}",
                parent_task=task,
                task_type=TaskType.ANALYSIS,
            ),
            self._create_subtask(
                name="Plan refactoring approach",
                description=f"Plan refactoring strategy for: {task.description}",
                parent_task=task,
                task_type=TaskType.ARCHITECTURE,
            ),
            self._create_subtask(
                name="Apply refactoring changes",
                description=f"Apply refactoring changes for: {task.description}",
                parent_task=task,
                task_type=TaskType.CODE_MODIFICATION,
            ),
            self._create_subtask(
                name="Verify backward compatibility",
                description=f"Verify tests pass and compatibility for: {task.description}",
                parent_task=task,
                task_type=TaskType.TESTING,
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
        Estimate complexity for refactoring task.

        Args:
            task: Task to estimate
            parent_complexity: Parent complexity

        Returns:
            Complexity score (0-1)
        """
        # Refactoring is medium-high complexity
        base_complexity = 0.6

        if parent_complexity is not None:
            base_complexity = (base_complexity * 0.7) + (parent_complexity * 0.3)

        return max(0.0, min(1.0, base_complexity))
