"""Research task decomposition strategy."""

import logging
from typing import List
from ..base import BaseDecomposer
from ...models.decomposition_models import Task, TaskType


logger = logging.getLogger(__name__)


class ResearchDecompositionStrategy(BaseDecomposer):
    """
    Decomposition strategy for research tasks.

    PATTERN: Break down by topics, sources, analysis steps
    """

    def __init__(self, max_subtasks: int = 5, min_complexity: float = 0.25):
        """Initialize research decomposition strategy."""
        super().__init__(
            task_type=TaskType.RESEARCH,
            max_subtasks=max_subtasks,
            min_complexity=min_complexity,
        )

    async def decompose(self, task: Task) -> List[Task]:
        """
        Decompose research task into subtasks.

        Args:
            task: Task to decompose

        Returns:
            List of subtasks
        """
        subtasks = [
            self._create_subtask(
                name="Gather information sources",
                description=f"Gather relevant information sources for: {task.description}",
                parent_task=task,
            ),
            self._create_subtask(
                name="Analyze findings",
                description=f"Analyze and synthesize findings for: {task.description}",
                parent_task=task,
                task_type=TaskType.ANALYSIS,
            ),
            self._create_subtask(
                name="Document results",
                description=f"Document research results for: {task.description}",
                parent_task=task,
                task_type=TaskType.DOCUMENTATION,
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
        Estimate complexity for research task.

        Args:
            task: Task to estimate
            parent_complexity: Parent complexity

        Returns:
            Complexity score (0-1)
        """
        base_complexity = 0.45

        if parent_complexity is not None:
            base_complexity = (base_complexity * 0.7) + (parent_complexity * 0.3)

        return max(0.0, min(1.0, base_complexity))
