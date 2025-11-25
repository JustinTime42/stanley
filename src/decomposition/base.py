"""Base decomposer abstract class for task decomposition strategies."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional
from ..models.decomposition_models import Task, TaskType


logger = logging.getLogger(__name__)


class BaseDecomposer(ABC):
    """
    Abstract base class for task decomposition strategies.

    All decomposition strategies must:
    - Implement async decompose() method
    - Implement async estimate_complexity() method
    - Return list of Task objects representing subtasks
    - Follow deterministic decomposition for checkpointing
    """

    def __init__(
        self,
        task_type: TaskType,
        max_subtasks: int = 5,
        min_complexity: float = 0.3,
    ):
        """
        Initialize base decomposer.

        Args:
            task_type: Type of tasks this decomposer handles
            max_subtasks: Maximum subtasks to generate per decomposition
            min_complexity: Minimum complexity threshold for decomposition
        """
        self.task_type = task_type
        self.max_subtasks = max_subtasks
        self.min_complexity = min_complexity
        self.logger = logging.getLogger(f"{__name__}.{task_type.value}")

    @abstractmethod
    async def decompose(self, task: Task) -> List[Task]:
        """
        Decompose a task into subtasks.

        CRITICAL: Must be async and return list of Task objects
        CRITICAL: Decomposition should be deterministic for same input
        PATTERN: Create subtasks with parent_id set to task.id

        Args:
            task: Task to decompose

        Returns:
            List of subtasks
        """
        pass

    @abstractmethod
    async def estimate_complexity(
        self,
        task: Task,
        parent_complexity: Optional[float] = None,
    ) -> float:
        """
        Estimate task complexity (0-1 scale).

        CRITICAL: Must complete quickly (<50ms)
        PATTERN: Consider parent complexity for inheritance
        GOTCHA: Return value should be normalized to 0-1 range

        Args:
            task: Task to estimate
            parent_complexity: Complexity of parent task (if any)

        Returns:
            Complexity score (0-1)
        """
        pass

    def can_decompose(self, task: Task) -> bool:
        """
        Check if a task can be further decomposed.

        PATTERN: Check complexity threshold and decomposition flag
        GOTCHA: Respect max depth limits set elsewhere

        Args:
            task: Task to check

        Returns:
            True if task can be decomposed
        """
        # Cannot decompose if explicitly marked
        if not task.can_decompose:
            return False

        # Cannot decompose if already a leaf
        if task.is_leaf:
            return False

        # Cannot decompose if complexity too low
        if task.estimated_complexity < self.min_complexity:
            return False

        return True

    def should_decompose(self, task: Task, current_depth: int, max_depth: int) -> bool:
        """
        Determine if a task should be decomposed based on all factors.

        PATTERN: Combine multiple checks for decomposition decision
        CRITICAL: Prevent infinite recursion with depth limit

        Args:
            task: Task to check
            current_depth: Current depth in tree
            max_depth: Maximum allowed depth

        Returns:
            True if task should be decomposed
        """
        # Check depth limit
        if current_depth >= max_depth:
            self.logger.debug(
                f"Task {task.id} at max depth {current_depth}, cannot decompose"
            )
            return False

        # Check if can decompose
        if not self.can_decompose(task):
            self.logger.debug(
                f"Task {task.id} cannot be decomposed "
                f"(complexity: {task.estimated_complexity:.2f})"
            )
            return False

        return True

    def _create_subtask(
        self,
        name: str,
        description: str,
        parent_task: Task,
        task_type: Optional[TaskType] = None,
    ) -> Task:
        """
        Create a subtask with proper parent linkage.

        PATTERN: Helper method for consistent subtask creation
        CRITICAL: Subtask IDs must be unique across entire tree

        Args:
            name: Subtask name
            description: Subtask description
            parent_task: Parent task
            task_type: Task type (defaults to parent's type)

        Returns:
            New Task object
        """
        from uuid import uuid4

        subtask = Task(
            id=str(uuid4()),
            parent_id=parent_task.id,
            name=name,
            description=description,
            type=task_type or parent_task.type,
            depth=parent_task.depth + 1,
            status=parent_task.status,
        )

        return subtask

    async def validate_subtasks(self, subtasks: List[Task]) -> bool:
        """
        Validate generated subtasks.

        PATTERN: Post-decomposition validation
        GOTCHA: Ensure subtasks don't exceed max_subtasks limit

        Args:
            subtasks: List of generated subtasks

        Returns:
            True if valid, False otherwise
        """
        if not subtasks:
            self.logger.warning("Decomposition generated no subtasks")
            return False

        if len(subtasks) > self.max_subtasks:
            self.logger.warning(
                f"Decomposition generated {len(subtasks)} subtasks, "
                f"exceeds max of {self.max_subtasks}"
            )
            return False

        # Check all subtasks have unique IDs
        ids = [t.id for t in subtasks]
        if len(ids) != len(set(ids)):
            self.logger.error("Duplicate subtask IDs detected")
            return False

        return True
