"""Code generation/modification decomposition strategy."""

import logging
from typing import List
from ..base import BaseDecomposer
from ...models.decomposition_models import Task, TaskType


logger = logging.getLogger(__name__)


class CodeDecompositionStrategy(BaseDecomposer):
    """
    Decomposition strategy for code generation and modification tasks.

    PATTERN: Break down by components, functions, files
    CRITICAL: Consider code dependencies and modularity
    """

    def __init__(self, max_subtasks: int = 5, min_complexity: float = 0.3):
        """Initialize code decomposition strategy."""
        super().__init__(
            task_type=TaskType.CODE_GENERATION,
            max_subtasks=max_subtasks,
            min_complexity=min_complexity,
        )

    async def decompose(self, task: Task) -> List[Task]:
        """
        Decompose code task into subtasks.

        PATTERN: Analyze description for components, files, modules
        CRITICAL: Create subtasks with clear boundaries

        Args:
            task: Task to decompose

        Returns:
            List of subtasks
        """
        subtasks = []
        description_lower = task.description.lower()

        # Pattern 1: File/Module creation
        if any(
            kw in description_lower for kw in ["create file", "new file", "implement"]
        ):
            subtasks.extend(self._decompose_file_creation(task))

        # Pattern 2: Feature implementation
        elif any(
            kw in description_lower for kw in ["feature", "functionality", "system"]
        ):
            subtasks.extend(self._decompose_feature(task))

        # Pattern 3: API/Interface creation
        elif any(kw in description_lower for kw in ["api", "interface", "endpoint"]):
            subtasks.extend(self._decompose_api(task))

        # Default: Generic breakdown
        else:
            subtasks.extend(self._decompose_generic(task))

        # Validate and return
        if await self.validate_subtasks(subtasks):
            return subtasks

        # Fallback: single task
        return [task]

    def _decompose_file_creation(self, task: Task) -> List[Task]:
        """Decompose file creation task."""
        subtasks = [
            self._create_subtask(
                name="Define file structure and interfaces",
                description=f"Define the structure, interfaces, and types for: {task.description}",
                parent_task=task,
                task_type=TaskType.ARCHITECTURE,
            ),
            self._create_subtask(
                name="Implement core logic",
                description=f"Implement the main logic for: {task.description}",
                parent_task=task,
            ),
            self._create_subtask(
                name="Add error handling and validation",
                description=f"Add error handling and input validation for: {task.description}",
                parent_task=task,
            ),
        ]
        return subtasks

    def _decompose_feature(self, task: Task) -> List[Task]:
        """Decompose feature implementation task."""
        subtasks = [
            self._create_subtask(
                name="Design feature architecture",
                description=f"Design the architecture and components for: {task.description}",
                parent_task=task,
                task_type=TaskType.ARCHITECTURE,
            ),
            self._create_subtask(
                name="Implement data models",
                description=f"Create data models and schemas for: {task.description}",
                parent_task=task,
            ),
            self._create_subtask(
                name="Implement business logic",
                description=f"Implement the core business logic for: {task.description}",
                parent_task=task,
            ),
            self._create_subtask(
                name="Add integration points",
                description=f"Add integration with existing systems for: {task.description}",
                parent_task=task,
            ),
        ]
        return subtasks

    def _decompose_api(self, task: Task) -> List[Task]:
        """Decompose API creation task."""
        subtasks = [
            self._create_subtask(
                name="Define API schema",
                description=f"Define request/response schemas for: {task.description}",
                parent_task=task,
                task_type=TaskType.ARCHITECTURE,
            ),
            self._create_subtask(
                name="Implement endpoints",
                description=f"Implement API endpoints for: {task.description}",
                parent_task=task,
            ),
            self._create_subtask(
                name="Add authentication/authorization",
                description=f"Add auth and permissions for: {task.description}",
                parent_task=task,
            ),
        ]
        return subtasks

    def _decompose_generic(self, task: Task) -> List[Task]:
        """Generic code task decomposition."""
        subtasks = [
            self._create_subtask(
                name="Plan implementation approach",
                description=f"Plan the implementation approach for: {task.description}",
                parent_task=task,
                task_type=TaskType.ARCHITECTURE,
            ),
            self._create_subtask(
                name="Implement main functionality",
                description=f"Implement the main functionality for: {task.description}",
                parent_task=task,
            ),
            self._create_subtask(
                name="Add tests and documentation",
                description=f"Add tests and documentation for: {task.description}",
                parent_task=task,
                task_type=TaskType.TESTING,
            ),
        ]
        return subtasks

    async def estimate_complexity(
        self,
        task: Task,
        parent_complexity: float = None,
    ) -> float:
        """
        Estimate complexity for code task.

        PATTERN: Consider LOC estimates, dependencies, novelty
        CRITICAL: Code tasks tend to be medium-high complexity

        Args:
            task: Task to estimate
            parent_complexity: Parent complexity

        Returns:
            Complexity score (0-1)
        """
        base_complexity = 0.5  # Default for code tasks

        description_lower = task.description.lower()

        # Increase complexity for certain patterns
        if any(
            kw in description_lower
            for kw in ["complex", "advanced", "system", "architecture"]
        ):
            base_complexity += 0.2

        if any(kw in description_lower for kw in ["integration", "api", "distributed"]):
            base_complexity += 0.15

        if any(
            kw in description_lower
            for kw in ["performance", "optimization", "scalable"]
        ):
            base_complexity += 0.1

        # Decrease for simple tasks
        if any(kw in description_lower for kw in ["simple", "basic", "small", "fix"]):
            base_complexity -= 0.2

        # Apply parent inheritance if available
        if parent_complexity is not None:
            base_complexity = (base_complexity * 0.7) + (parent_complexity * 0.3)

        # Normalize
        return max(0.0, min(1.0, base_complexity))
