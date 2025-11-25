"""Complexity estimation for subtasks with inheritance from parent tasks."""

import logging
from typing import Optional
from ..llm.analyzer import TaskComplexityAnalyzer
from ..models.decomposition_models import Task, TaskType
from ..models.llm_models import TaskComplexity


logger = logging.getLogger(__name__)


class ComplexityEstimator:
    """
    Estimates task complexity for decomposition decisions and model assignment.

    PATTERN: Multi-factor complexity estimation with parent inheritance
    CRITICAL: Must complete in <100ms per task
    GOTCHA: Balance between accuracy and speed
    """

    def __init__(self, task_analyzer: Optional[TaskComplexityAnalyzer] = None):
        """
        Initialize complexity estimator.

        Args:
            task_analyzer: Task complexity analyzer (creates default if None)
        """
        self.task_analyzer = task_analyzer or TaskComplexityAnalyzer()
        self.logger = logging.getLogger(__name__)

        # Task type complexity modifiers
        self.type_modifiers = {
            TaskType.ARCHITECTURE: 1.2,
            TaskType.REFACTORING: 1.1,
            TaskType.DEBUGGING: 1.15,
            TaskType.CODE_GENERATION: 1.0,
            TaskType.CODE_MODIFICATION: 0.9,
            TaskType.TESTING: 0.8,
            TaskType.DOCUMENTATION: 0.6,
            TaskType.ANALYSIS: 0.9,
            TaskType.RESEARCH: 0.85,
        }

    async def estimate_task_complexity(
        self,
        task: Task,
        parent_complexity: Optional[float] = None,
    ) -> float:
        """
        Estimate task complexity on 0-1 scale.

        PATTERN: Combine base estimation with parent inheritance and type modifiers
        CRITICAL: Ensure result is normalized to 0-1 range

        Args:
            task: Task to estimate
            parent_complexity: Complexity of parent task (for inheritance)

        Returns:
            Complexity score (0-1)
        """
        # Base estimation from task description
        analysis = self.task_analyzer.analyze_task(
            task_description=task.description,
            agent_role=task.type.value,
        )

        # Map TaskComplexity enum to normalized score
        base_complexity = self._map_to_normalized_score(analysis.complexity)

        # Apply parent complexity inheritance if available
        if parent_complexity is not None:
            # Inherit 30% from parent, 70% from analysis
            inherited_factor = 0.3
            estimated = (base_complexity * 0.7) + (parent_complexity * inherited_factor)
        else:
            estimated = base_complexity

        # Apply task type modifier
        modifier = self.type_modifiers.get(task.type, 1.0)
        final_complexity = estimated * modifier

        # Normalize to 0-1 range
        final_complexity = max(0.0, min(1.0, final_complexity))

        self.logger.debug(
            f"Estimated complexity for '{task.name}': {final_complexity:.2f} "
            f"(base: {base_complexity:.2f}, type: {task.type.value}, "
            f"modifier: {modifier:.2f})"
        )

        return final_complexity

    def _map_to_normalized_score(self, complexity: TaskComplexity) -> float:
        """
        Map TaskComplexity enum to normalized 0-1 score.

        PATTERN: Simple linear mapping
        CRITICAL: Maintain consistency with routing thresholds

        Args:
            complexity: TaskComplexity enum

        Returns:
            Normalized score (0-1)
        """
        mapping = {
            TaskComplexity.SIMPLE: 0.25,
            TaskComplexity.MEDIUM: 0.55,
            TaskComplexity.COMPLEX: 0.85,
        }

        return mapping.get(complexity, 0.5)

    async def estimate_subtasks_complexity(
        self,
        subtasks: list[Task],
        parent_complexity: float,
    ) -> None:
        """
        Estimate complexity for all subtasks in batch.

        PATTERN: Batch estimation with parent inheritance
        GOTCHA: Updates Task objects in place

        Args:
            subtasks: List of subtasks to estimate
            parent_complexity: Parent task complexity
        """
        for subtask in subtasks:
            complexity = await self.estimate_task_complexity(
                subtask,
                parent_complexity=parent_complexity,
            )
            subtask.estimated_complexity = complexity

        self.logger.info(
            f"Estimated complexity for {len(subtasks)} subtasks "
            f"(parent: {parent_complexity:.2f})"
        )

    def propagate_complexity(
        self,
        parent_task: Task,
        subtasks: list[Task],
    ) -> float:
        """
        Calculate parent complexity from subtasks.

        PATTERN: Weighted average based on subtask complexities
        CRITICAL: Use when parent complexity unknown but subtasks estimated

        Args:
            parent_task: Parent task
            subtasks: List of subtasks

        Returns:
            Calculated parent complexity
        """
        if not subtasks:
            return parent_task.estimated_complexity

        # Calculate weighted average
        total_complexity = sum(t.estimated_complexity for t in subtasks)
        avg_complexity = total_complexity / len(subtasks)

        # Apply slight multiplier for composition complexity
        composition_factor = 1.1  # 10% overhead for coordinating subtasks
        parent_complexity = min(1.0, avg_complexity * composition_factor)

        self.logger.debug(
            f"Propagated complexity to parent: {parent_complexity:.2f} "
            f"(from {len(subtasks)} subtasks)"
        )

        return parent_complexity

    async def refine_estimate(
        self,
        task: Task,
        actual_complexity: float,
    ) -> None:
        """
        Refine complexity estimate based on actual execution.

        PATTERN: Learning from execution for future estimates
        GOTCHA: Store actual complexity for correlation analysis

        Args:
            task: Task that was executed
            actual_complexity: Observed complexity from execution
        """
        task.actual_complexity = actual_complexity

        # Calculate estimation error
        if task.estimated_complexity > 0:
            error = abs(actual_complexity - task.estimated_complexity)
            error_pct = (error / task.estimated_complexity) * 100

            self.logger.info(
                f"Complexity estimate refinement for {task.id}: "
                f"estimated={task.estimated_complexity:.2f}, "
                f"actual={actual_complexity:.2f}, "
                f"error={error_pct:.1f}%"
            )

    def get_complexity_distribution(self, tasks: list[Task]) -> dict[str, int]:
        """
        Get distribution of task complexities.

        PATTERN: Categorize tasks by complexity for analysis
        GOTCHA: Useful for balancing task assignment

        Args:
            tasks: List of tasks

        Returns:
            Dictionary with counts for each complexity level
        """
        distribution = {
            "simple": 0,  # < 0.3
            "medium": 0,  # 0.3 - 0.7
            "complex": 0,  # > 0.7
        }

        for task in tasks:
            if task.estimated_complexity < 0.3:
                distribution["simple"] += 1
            elif task.estimated_complexity < 0.7:
                distribution["medium"] += 1
            else:
                distribution["complex"] += 1

        return distribution

    async def estimate_tokens(self, task: Task) -> int:
        """
        Estimate token usage for task execution.

        PATTERN: Based on task description and complexity
        CRITICAL: Used for cost estimation

        Args:
            task: Task to estimate

        Returns:
            Estimated token count
        """
        # Base token count from description length
        desc_words = len(task.description.split())
        base_tokens = int(desc_words * 1.3)  # ~1.3 tokens per word

        # Multiply by complexity factor
        complexity_multiplier = 1 + (task.estimated_complexity * 2)  # 1x to 3x
        estimated_tokens = int(base_tokens * complexity_multiplier)

        # Add overhead for task coordination
        overhead_tokens = 100
        total_tokens = estimated_tokens + overhead_tokens

        task.estimated_tokens = total_tokens

        return total_tokens
