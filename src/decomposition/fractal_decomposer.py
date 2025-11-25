"""Fractal task decomposer - recursive decomposition engine."""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from .base import BaseDecomposer
from .complexity_estimator import ComplexityEstimator
from .dependency_manager import DependencyManager
from .strategies import (
    CodeDecompositionStrategy,
    TestingDecompositionStrategy,
    ResearchDecompositionStrategy,
    RefactorDecompositionStrategy,
)
from ..models.decomposition_models import (
    Task,
    TaskType,
    TaskStatus,
    DecompositionTree,
    DecompositionRequest,
    DecompositionResult,
)


logger = logging.getLogger(__name__)


class FractalDecomposer:
    """
    Recursive task decomposition engine with fractal structure.

    PATTERN: Recursively decompose tasks with strategy selection
    CRITICAL: Must limit depth to prevent stack overflow
    GOTCHA: Same input should produce deterministic output for checkpointing
    """

    def __init__(
        self,
        max_depth: int = 10,
        complexity_threshold: float = 0.2,
        max_subtasks_per_level: int = 5,
    ):
        """
        Initialize fractal decomposer.

        Args:
            max_depth: Maximum recursion depth
            complexity_threshold: Minimum complexity to decompose
            max_subtasks_per_level: Maximum subtasks per decomposition
        """
        self.max_depth = max_depth
        self.complexity_threshold = complexity_threshold
        self.max_subtasks_per_level = max_subtasks_per_level

        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.complexity_estimator = ComplexityEstimator()
        self.dependency_manager = DependencyManager()

        # Register strategies
        self.strategies: Dict[TaskType, BaseDecomposer] = {}
        self._register_strategies()

    def _register_strategies(self) -> None:
        """Register decomposition strategies for each task type."""
        # Code strategies
        code_strategy = CodeDecompositionStrategy(
            max_subtasks=self.max_subtasks_per_level,
            min_complexity=self.complexity_threshold,
        )
        self.strategies[TaskType.CODE_GENERATION] = code_strategy
        self.strategies[TaskType.CODE_MODIFICATION] = code_strategy
        self.strategies[TaskType.ARCHITECTURE] = code_strategy

        # Testing strategy
        testing_strategy = TestingDecompositionStrategy(
            max_subtasks=self.max_subtasks_per_level,
            min_complexity=self.complexity_threshold,
        )
        self.strategies[TaskType.TESTING] = testing_strategy

        # Research strategy
        research_strategy = ResearchDecompositionStrategy(
            max_subtasks=self.max_subtasks_per_level,
            min_complexity=self.complexity_threshold,
        )
        self.strategies[TaskType.RESEARCH] = research_strategy
        self.strategies[TaskType.ANALYSIS] = research_strategy

        # Refactoring strategy
        refactor_strategy = RefactorDecompositionStrategy(
            max_subtasks=self.max_subtasks_per_level,
            min_complexity=self.complexity_threshold,
        )
        self.strategies[TaskType.REFACTORING] = refactor_strategy

        # Generic strategies for remaining types
        self.strategies[TaskType.DEBUGGING] = code_strategy
        self.strategies[TaskType.DOCUMENTATION] = research_strategy

        self.logger.info(f"Registered {len(self.strategies)} decomposition strategies")

    async def decompose(
        self,
        request: DecompositionRequest,
    ) -> DecompositionResult:
        """
        Decompose a task recursively into a tree of subtasks.

        PATTERN: Create root task -> decompose recursively -> build tree
        CRITICAL: Ensure deterministic decomposition for checkpointing

        Args:
            request: Decomposition request

        Returns:
            DecompositionResult with tree and execution plan
        """
        start_time = datetime.now()

        # Create root task
        root_task = Task(
            name=request.task_description[:50],  # Truncate for name
            description=request.task_description,
            type=request.task_type or self._infer_task_type(request.task_description),
            status=TaskStatus.PENDING,
            depth=0,
        )

        # Estimate root complexity
        root_task.estimated_complexity = (
            await self.complexity_estimator.estimate_task_complexity(root_task)
        )

        self.logger.info(
            f"Starting decomposition: '{root_task.name}' "
            f"(type: {root_task.type.value}, complexity: {root_task.estimated_complexity:.2f})"
        )

        # Decompose recursively
        all_tasks = await self._decompose_recursive(
            task=root_task,
            depth=0,
            max_depth=request.max_depth,
        )

        # Build decomposition tree
        tree = self._build_tree(root_task, all_tasks)

        # Build dependency graph if requested
        if request.include_dependencies:
            self._build_dependencies(tree)

        # Validate dependencies and get execution order
        validation = self.dependency_manager.validate_dependencies()

        if not validation.is_valid:
            self.logger.error(
                f"Dependency validation failed: {len(validation.cycles)} cycles detected"
            )
            warnings = [f"Circular dependencies detected: {validation.cycles}"]
        else:
            warnings = []
            tree.execution_order = validation.execution_order

        # Generate execution batches
        execution_plan = self.dependency_manager.get_execution_batches()

        # Calculate estimated duration
        estimated_duration_ms = self._estimate_duration(tree)

        # Execution time
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        self.logger.info(
            f"Decomposition complete: {len(all_tasks)} tasks, "
            f"{len(tree.leaf_tasks)} leaf tasks, "
            f"max depth: {tree.max_depth}, "
            f"time: {elapsed_ms}ms"
        )

        return DecompositionResult(
            tree=tree,
            execution_plan=execution_plan,
            estimated_duration_ms=estimated_duration_ms,
            warnings=warnings,
        )

    async def _decompose_recursive(
        self,
        task: Task,
        depth: int,
        max_depth: int,
    ) -> List[Task]:
        """
        Recursively decompose a task into subtasks.

        PATTERN: Base case checks -> strategy decomposition -> recursive calls
        CRITICAL: Prevent stack overflow with depth limit

        Args:
            task: Task to decompose
            depth: Current recursion depth
            max_depth: Maximum allowed depth

        Returns:
            List of all tasks (task + all descendants)
        """
        all_tasks = [task]

        # Base case 1: Max depth reached
        if depth >= max_depth:
            task.is_leaf = True
            task.can_decompose = False
            task.status = TaskStatus.READY
            return all_tasks

        # Base case 2: Complexity too low
        if task.estimated_complexity < self.complexity_threshold:
            task.is_leaf = True
            task.status = TaskStatus.READY
            return all_tasks

        # Select strategy
        strategy = self._select_strategy(task.type)

        # Check if should decompose
        if not strategy.should_decompose(task, depth, max_depth):
            task.is_leaf = True
            task.status = TaskStatus.READY
            return all_tasks

        # Decompose into subtasks
        try:
            subtasks = await strategy.decompose(task)

            if not subtasks or len(subtasks) == 0:
                # No valid decomposition, mark as leaf
                task.is_leaf = True
                task.status = TaskStatus.READY
                return all_tasks

            # Estimate complexity for subtasks
            for subtask in subtasks:
                subtask.estimated_complexity = (
                    await self.complexity_estimator.estimate_task_complexity(
                        subtask,
                        parent_complexity=task.estimated_complexity,
                    )
                )

            # Update task metadata
            task.subtask_count = len(subtasks)
            task.is_leaf = False
            task.status = TaskStatus.DECOMPOSING

            self.logger.debug(
                f"Decomposed '{task.name}' into {len(subtasks)} subtasks at depth {depth}"
            )

            # Recursively decompose each subtask
            for subtask in subtasks:
                subtask_and_descendants = await self._decompose_recursive(
                    task=subtask,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
                all_tasks.extend(subtask_and_descendants)

            return all_tasks

        except Exception as e:
            self.logger.error(f"Decomposition failed for task {task.id}: {e}")

            # Mark as leaf on error
            task.is_leaf = True
            task.can_decompose = False
            task.status = TaskStatus.READY
            task.error_message = str(e)

            return all_tasks

    def _select_strategy(self, task_type: TaskType) -> BaseDecomposer:
        """
        Select decomposition strategy based on task type.

        PATTERN: Direct lookup with fallback to default
        CRITICAL: Always return a valid strategy

        Args:
            task_type: Type of task

        Returns:
            Decomposition strategy
        """
        strategy = self.strategies.get(task_type)

        if not strategy:
            # Fall back to code strategy as default
            self.logger.warning(
                f"No strategy found for {task_type}, using default code strategy"
            )
            strategy = self.strategies[TaskType.CODE_GENERATION]

        return strategy

    def _infer_task_type(self, description: str) -> TaskType:
        """
        Infer task type from description.

        PATTERN: Keyword matching with priority order
        GOTCHA: May not always be accurate

        Args:
            description: Task description

        Returns:
            Inferred TaskType
        """
        description_lower = description.lower()

        # Check in priority order
        if any(kw in description_lower for kw in ["test", "testing", "unit test"]):
            return TaskType.TESTING

        if any(
            kw in description_lower for kw in ["refactor", "refactoring", "restructure"]
        ):
            return TaskType.REFACTORING

        if any(kw in description_lower for kw in ["debug", "fix bug", "troubleshoot"]):
            return TaskType.DEBUGGING

        if any(
            kw in description_lower for kw in ["document", "documentation", "readme"]
        ):
            return TaskType.DOCUMENTATION

        if any(
            kw in description_lower for kw in ["research", "investigate", "analyze"]
        ):
            return TaskType.RESEARCH

        if any(kw in description_lower for kw in ["architecture", "design", "plan"]):
            return TaskType.ARCHITECTURE

        if any(kw in description_lower for kw in ["create", "implement", "add"]):
            return TaskType.CODE_GENERATION

        if any(kw in description_lower for kw in ["modify", "update", "change"]):
            return TaskType.CODE_MODIFICATION

        # Default to code generation
        return TaskType.CODE_GENERATION

    def _build_tree(self, root_task: Task, all_tasks: List[Task]) -> DecompositionTree:
        """
        Build decomposition tree from list of tasks.

        PATTERN: Organize tasks by ID, identify leaf tasks, calculate stats
        CRITICAL: Ensure tree structure is consistent

        Args:
            root_task: Root task
            all_tasks: All tasks in decomposition

        Returns:
            DecompositionTree
        """
        # Create task dictionary
        tasks_dict = {task.id: task for task in all_tasks}

        # Identify leaf tasks
        leaf_task_ids = [task.id for task in all_tasks if task.is_leaf]

        # Calculate max depth
        max_depth = max((task.depth for task in all_tasks), default=0)

        # Create tree
        tree = DecompositionTree(
            root_task_id=root_task.id,
            tasks=tasks_dict,
            total_tasks=len(all_tasks),
            max_depth=max_depth,
            leaf_tasks=leaf_task_ids,
        )

        return tree

    def _build_dependencies(self, tree: DecompositionTree) -> None:
        """
        Build dependency graph from task relationships.

        PATTERN: Sequential dependencies between siblings
        GOTCHA: Don't create unnecessary dependencies

        Args:
            tree: Decomposition tree
        """
        # Clear existing graph
        self.dependency_manager.build_from_tree(tree)

        # For now, create simple sequential dependencies between siblings
        # Group tasks by parent
        children_by_parent: Dict[Optional[str], List[str]] = {}

        for task_id, task in tree.tasks.items():
            parent_id = task.parent_id
            if parent_id not in children_by_parent:
                children_by_parent[parent_id] = []
            children_by_parent[parent_id].append(task_id)

        # Create sequential dependencies for each parent's children
        for parent_id, child_ids in children_by_parent.items():
            if len(child_ids) <= 1:
                continue

            # Sort by depth and creation order (implicitly by ID)
            child_ids_sorted = sorted(
                child_ids, key=lambda tid: (tree.tasks[tid].depth, tid)
            )

            # Create sequential dependencies
            for i in range(1, len(child_ids_sorted)):
                prev_task_id = child_ids_sorted[i - 1]
                curr_task_id = child_ids_sorted[i]

                # Add dependency to tasks
                tree.tasks[curr_task_id].dependencies.add(prev_task_id)
                tree.tasks[prev_task_id].dependents.add(curr_task_id)

                # Add to dependency manager
                self.dependency_manager.add_dependency(curr_task_id, prev_task_id)

    def _estimate_duration(self, tree: DecompositionTree) -> int:
        """
        Estimate total execution duration in milliseconds.

        PATTERN: Sum of sequential path, parallel execution considered
        GOTCHA: Rough estimate based on complexity

        Args:
            tree: Decomposition tree

        Returns:
            Estimated duration in milliseconds
        """
        # Simple estimation: assume each leaf task takes time based on complexity
        # Complex task: ~30 seconds, Simple task: ~5 seconds
        total_ms = 0

        for task_id in tree.leaf_tasks:
            task = tree.tasks[task_id]

            # Base time: 5-30 seconds based on complexity
            base_time_ms = 5000 + int(task.estimated_complexity * 25000)

            total_ms += base_time_ms

        # Account for parallel execution (rough estimate: 40% reduction)
        parallel_factor = 0.6
        estimated_ms = int(total_ms * parallel_factor)

        return estimated_ms
