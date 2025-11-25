"""Dependency graph management with DAG validation and topological sorting."""

import logging
from graphlib import TopologicalSorter, CycleError
from typing import Dict, List, Set
from ..models.decomposition_models import (
    DecompositionTree,
    DependencyValidation,
)


logger = logging.getLogger(__name__)


class DependencyManager:
    """
    Manages task dependencies with DAG validation and topological sorting.

    PATTERN: Use Python's built-in graphlib for efficient topological sorting
    CRITICAL: Must detect circular dependencies before execution
    GOTCHA: Tasks with no dependencies can run in parallel
    """

    def __init__(self):
        """Initialize dependency manager."""
        self.graph: Dict[str, Set[str]] = {}  # task_id -> set of dependencies
        self.reverse_graph: Dict[str, Set[str]] = {}  # task_id -> set of dependents
        self.logger = logging.getLogger(__name__)

    def add_task(self, task_id: str) -> None:
        """
        Add a task to the graph.

        Args:
            task_id: Task ID to add
        """
        if task_id not in self.graph:
            self.graph[task_id] = set()
            self.reverse_graph[task_id] = set()

    def add_dependency(self, task_id: str, depends_on: str) -> None:
        """
        Add a dependency relationship.

        PATTERN: task_id depends on depends_on
        CRITICAL: Both tasks must be added to graph first

        Args:
            task_id: Task that has the dependency
            depends_on: Task that must complete first
        """
        # Ensure both tasks exist in graph
        self.add_task(task_id)
        self.add_task(depends_on)

        # Add dependency
        self.graph[task_id].add(depends_on)
        self.reverse_graph[depends_on].add(task_id)

        self.logger.debug(f"Added dependency: {task_id} depends on {depends_on}")

    def remove_dependency(self, task_id: str, depends_on: str) -> None:
        """
        Remove a dependency relationship.

        Args:
            task_id: Task ID
            depends_on: Dependency to remove
        """
        if task_id in self.graph:
            self.graph[task_id].discard(depends_on)

        if depends_on in self.reverse_graph:
            self.reverse_graph[depends_on].discard(task_id)

    def get_dependencies(self, task_id: str) -> Set[str]:
        """
        Get all direct dependencies for a task.

        Args:
            task_id: Task ID

        Returns:
            Set of task IDs this task depends on
        """
        return self.graph.get(task_id, set()).copy()

    def get_dependents(self, task_id: str) -> Set[str]:
        """
        Get all tasks that depend on this task.

        Args:
            task_id: Task ID

        Returns:
            Set of task IDs that depend on this task
        """
        return self.reverse_graph.get(task_id, set()).copy()

    def validate_dependencies(self) -> DependencyValidation:
        """
        Validate dependency graph and detect cycles.

        PATTERN: Use TopologicalSorter for efficient cycle detection
        CRITICAL: Must detect all circular dependencies

        Returns:
            DependencyValidation with results
        """
        try:
            # Create topological sorter
            sorter = TopologicalSorter(self.graph)

            # Get execution order (validates DAG)
            execution_order = list(sorter.static_order())

            self.logger.info(
                f"Dependency validation successful: {len(execution_order)} tasks"
            )

            return DependencyValidation(
                is_valid=True,
                has_cycles=False,
                cycles=[],
                missing_dependencies=[],
                execution_order=execution_order,
            )

        except CycleError as e:
            # Extract cycle information from error
            self.logger.error(f"Circular dependencies detected: {e}")

            # Find all cycles
            cycles = self._find_all_cycles()

            return DependencyValidation(
                is_valid=False,
                has_cycles=True,
                cycles=cycles,
                missing_dependencies=[],
                execution_order=[],
            )

    def _find_all_cycles(self) -> List[List[str]]:
        """
        Find all circular dependency chains using DFS.

        PATTERN: Depth-first search with cycle detection
        GOTCHA: May not find all cycles in complex graphs

        Returns:
            List of cycle chains (each cycle is a list of task IDs)
        """
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> None:
            """DFS helper to detect cycles."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            # Visit all dependencies
            for neighbor in self.graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            rec_stack.remove(node)

        # Run DFS from each node
        for node in self.graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def get_execution_order(self) -> List[str]:
        """
        Get topologically sorted execution order.

        PATTERN: Tasks with no dependencies come first
        CRITICAL: Must validate before calling this

        Returns:
            List of task IDs in execution order

        Raises:
            CycleError: If circular dependencies exist
        """
        sorter = TopologicalSorter(self.graph)
        return list(sorter.static_order())

    def get_execution_batches(self) -> List[List[str]]:
        """
        Get batches of tasks that can execute in parallel.

        PATTERN: Tasks in same batch have no dependencies on each other
        CRITICAL: Each batch can run concurrently

        Returns:
            List of batches, where each batch is a list of task IDs

        Raises:
            CycleError: If circular dependencies exist
        """
        sorter = TopologicalSorter(self.graph)
        sorter.prepare()

        batches = []
        while sorter.is_active():
            # Get ready tasks (no pending dependencies)
            ready = list(sorter.get_ready())

            if ready:
                batches.append(ready)

                # Mark tasks as done
                for task_id in ready:
                    sorter.done(task_id)

        self.logger.info(
            f"Generated {len(batches)} execution batches "
            f"with {sum(len(b) for b in batches)} tasks"
        )

        return batches

    def build_from_tree(self, tree: DecompositionTree) -> None:
        """
        Build dependency graph from decomposition tree.

        PATTERN: Extract dependencies from all tasks
        CRITICAL: Validate all referenced tasks exist

        Args:
            tree: Decomposition tree
        """
        # Clear existing graph
        self.graph.clear()
        self.reverse_graph.clear()

        # Add all tasks
        for task_id in tree.tasks:
            self.add_task(task_id)

        # Add all dependencies
        for task_id, task in tree.tasks.items():
            for dependency_id in task.dependencies:
                if dependency_id not in tree.tasks:
                    self.logger.warning(
                        f"Task {task_id} depends on non-existent task {dependency_id}"
                    )
                    continue

                self.add_dependency(task_id, dependency_id)

        self.logger.info(
            f"Built dependency graph from tree: "
            f"{len(self.graph)} tasks, "
            f"{sum(len(deps) for deps in self.graph.values())} dependencies"
        )

    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[str]:
        """
        Get tasks that are ready to execute (all dependencies completed).

        PATTERN: Check if all dependencies are in completed set
        CRITICAL: Only return tasks not already completed

        Args:
            completed_tasks: Set of completed task IDs

        Returns:
            List of task IDs ready for execution
        """
        ready = []

        for task_id, dependencies in self.graph.items():
            # Skip if already completed
            if task_id in completed_tasks:
                continue

            # Check if all dependencies completed
            if dependencies.issubset(completed_tasks):
                ready.append(task_id)

        return ready

    def get_blocked_tasks(self, completed_tasks: Set[str]) -> List[str]:
        """
        Get tasks that are blocked by incomplete dependencies.

        Args:
            completed_tasks: Set of completed task IDs

        Returns:
            List of blocked task IDs
        """
        blocked = []

        for task_id, dependencies in self.graph.items():
            # Skip if already completed
            if task_id in completed_tasks:
                continue

            # Check if any dependencies incomplete
            if not dependencies.issubset(completed_tasks):
                blocked.append(task_id)

        return blocked

    def visualize(self) -> str:
        """
        Generate a text visualization of the dependency graph.

        PATTERN: Simple ASCII visualization for debugging
        GOTCHA: May be large for complex graphs

        Returns:
            String visualization
        """
        lines = ["Dependency Graph:"]
        lines.append("=" * 50)

        for task_id, dependencies in sorted(self.graph.items()):
            if dependencies:
                deps_str = ", ".join(sorted(dependencies))
                lines.append(f"{task_id} depends on: {deps_str}")
            else:
                lines.append(f"{task_id} (no dependencies)")

        return "\n".join(lines)
