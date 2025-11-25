"""Tests for dependency management."""

import pytest
from src.decomposition.dependency_manager import DependencyManager
from src.models.decomposition_models import (
    Task,
    TaskType,
    DecompositionTree,
)


def test_add_dependency():
    """Test adding dependencies."""
    manager = DependencyManager()

    manager.add_task("task1")
    manager.add_task("task2")
    manager.add_dependency("task2", "task1")  # task2 depends on task1

    deps = manager.get_dependencies("task2")
    assert "task1" in deps


def test_cycle_detection():
    """Test circular dependency detection."""
    manager = DependencyManager()

    # Create circular dependency: A -> B -> C -> A
    manager.add_task("A")
    manager.add_task("B")
    manager.add_task("C")

    manager.add_dependency("B", "A")
    manager.add_dependency("C", "B")
    manager.add_dependency("A", "C")  # Creates cycle

    validation = manager.validate_dependencies()

    assert not validation.is_valid
    assert validation.has_cycles
    assert len(validation.cycles) > 0


def test_topological_sort():
    """Test topological sorting of dependencies."""
    manager = DependencyManager()

    # Create DAG: A -> B -> D, A -> C -> D
    manager.add_task("A")
    manager.add_task("B")
    manager.add_task("C")
    manager.add_task("D")

    manager.add_dependency("B", "A")
    manager.add_dependency("C", "A")
    manager.add_dependency("D", "B")
    manager.add_dependency("D", "C")

    validation = manager.validate_dependencies()

    assert validation.is_valid
    assert not validation.has_cycles
    assert len(validation.execution_order) == 4

    # A should come before B and C
    order = validation.execution_order
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")

    # B and C should come before D
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")


def test_execution_batches():
    """Test parallel execution batching."""
    manager = DependencyManager()

    # Create DAG with parallelizable tasks
    manager.add_task("A")
    manager.add_task("B")
    manager.add_task("C")
    manager.add_task("D")
    manager.add_task("E")

    # B and C can run in parallel after A
    manager.add_dependency("B", "A")
    manager.add_dependency("C", "A")

    # D and E can run in parallel after B and C
    manager.add_dependency("D", "B")
    manager.add_dependency("D", "C")
    manager.add_dependency("E", "B")
    manager.add_dependency("E", "C")

    batches = manager.get_execution_batches()

    # Should have 3 batches: [A], [B, C], [D, E]
    assert len(batches) == 3
    assert "A" in batches[0]
    assert set(batches[1]) == {"B", "C"}
    assert set(batches[2]) == {"D", "E"}


def test_ready_tasks():
    """Test getting ready tasks based on completion."""
    manager = DependencyManager()

    manager.add_task("A")
    manager.add_task("B")
    manager.add_task("C")

    manager.add_dependency("B", "A")
    manager.add_dependency("C", "B")

    # Initially only A is ready
    completed = set()
    ready = manager.get_ready_tasks(completed)
    assert "A" in ready
    assert "B" not in ready
    assert "C" not in ready

    # After A completes, B is ready
    completed.add("A")
    ready = manager.get_ready_tasks(completed)
    assert "B" in ready
    assert "C" not in ready

    # After B completes, C is ready
    completed.add("B")
    ready = manager.get_ready_tasks(completed)
    assert "C" in ready
