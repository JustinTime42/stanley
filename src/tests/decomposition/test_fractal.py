"""Tests for fractal decomposition engine."""

import pytest
from src.decomposition.fractal_decomposer import FractalDecomposer
from src.models.decomposition_models import (
    DecompositionRequest,
    TaskType,
)


@pytest.mark.asyncio
async def test_simple_decomposition():
    """Test basic decomposition of a simple task."""
    decomposer = FractalDecomposer(
        max_depth=3,
        complexity_threshold=0.2,
        max_subtasks_per_level=5,
    )

    request = DecompositionRequest(
        task_description="Create a simple REST API with authentication",
        task_type=TaskType.CODE_GENERATION,
        max_depth=3,
        include_dependencies=True,
        estimate_costs=False,
        target_model_routing=False,
    )

    result = await decomposer.decompose(request)

    # Verify tree structure
    assert result.tree is not None
    assert result.tree.root_task_id is not None
    assert result.tree.total_tasks > 1
    assert len(result.tree.leaf_tasks) > 0

    # Verify execution plan
    assert len(result.execution_plan) > 0


@pytest.mark.asyncio
async def test_depth_limiting():
    """Test that decomposition respects max depth."""
    decomposer = FractalDecomposer(
        max_depth=2,
        complexity_threshold=0.1,  # Low threshold to encourage decomposition
        max_subtasks_per_level=5,
    )

    request = DecompositionRequest(
        task_description="Build a complex distributed system",
        task_type=TaskType.ARCHITECTURE,
        max_depth=2,
        include_dependencies=False,
        estimate_costs=False,
        target_model_routing=False,
    )

    result = await decomposer.decompose(request)

    # Verify max depth is respected
    max_task_depth = max(task.depth for task in result.tree.tasks.values())
    assert max_task_depth <= 2


@pytest.mark.asyncio
async def test_complexity_threshold():
    """Test that tasks below complexity threshold are not decomposed."""
    decomposer = FractalDecomposer(
        max_depth=5,
        complexity_threshold=0.8,  # High threshold to prevent decomposition
        max_subtasks_per_level=5,
    )

    request = DecompositionRequest(
        task_description="Fix a simple typo",
        task_type=TaskType.CODE_MODIFICATION,
        max_depth=5,
        include_dependencies=False,
        estimate_costs=False,
        target_model_routing=False,
    )

    result = await decomposer.decompose(request)

    # Should have minimal decomposition due to high threshold
    # Root task might be the only task or very few
    assert result.tree.total_tasks <= 3
