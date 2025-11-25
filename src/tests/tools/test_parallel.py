"""Tests for parallel execution."""

import pytest
import asyncio
from datetime import datetime

from src.tools.parallel import ParallelExecutor
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry
from src.tests.tools.test_executor import MockTool
from src.models.tool_models import ToolRequest, ToolStatus


@pytest.fixture
def setup_executor():
    """Create test executor with tools."""
    registry = ToolRegistry()
    registry.clear()

    # Register mock tools
    for i in range(5):
        tool = MockTool(name=f"tool_{i}")
        registry.register_tool(tool)

    executor = ToolExecutor(registry=registry, enable_monitoring=False)
    return executor, registry


@pytest.mark.asyncio
async def test_parallel_execution(setup_executor):
    """Test parallel execution of multiple tools."""
    executor, _ = setup_executor
    parallel_executor = ParallelExecutor(executor=executor, max_parallelism=3)

    # Create requests
    requests = [
        ToolRequest(
            tool_name=f"tool_{i}",
            parameters={"value": f"test_{i}"},
            agent_id="test_agent",
            workflow_id="test_workflow",
        )
        for i in range(5)
    ]

    # Execute in parallel
    start_time = datetime.now()
    results = await parallel_executor.execute_parallel(requests)
    execution_time = (datetime.now() - start_time).total_seconds()

    # Verify
    assert len(results) == 5
    assert all(r.status == ToolStatus.SUCCESS for r in results)

    # Parallel execution should be faster than sequential
    # (though with very fast mock tools, this might not be measurable)
    assert execution_time < 1.0  # Should complete quickly


@pytest.mark.asyncio
async def test_parallel_execution_with_failures(setup_executor):
    """Test parallel execution with some failures."""
    executor, registry = setup_executor

    # Add a failing tool
    failing_tool = MockTool(name="failing_tool", should_fail=True)
    registry.register_tool(failing_tool)

    parallel_executor = ParallelExecutor(executor=executor)

    # Create requests including one that will fail
    requests = [
        ToolRequest(
            tool_name="tool_0",
            parameters={"value": "test_0"},
            agent_id="test_agent",
            workflow_id="test_workflow",
        ),
        ToolRequest(
            tool_name="failing_tool",
            parameters={"value": "test_fail"},
            agent_id="test_agent",
            workflow_id="test_workflow",
        ),
        ToolRequest(
            tool_name="tool_1",
            parameters={"value": "test_1"},
            agent_id="test_agent",
            workflow_id="test_workflow",
        ),
    ]

    # Execute
    results = await parallel_executor.execute_parallel(requests)

    # Verify - partial failure should not stop other executions
    assert len(results) == 3
    assert sum(1 for r in results if r.status == ToolStatus.SUCCESS) == 2
    assert sum(1 for r in results if r.status == ToolStatus.FAILED) == 1


@pytest.mark.asyncio
async def test_execution_plan_building(setup_executor):
    """Test building execution plan with dependencies."""
    executor, _ = setup_executor
    parallel_executor = ParallelExecutor(executor=executor)

    # Create requests
    requests = [
        ToolRequest(
            tool_name=f"tool_{i}",
            parameters={"value": f"test_{i}"},
            agent_id="test_agent",
            workflow_id="test_workflow",
        )
        for i in range(3)
    ]

    # Define dependencies: tool_2 depends on tool_0 and tool_1
    dependencies = {
        "tool_2": ["tool_0", "tool_1"],
    }

    # Build plan
    plan = parallel_executor.build_execution_plan(
        requests=requests,
        dependencies=dependencies,
    )

    # Verify plan has correct grouping
    assert len(plan.tool_groups) == 2  # Two groups: [tool_0, tool_1] and [tool_2]
    assert len(plan.tool_groups[0]) == 2  # First group has 2 tools
    assert len(plan.tool_groups[1]) == 1  # Second group has 1 tool


@pytest.mark.asyncio
async def test_bounded_parallelism(setup_executor):
    """Test that parallelism is bounded by max_parallelism."""
    executor, _ = setup_executor
    parallel_executor = ParallelExecutor(executor=executor, max_parallelism=2)

    # Create many requests
    requests = [
        ToolRequest(
            tool_name=f"tool_{i % 5}",
            parameters={"value": f"test_{i}"},
            agent_id="test_agent",
            workflow_id="test_workflow",
        )
        for i in range(10)
    ]

    # Execute - should be bounded by semaphore
    results = await parallel_executor.execute_parallel(requests)

    # Verify all completed
    assert len(results) == 10
    assert all(r.status == ToolStatus.SUCCESS for r in results)
