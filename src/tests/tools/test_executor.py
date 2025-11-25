"""Tests for tool executor."""

import pytest
import asyncio
from datetime import datetime

from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry
from src.tools.base import BaseTool
from src.models.tool_models import (
    ToolCategory,
    ToolSchema,
    ToolParameter,
    ToolResult,
    ToolRequest,
    ToolStatus,
)


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, name: str = "mock_tool", should_fail: bool = False):
        self.should_fail = should_fail
        super().__init__(name, ToolCategory.TESTING)

    def _build_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description="Mock tool for testing",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="value",
                    type="string",
                    description="Test value",
                    required=True,
                ),
            ],
            returns="Test result",
            timeout_seconds=5,
            max_retries=2,
        )

    async def execute(self, value: str, **kwargs) -> ToolResult:
        """Execute mock tool."""
        if self.should_fail:
            raise Exception("Mock tool failure")

        return self._create_success_result(
            result={"value": value, "success": True},
            execution_time_ms=10,
        )


@pytest.fixture
def registry():
    """Create test registry."""
    reg = ToolRegistry()
    reg.clear()  # Clear any existing tools
    return reg


@pytest.fixture
def executor(registry):
    """Create test executor."""
    return ToolExecutor(registry=registry)


@pytest.mark.asyncio
async def test_executor_success(executor, registry):
    """Test successful tool execution."""
    # Register mock tool
    tool = MockTool()
    registry.register_tool(tool)

    # Create request
    request = ToolRequest(
        tool_name="mock_tool",
        parameters={"value": "test"},
        agent_id="test_agent",
        workflow_id="test_workflow",
    )

    # Execute
    result = await executor.execute_tool(request)

    # Verify
    assert result.status == ToolStatus.SUCCESS
    assert result.result["value"] == "test"
    assert result.result["success"] is True


@pytest.mark.asyncio
async def test_executor_failure(executor, registry):
    """Test tool execution failure."""
    # Register failing mock tool
    tool = MockTool(should_fail=True)
    registry.register_tool(tool)

    # Create request
    request = ToolRequest(
        tool_name="mock_tool",
        parameters={"value": "test"},
        agent_id="test_agent",
        workflow_id="test_workflow",
    )

    # Execute
    result = await executor.execute_tool(request)

    # Verify
    assert result.status == ToolStatus.FAILED
    assert result.error is not None


@pytest.mark.asyncio
async def test_executor_tool_not_found(executor):
    """Test execution with non-existent tool."""
    # Create request for non-existent tool
    request = ToolRequest(
        tool_name="nonexistent_tool",
        parameters={},
        agent_id="test_agent",
        workflow_id="test_workflow",
    )

    # Execute - should catch exception and return failed result
    result = await executor.execute_tool(request)

    # Verify - executor returns failed result
    assert result.status == ToolStatus.FAILED
    assert result.tool_name == "nonexistent_tool"
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_executor_parameter_validation(executor, registry):
    """Test parameter validation."""
    # Register mock tool
    tool = MockTool()
    registry.register_tool(tool)

    # Create request with missing parameter
    request = ToolRequest(
        tool_name="mock_tool",
        parameters={},  # Missing required 'value' parameter
        agent_id="test_agent",
        workflow_id="test_workflow",
    )

    # Execute should fail validation
    with pytest.raises(ValueError):
        await executor.validate_request(request)
