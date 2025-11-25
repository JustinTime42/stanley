"""Base tool class for all agent tools."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
from datetime import datetime

from ..models.tool_models import (
    ToolCategory,
    ToolSchema,
    ToolResult,
    ToolStatus,
)

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    PATTERN: Abstract base for all tools
    CRITICAL: Must be async and JSON-serializable
    GOTCHA: All execute methods must be async for agent integration
    """

    def __init__(self, name: str, category: ToolCategory):
        """
        Initialize base tool.

        Args:
            name: Tool name (must be unique)
            category: Tool category for organization
        """
        self.name = name
        self.category = category
        self.schema = self._build_schema()
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute tool with given parameters.

        CRITICAL: Must be async and return ToolResult
        CRITICAL: Results must be JSON-serializable

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with execution outcome
        """
        pass

    @abstractmethod
    def _build_schema(self) -> ToolSchema:
        """
        Build tool schema for discovery and validation.

        Returns:
            ToolSchema describing this tool
        """
        pass

    async def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Validate parameters against schema.

        Args:
            parameters: Parameters to validate

        Raises:
            ValueError: If parameters are invalid
        """
        schema = self.schema

        # Check required parameters
        for param in schema.parameters:
            if param.required and param.name not in parameters:
                raise ValueError(
                    f"Missing required parameter '{param.name}' for tool '{self.name}'"
                )

        # Check parameter types (basic validation)
        for param_name, param_value in parameters.items():
            # Find parameter in schema
            param_schema = next(
                (p for p in schema.parameters if p.name == param_name), None
            )

            if not param_schema:
                self.logger.warning(
                    f"Unknown parameter '{param_name}' for tool '{self.name}'"
                )
                continue

            # Validate enum values
            if param_schema.enum and param_value not in param_schema.enum:
                raise ValueError(
                    f"Parameter '{param_name}' must be one of {param_schema.enum}, "
                    f"got '{param_value}'"
                )

    def get_schema(self) -> ToolSchema:
        """
        Get tool schema.

        Returns:
            Tool schema
        """
        return self.schema

    def _create_success_result(
        self,
        result: Any,
        execution_time_ms: int,
        resource_usage: Dict[str, float] = None,
    ) -> ToolResult:
        """
        Create a success result.

        Args:
            result: Tool output (must be JSON-serializable)
            execution_time_ms: Execution time in milliseconds
            resource_usage: Optional resource usage metrics

        Returns:
            ToolResult
        """
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            result=result,
            execution_time_ms=execution_time_ms,
            resource_usage=resource_usage or {},
            timestamp=datetime.now(),
        )

    def _create_error_result(
        self,
        error: str,
        execution_time_ms: int,
        retry_count: int = 0,
    ) -> ToolResult:
        """
        Create an error result.

        Args:
            error: Error message
            execution_time_ms: Execution time in milliseconds
            retry_count: Number of retries attempted

        Returns:
            ToolResult
        """
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.FAILED,
            error=error,
            execution_time_ms=execution_time_ms,
            retry_count=retry_count,
            timestamp=datetime.now(),
        )

    def _create_timeout_result(
        self,
        execution_time_ms: int,
    ) -> ToolResult:
        """
        Create a timeout result.

        Args:
            execution_time_ms: Execution time in milliseconds

        Returns:
            ToolResult
        """
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.TIMEOUT,
            error=f"Tool '{self.name}' timed out after {execution_time_ms}ms",
            execution_time_ms=execution_time_ms,
            timestamp=datetime.now(),
        )
