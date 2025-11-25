"""Tool executor with timeout, retry, and resource monitoring."""

import asyncio
import logging
from typing import Optional
from datetime import datetime

from .registry import ToolRegistry
from .retry import RetryManager
from .monitor import ResourceMonitor
from ..models.tool_models import (
    ToolRequest,
    ToolResult,
    ToolStatus,
)

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Tool executor with timeout, retry, and monitoring.

    PATTERN: Async executor with timeout and retry
    CRITICAL: Uses registry for tool lookup
    GOTCHA: Timeout must be handled with asyncio.wait_for
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        retry_manager: Optional[RetryManager] = None,
        enable_monitoring: bool = True,
    ):
        """
        Initialize tool executor.

        Args:
            registry: Tool registry (creates default if None)
            retry_manager: Retry manager (creates default if None)
            enable_monitoring: Enable resource monitoring
        """
        self.registry = registry or ToolRegistry()
        self.retry_manager = retry_manager or RetryManager()
        self.enable_monitoring = enable_monitoring
        self.logger = logging.getLogger(__name__)

    async def execute_tool(
        self,
        request: ToolRequest,
    ) -> ToolResult:
        """
        Execute a tool with timeout, retry, and monitoring.

        PATTERN: Main execution flow with all features
        CRITICAL: Handles timeout, retry, and resource tracking

        Args:
            request: Tool execution request

        Returns:
            ToolResult

        Raises:
            ValueError: If tool not found or parameters invalid
        """
        start_time = datetime.now()

        # Get tool first
        tool = self.registry.get_tool(request.tool_name)
        if not tool:
            return ToolResult(
                tool_name=request.tool_name,
                status=ToolStatus.FAILED,
                error=f"Tool '{request.tool_name}' not found in registry",
                execution_time_ms=0,
            )

        # Determine timeout
        timeout = request.timeout_override or tool.schema.timeout_seconds

        # Execute with resource monitoring
        try:
            if self.enable_monitoring and ResourceMonitor.is_available():
                async with ResourceMonitor(tool.name) as monitor:
                    result = await self._execute_with_timeout_and_retry(
                        tool=tool,
                        request=request,
                        timeout=timeout,
                    )
                    # Add resource usage to result
                    result.resource_usage = monitor.get_usage_delta()
            else:
                result = await self._execute_with_timeout_and_retry(
                    tool=tool,
                    request=request,
                    timeout=timeout,
                )

            # Calculate total execution time
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            result.execution_time_ms = execution_time_ms

            self.logger.info(
                f"Tool '{request.tool_name}' executed: "
                f"status={result.status.value}, "
                f"time={execution_time_ms}ms"
            )

            return result

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            self.logger.error(f"Tool '{request.tool_name}' execution failed: {e}")

            return ToolResult(
                tool_name=request.tool_name,
                status=ToolStatus.FAILED,
                error=str(e),
                execution_time_ms=execution_time_ms,
            )

    async def _execute_with_timeout_and_retry(
        self,
        tool,
        request: ToolRequest,
        timeout: int,
    ) -> ToolResult:
        """
        Execute tool with timeout and retry.

        Args:
            tool: Tool instance
            request: Tool request
            timeout: Timeout in seconds

        Returns:
            ToolResult
        """
        # Execute with retry
        retry_count = 0

        async def execute_attempt():
            nonlocal retry_count
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    tool.execute(**request.parameters),
                    timeout=float(timeout),
                )
                result.retry_count = retry_count
                return result

            except asyncio.TimeoutError:
                self.logger.warning(f"Tool '{tool.name}' timed out after {timeout}s")
                return tool._create_timeout_result(execution_time_ms=timeout * 1000)

            except Exception as e:
                retry_count += 1
                self.logger.error(
                    f"Tool '{tool.name}' execution error (attempt {retry_count}): {e}"
                )
                raise

        # Use retry manager
        max_retries = tool.schema.max_retries
        if max_retries > 1:
            try:
                result = await self.retry_manager.execute_with_retry(
                    execute_attempt,
                    retry_on=(Exception,),
                )
            except Exception as e:
                # All retries failed
                return tool._create_error_result(
                    error=str(e),
                    execution_time_ms=0,
                    retry_count=retry_count,
                )
        else:
            result = await execute_attempt()

        return result

    async def _validate_request(self, request: ToolRequest) -> None:
        """
        Validate tool request.

        Args:
            request: Tool request

        Raises:
            ValueError: If request is invalid
        """
        # Check tool exists
        tool = self.registry.get_tool(request.tool_name)
        if not tool:
            raise ValueError(
                f"Tool '{request.tool_name}' not found. "
                f"Available tools: {self.registry.list_tool_names()}"
            )

        # Validate parameters
        await tool.validate_parameters(request.parameters)

    async def validate_request(self, request: ToolRequest) -> bool:
        """
        Public method to validate a request without executing.

        Args:
            request: Tool request

        Returns:
            True if valid

        Raises:
            ValueError: If request is invalid
        """
        await self._validate_request(request)
        return True
