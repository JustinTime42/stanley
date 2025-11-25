"""High-level tool orchestration service."""

import logging
from typing import List, Optional, Dict, Any

from ..tools import (
    BaseTool,
    ToolRegistry,
    ToolExecutor,
    RetryManager,
    ResourceMonitor,
    ParallelExecutor,
)
from ..tools.implementations import (
    # File tools
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    CreateDirectoryTool,
    DeleteFileTool,
    # Code tools
    GenerateCodeTool,
    RefactorCodeTool,
    AddTestsTool,
    # Git tools
    GitStatusTool,
    GitCommitTool,
    GitDiffTool,
    # Test tools
    PytestTool,
    UnittestTool,
    # Validation tools
    RuffTool,
    MypyTool,
    BlackTool,
)
from ..models.tool_models import (
    ToolRequest,
    ToolResult,
    ToolSchema,
    ToolCategory,
)
from ..models.execution_models import ParallelExecutionPlan

logger = logging.getLogger(__name__)


class ToolOrchestrator:
    """
    High-level tool orchestration service.

    PATTERN: Facade pattern like LLMOrchestrator
    CRITICAL: Single entry point for all tool operations
    GOTCHA: Initialize all tools and register them
    """

    def __init__(
        self,
        llm_service: Optional[Any] = None,
        max_retries: int = 3,
        enable_monitoring: bool = True,
        max_parallelism: int = 5,
    ):
        """
        Initialize tool orchestrator.

        Args:
            llm_service: Optional LLM service for code generation tools
            max_retries: Maximum retry attempts
            enable_monitoring: Enable resource monitoring
            max_parallelism: Maximum parallel executions
        """
        self.llm_service = llm_service
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.registry = ToolRegistry()
        self.retry_manager = RetryManager(max_retries=max_retries)
        self.executor = ToolExecutor(
            registry=self.registry,
            retry_manager=self.retry_manager,
            enable_monitoring=enable_monitoring,
        )
        self.parallel_executor = ParallelExecutor(
            executor=self.executor,
            max_parallelism=max_parallelism,
        )

        # Register all tools
        self._register_all_tools()

        self.logger.info(
            f"Tool Orchestrator initialized with {self.registry.get_tool_count()} tools"
        )

    def _register_all_tools(self) -> None:
        """
        Register all available tools.

        PATTERN: Register all tool implementations
        CRITICAL: Each tool must have unique name
        """
        tools_to_register: List[BaseTool] = [
            # File system tools
            ReadFileTool(),
            WriteFileTool(),
            ListDirectoryTool(),
            CreateDirectoryTool(),
            DeleteFileTool(),
            # Code generation tools
            GenerateCodeTool(llm_service=self.llm_service),
            RefactorCodeTool(llm_service=self.llm_service),
            AddTestsTool(llm_service=self.llm_service),
            # Git tools
            GitStatusTool(),
            GitCommitTool(),
            GitDiffTool(),
            # Test tools
            PytestTool(),
            UnittestTool(),
            # Validation tools
            RuffTool(),
            MypyTool(),
            BlackTool(),
        ]

        for tool in tools_to_register:
            try:
                self.registry.register_tool(tool)
            except ValueError as e:
                # Tool already registered (singleton registry), skip silently
                if "already registered" in str(e):
                    self.logger.debug(f"Tool {tool.name} already registered, skipping")
                else:
                    self.logger.error(f"Failed to register tool {tool.name}: {e}")
            except Exception as e:
                self.logger.error(f"Failed to register tool {tool.name}: {e}")

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        agent_id: str = "unknown",
        workflow_id: str = "unknown",
        timeout_override: Optional[int] = None,
    ) -> ToolResult:
        """
        Execute a single tool.

        PATTERN: Main entry point for tool execution
        CRITICAL: Use executor for monitoring and retry

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            agent_id: Agent making the request
            workflow_id: Workflow context
            timeout_override: Optional timeout override

        Returns:
            ToolResult
        """
        request = ToolRequest(
            tool_name=tool_name,
            parameters=parameters,
            agent_id=agent_id,
            workflow_id=workflow_id,
            timeout_override=timeout_override,
        )

        return await self.executor.execute_tool(request)

    async def execute_batch(
        self,
        requests: List[ToolRequest],
        parallel: bool = True,
        max_parallelism: Optional[int] = None,
    ) -> List[ToolResult]:
        """
        Execute multiple tools.

        PATTERN: Batch execution with optional parallelism
        CRITICAL: Use parallel executor for concurrent execution

        Args:
            requests: List of tool requests
            parallel: Execute in parallel
            max_parallelism: Override default parallelism

        Returns:
            List of tool results
        """
        if not requests:
            return []

        if parallel:
            return await self.parallel_executor.execute_parallel(
                requests=requests,
                max_parallelism=max_parallelism,
            )
        else:
            # Execute sequentially
            results = []
            for request in requests:
                result = await self.executor.execute_tool(request)
                results.append(result)
            return results

    async def execute_with_dependencies(
        self,
        plan: ParallelExecutionPlan,
    ) -> List[ToolResult]:
        """
        Execute tools with dependency resolution.

        Args:
            plan: Execution plan with dependencies

        Returns:
            List of all tool results
        """
        return await self.parallel_executor.execute_with_dependencies(plan)

    def build_execution_plan(
        self,
        requests: List[ToolRequest],
        dependencies: Optional[Dict[str, List[str]]] = None,
        max_parallelism: Optional[int] = None,
    ) -> ParallelExecutionPlan:
        """
        Build execution plan from requests and dependencies.

        Args:
            requests: List of tool requests
            dependencies: Tool dependencies
            max_parallelism: Maximum parallel executions

        Returns:
            ParallelExecutionPlan
        """
        return self.parallel_executor.build_execution_plan(
            requests=requests,
            dependencies=dependencies,
            max_parallelism=max_parallelism,
        )

    def get_available_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        """
        Get list of available tool names.

        Args:
            category: Optional category filter

        Returns:
            List of tool names
        """
        return self.registry.list_tool_names(category)

    def get_tool_schemas(
        self, category: Optional[ToolCategory] = None
    ) -> List[ToolSchema]:
        """
        Get schemas for all tools.

        Args:
            category: Optional category filter

        Returns:
            List of tool schemas
        """
        return self.registry.get_tool_schemas(category)

    def get_tool_schema(self, tool_name: str) -> Optional[ToolSchema]:
        """
        Get schema for a specific tool.

        Args:
            tool_name: Tool name

        Returns:
            ToolSchema or None if not found
        """
        tool = self.registry.get_tool(tool_name)
        return tool.get_schema() if tool else None

    def is_monitoring_available(self) -> bool:
        """
        Check if resource monitoring is available.

        Returns:
            True if monitoring available
        """
        return ResourceMonitor.is_available()

    async def validate_request(self, request: ToolRequest) -> bool:
        """
        Validate a tool request.

        Args:
            request: Tool request

        Returns:
            True if valid

        Raises:
            ValueError: If request is invalid
        """
        return await self.executor.validate_request(request)

    async def cleanup(self) -> None:
        """Clean up resources."""
        # Clear registry if needed
        self.logger.info("Tool Orchestrator cleaned up")

    async def close(self) -> None:
        """Alias for cleanup() to match common close() pattern."""
        await self.cleanup()
