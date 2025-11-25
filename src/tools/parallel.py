"""Parallel tool execution with dependency resolution."""

import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime

from .executor import ToolExecutor
from ..models.tool_models import ToolRequest, ToolResult, ToolStatus
from ..models.execution_models import ParallelExecutionPlan

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """
    Parallel executor for concurrent tool execution.

    PATTERN: asyncio.gather with error handling
    CRITICAL: Must handle partial failures
    GOTCHA: Dependency resolution required for correct ordering
    """

    def __init__(
        self,
        executor: Optional[ToolExecutor] = None,
        max_parallelism: int = 5,
    ):
        """
        Initialize parallel executor.

        Args:
            executor: Tool executor (creates default if None)
            max_parallelism: Maximum concurrent executions
        """
        self.executor = executor or ToolExecutor()
        self.max_parallelism = max_parallelism
        self.logger = logging.getLogger(__name__)

    async def execute_parallel(
        self,
        requests: List[ToolRequest],
        max_parallelism: Optional[int] = None,
    ) -> List[ToolResult]:
        """
        Execute multiple tools in parallel.

        PATTERN: Bounded parallel execution with error handling
        CRITICAL: Must handle partial failures

        Args:
            requests: List of tool requests
            max_parallelism: Override default max parallelism

        Returns:
            List of tool results (in same order as requests)
        """
        if not requests:
            return []

        max_parallel = max_parallelism or self.max_parallelism
        semaphore = asyncio.Semaphore(max_parallel)

        self.logger.info(
            f"Executing {len(requests)} tools in parallel "
            f"(max_parallelism={max_parallel})"
        )

        async def bounded_execute(request: ToolRequest) -> ToolResult:
            """Execute with semaphore to limit concurrency."""
            async with semaphore:
                try:
                    return await self.executor.execute_tool(request)
                except Exception as e:
                    self.logger.error(
                        f"Error executing tool '{request.tool_name}': {e}"
                    )
                    return ToolResult(
                        tool_name=request.tool_name,
                        status=ToolStatus.FAILED,
                        error=str(e),
                        execution_time_ms=0,
                    )

        # Execute all tools, collecting both successes and failures
        start_time = datetime.now()

        results = await asyncio.gather(
            *[bounded_execute(req) for req in requests],
            return_exceptions=False,  # Return results, not exceptions
        )

        total_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        successful = sum(1 for r in results if r.status == ToolStatus.SUCCESS)
        failed = sum(1 for r in results if r.status == ToolStatus.FAILED)

        self.logger.info(
            f"Parallel execution complete: {successful} succeeded, {failed} failed "
            f"in {total_time_ms}ms"
        )

        return results

    async def execute_with_dependencies(
        self,
        plan: ParallelExecutionPlan,
    ) -> List[ToolResult]:
        """
        Execute tools with dependency resolution.

        PATTERN: Topological sort and staged execution
        CRITICAL: Dependencies must be satisfied before execution

        Args:
            plan: Execution plan with dependencies

        Returns:
            List of all tool results
        """
        self.logger.info(
            f"Executing plan {plan.execution_id} with "
            f"{sum(len(group) for group in plan.tool_groups)} tools"
        )

        all_results: List[ToolResult] = []

        # Execute each group in order
        for i, group in enumerate(plan.tool_groups):
            self.logger.info(
                f"Executing group {i + 1}/{len(plan.tool_groups)} "
                f"with {len(group)} tools"
            )

            # Execute group in parallel
            group_results = await self.execute_parallel(
                requests=group,
                max_parallelism=plan.max_parallelism,
            )

            all_results.extend(group_results)

            # Check for failures in this group
            failed_count = sum(
                1 for r in group_results if r.status == ToolStatus.FAILED
            )

            if failed_count > 0:
                self.logger.warning(f"Group {i + 1} had {failed_count} failures")

        return all_results

    def build_execution_plan(
        self,
        requests: List[ToolRequest],
        dependencies: Optional[Dict[str, List[str]]] = None,
        max_parallelism: Optional[int] = None,
    ) -> ParallelExecutionPlan:
        """
        Build execution plan from requests and dependencies.

        PATTERN: Dependency graph to execution groups
        CRITICAL: Topological sort to resolve dependencies

        Args:
            requests: List of tool requests
            dependencies: Tool dependencies (tool_name -> [depends_on])
            max_parallelism: Maximum parallel executions

        Returns:
            ParallelExecutionPlan
        """
        deps = dependencies or {}
        max_parallel = max_parallelism or self.max_parallelism

        # Build dependency graph
        request_map = {req.tool_name: req for req in requests}

        # Perform topological sort to group tools
        groups = self._topological_sort(list(request_map.keys()), deps)

        # Convert groups to request groups
        tool_groups = [
            [request_map[tool_name] for tool_name in group] for group in groups
        ]

        # Estimate execution time (simplified)
        estimated_time_ms = 0
        for group in tool_groups:
            # Group time is max tool time (parallel)
            group_time = max(
                self.executor.registry.get_tool(req.tool_name).schema.timeout_seconds
                * 1000
                if self.executor.registry.get_tool(req.tool_name)
                else 30000
                for req in group
            )
            estimated_time_ms += group_time

        return ParallelExecutionPlan(
            tool_groups=tool_groups,
            dependencies=deps,
            estimated_time_ms=estimated_time_ms,
            max_parallelism=max_parallel,
        )

    def _topological_sort(
        self,
        tools: List[str],
        dependencies: Dict[str, List[str]],
    ) -> List[List[str]]:
        """
        Topological sort to group tools by dependency level.

        Args:
            tools: List of tool names
            dependencies: Tool dependencies

        Returns:
            List of groups (each group can execute in parallel)
        """
        # Calculate in-degree for each tool
        in_degree: Dict[str, int] = {tool: 0 for tool in tools}
        graph: Dict[str, List[str]] = {tool: [] for tool in tools}

        # Build graph
        for tool, deps in dependencies.items():
            if tool in tools:
                for dep in deps:
                    if dep in tools:
                        graph[dep].append(tool)
                        in_degree[tool] += 1

        # Group tools by level
        groups: List[List[str]] = []
        remaining = set(tools)

        while remaining:
            # Find tools with no dependencies
            ready = [tool for tool in remaining if in_degree[tool] == 0]

            if not ready:
                # Circular dependency or invalid graph
                # Add remaining tools to final group
                self.logger.warning(
                    f"Circular dependency detected. "
                    f"Adding {len(remaining)} tools to final group"
                )
                groups.append(list(remaining))
                break

            # Add ready tools as new group
            groups.append(ready)

            # Remove ready tools and update in-degrees
            for tool in ready:
                remaining.remove(tool)
                for dependent in graph[tool]:
                    in_degree[dependent] -= 1

        return groups
