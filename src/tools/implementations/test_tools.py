"""Testing tools for running tests."""

import asyncio
import logging
from datetime import datetime

from ..base import BaseTool
from ...models.tool_models import (
    ToolCategory,
    ToolSchema,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger(__name__)


class PytestTool(BaseTool):
    """Tool for running pytest tests."""

    def __init__(self):
        """Initialize pytest tool."""
        super().__init__(
            name="pytest",
            category=ToolCategory.TESTING,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Run pytest tests",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to test file or directory",
                    required=True,
                ),
                ToolParameter(
                    name="args",
                    type="string",
                    description="Additional pytest arguments",
                    required=False,
                    default="",
                ),
                ToolParameter(
                    name="verbose",
                    type="boolean",
                    description="Verbose output",
                    required=False,
                    default=True,
                ),
            ],
            returns="Test results",
            timeout_seconds=120,
        )

    async def execute(
        self, path: str, args: str = "", verbose: bool = True, **kwargs
    ) -> ToolResult:
        """
        Execute pytest.

        Args:
            path: Test path
            args: Additional arguments
            verbose: Verbose output

        Returns:
            ToolResult with test results
        """
        start_time = datetime.now()

        try:
            # Build command
            cmd = ["pytest", path]
            if verbose:
                cmd.append("-v")
            if args:
                cmd.extend(args.split())

            # Run pytest
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            success = process.returncode == 0

            return self._create_success_result(
                result={
                    "success": success,
                    "returncode": process.returncode,
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "path": path,
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Pytest execution failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )


class UnittestTool(BaseTool):
    """Tool for running unittest tests."""

    def __init__(self):
        """Initialize unittest tool."""
        super().__init__(
            name="unittest",
            category=ToolCategory.TESTING,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Run unittest tests",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to test module",
                    required=True,
                ),
                ToolParameter(
                    name="verbose",
                    type="boolean",
                    description="Verbose output",
                    required=False,
                    default=True,
                ),
            ],
            returns="Test results",
            timeout_seconds=120,
        )

    async def execute(self, path: str, verbose: bool = True, **kwargs) -> ToolResult:
        """Execute unittest."""
        start_time = datetime.now()

        try:
            # Build command
            cmd = ["python", "-m", "unittest"]
            if verbose:
                cmd.append("-v")
            cmd.append(path)

            # Run unittest
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            success = process.returncode == 0

            return self._create_success_result(
                result={
                    "success": success,
                    "returncode": process.returncode,
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "path": path,
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Unittest execution failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )
