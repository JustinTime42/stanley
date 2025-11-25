"""Code validation and linting tools."""

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


class RuffTool(BaseTool):
    """Tool for running Ruff linter."""

    def __init__(self):
        """Initialize Ruff tool."""
        super().__init__(
            name="ruff",
            category=ToolCategory.VALIDATION,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Run Ruff linter on Python code",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to file or directory to lint",
                    required=True,
                ),
                ToolParameter(
                    name="fix",
                    type="boolean",
                    description="Auto-fix issues",
                    required=False,
                    default=False,
                ),
            ],
            returns="Linting results",
            timeout_seconds=30,
        )

    async def execute(self, path: str, fix: bool = False, **kwargs) -> ToolResult:
        """Execute Ruff linting."""
        start_time = datetime.now()

        try:
            cmd = ["ruff", "check", path]
            if fix:
                cmd.append("--fix")

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
                    "output": stdout.decode(),
                    "errors": stderr.decode(),
                    "path": path,
                    "fixed": fix,
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Ruff execution failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )


class MypyTool(BaseTool):
    """Tool for running Mypy type checker."""

    def __init__(self):
        """Initialize Mypy tool."""
        super().__init__(
            name="mypy",
            category=ToolCategory.VALIDATION,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Run Mypy type checker on Python code",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to file or directory to check",
                    required=True,
                ),
                ToolParameter(
                    name="strict",
                    type="boolean",
                    description="Use strict mode",
                    required=False,
                    default=False,
                ),
            ],
            returns="Type checking results",
            timeout_seconds=60,
        )

    async def execute(self, path: str, strict: bool = False, **kwargs) -> ToolResult:
        """Execute Mypy type checking."""
        start_time = datetime.now()

        try:
            cmd = ["mypy", path]
            if strict:
                cmd.append("--strict")

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
                    "output": stdout.decode(),
                    "errors": stderr.decode(),
                    "path": path,
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Mypy execution failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )


class BlackTool(BaseTool):
    """Tool for running Black formatter."""

    def __init__(self):
        """Initialize Black tool."""
        super().__init__(
            name="black",
            category=ToolCategory.VALIDATION,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Run Black code formatter on Python code",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to file or directory to format",
                    required=True,
                ),
                ToolParameter(
                    name="check",
                    type="boolean",
                    description="Check only, don't modify files",
                    required=False,
                    default=False,
                ),
            ],
            returns="Formatting results",
            timeout_seconds=30,
        )

    async def execute(self, path: str, check: bool = False, **kwargs) -> ToolResult:
        """Execute Black formatting."""
        start_time = datetime.now()

        try:
            cmd = ["black", path]
            if check:
                cmd.append("--check")

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
                    "output": stdout.decode(),
                    "errors": stderr.decode(),
                    "path": path,
                    "check_only": check,
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Black execution failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )
