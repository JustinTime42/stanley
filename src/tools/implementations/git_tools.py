"""Git version control tools."""

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


class GitStatusTool(BaseTool):
    """Tool for git status."""

    def __init__(self):
        """Initialize git status tool."""
        super().__init__(
            name="git_status",
            category=ToolCategory.VERSION_CONTROL,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Get git repository status",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="repo_path",
                    type="string",
                    description="Path to git repository",
                    required=False,
                    default=".",
                ),
            ],
            returns="Git status output",
            timeout_seconds=10,
        )

    async def execute(self, repo_path: str = ".", **kwargs) -> ToolResult:
        """Execute git status."""
        start_time = datetime.now()

        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                "status",
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return self._create_success_result(
                result={
                    "status": stdout.decode(),
                    "repo_path": repo_path,
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Git status failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )


class GitCommitTool(BaseTool):
    """Tool for git commit."""

    def __init__(self):
        """Initialize git commit tool."""
        super().__init__(
            name="git_commit",
            category=ToolCategory.VERSION_CONTROL,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Create a git commit",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    description="Commit message",
                    required=True,
                ),
                ToolParameter(
                    name="repo_path",
                    type="string",
                    description="Path to git repository",
                    required=False,
                    default=".",
                ),
                ToolParameter(
                    name="add_all",
                    type="boolean",
                    description="Add all changes before commit",
                    required=False,
                    default=True,
                ),
            ],
            returns="Commit result",
            timeout_seconds=15,
            requires_confirmation=True,
        )

    async def execute(
        self, message: str, repo_path: str = ".", add_all: bool = True, **kwargs
    ) -> ToolResult:
        """Execute git commit."""
        start_time = datetime.now()

        try:
            # Add all if requested
            if add_all:
                await asyncio.create_subprocess_exec(
                    "git",
                    "add",
                    "-A",
                    cwd=repo_path,
                )

            # Commit
            process = await asyncio.create_subprocess_exec(
                "git",
                "commit",
                "-m",
                message,
                cwd=repo_path,
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
                    "message": message,
                    "output": stdout.decode(),
                    "repo_path": repo_path,
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Git commit failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )


class GitDiffTool(BaseTool):
    """Tool for git diff."""

    def __init__(self):
        """Initialize git diff tool."""
        super().__init__(
            name="git_diff",
            category=ToolCategory.VERSION_CONTROL,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Get git diff",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="repo_path",
                    type="string",
                    description="Path to git repository",
                    required=False,
                    default=".",
                ),
                ToolParameter(
                    name="staged",
                    type="boolean",
                    description="Show staged changes only",
                    required=False,
                    default=False,
                ),
            ],
            returns="Git diff output",
            timeout_seconds=15,
        )

    async def execute(
        self, repo_path: str = ".", staged: bool = False, **kwargs
    ) -> ToolResult:
        """Execute git diff."""
        start_time = datetime.now()

        try:
            cmd = ["git", "diff"]
            if staged:
                cmd.append("--staged")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return self._create_success_result(
                result={
                    "diff": stdout.decode(),
                    "repo_path": repo_path,
                    "staged": staged,
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Git diff failed: {str(e)}",
                execution_time_ms=execution_time_ms,
            )
