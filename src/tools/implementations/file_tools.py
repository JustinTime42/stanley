"""File system tools for agent operations."""

import os
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    import aiofiles
    import aiofiles.os
except ImportError:
    aiofiles = None

from ..base import BaseTool
from ...models.tool_models import (
    ToolCategory,
    ToolSchema,
    ToolParameter,
    ToolResult,
)

logger = logging.getLogger(__name__)


class ReadFileTool(BaseTool):
    """
    Tool for reading file contents.

    PATTERN: Concrete tool implementation
    GOTCHA: Must use async file operations
    """

    def __init__(self):
        """Initialize read file tool."""
        super().__init__(
            name="read_file",
            category=ToolCategory.FILE_SYSTEM,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Read contents of a file",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to file to read",
                    required=True,
                ),
                ToolParameter(
                    name="encoding",
                    type="string",
                    description="File encoding",
                    required=False,
                    default="utf-8",
                ),
            ],
            returns="File contents as string",
            examples=[
                {
                    "path": "src/main.py",
                    "encoding": "utf-8",
                }
            ],
            timeout_seconds=10,
            max_retries=2,
        )

    async def execute(self, path: str, encoding: str = "utf-8", **kwargs) -> ToolResult:
        """
        Execute file read.

        Args:
            path: Path to file
            encoding: File encoding

        Returns:
            ToolResult with file contents
        """
        start_time = datetime.now()

        try:
            # Use aiofiles if available, otherwise fallback to sync
            if aiofiles:
                async with aiofiles.open(path, "r", encoding=encoding) as f:
                    content = await f.read()
            else:
                with open(path, "r", encoding=encoding) as f:
                    content = f.read()

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return self._create_success_result(
                result={
                    "content": content,
                    "path": path,
                    "size_bytes": len(content.encode(encoding)),
                    "encoding": encoding,
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Failed to read file '{path}': {str(e)}",
                execution_time_ms=execution_time_ms,
            )


class WriteFileTool(BaseTool):
    """Tool for writing file contents."""

    def __init__(self):
        """Initialize write file tool."""
        super().__init__(
            name="write_file",
            category=ToolCategory.FILE_SYSTEM,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Write contents to a file",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to file to write",
                    required=True,
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write",
                    required=True,
                ),
                ToolParameter(
                    name="encoding",
                    type="string",
                    description="File encoding",
                    required=False,
                    default="utf-8",
                ),
                ToolParameter(
                    name="create_dirs",
                    type="boolean",
                    description="Create parent directories if they don't exist",
                    required=False,
                    default=True,
                ),
            ],
            returns="Write confirmation with file info",
            examples=[
                {
                    "path": "src/output.py",
                    "content": "# Python code",
                    "encoding": "utf-8",
                    "create_dirs": True,
                }
            ],
            timeout_seconds=15,
            max_retries=2,
        )

    async def execute(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True,
        **kwargs,
    ) -> ToolResult:
        """
        Execute file write.

        Args:
            path: Path to file
            content: Content to write
            encoding: File encoding
            create_dirs: Create parent directories

        Returns:
            ToolResult with write confirmation
        """
        start_time = datetime.now()

        try:
            # Create parent directories if needed
            if create_dirs:
                parent_dir = os.path.dirname(path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)

            # Use aiofiles if available
            if aiofiles:
                async with aiofiles.open(path, "w", encoding=encoding) as f:
                    await f.write(content)
            else:
                with open(path, "w", encoding=encoding) as f:
                    f.write(content)

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return self._create_success_result(
                result={
                    "path": path,
                    "size_bytes": len(content.encode(encoding)),
                    "encoding": encoding,
                    "created": not os.path.exists(path),
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Failed to write file '{path}': {str(e)}",
                execution_time_ms=execution_time_ms,
            )


class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents."""

    def __init__(self):
        """Initialize list directory tool."""
        super().__init__(
            name="list_directory",
            category=ToolCategory.FILE_SYSTEM,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="List contents of a directory",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to directory",
                    required=True,
                ),
                ToolParameter(
                    name="recursive",
                    type="boolean",
                    description="List recursively",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="pattern",
                    type="string",
                    description="Glob pattern to filter files",
                    required=False,
                    default=None,
                ),
            ],
            returns="List of files and directories",
            timeout_seconds=20,
        )

    async def execute(
        self,
        path: str,
        recursive: bool = False,
        pattern: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute directory listing.

        Args:
            path: Directory path
            recursive: List recursively
            pattern: Glob pattern filter

        Returns:
            ToolResult with directory contents
        """
        start_time = datetime.now()

        try:
            path_obj = Path(path)

            if not path_obj.exists():
                return self._create_error_result(
                    error=f"Directory '{path}' does not exist",
                    execution_time_ms=0,
                )

            if not path_obj.is_dir():
                return self._create_error_result(
                    error=f"Path '{path}' is not a directory",
                    execution_time_ms=0,
                )

            # List files
            if recursive:
                if pattern:
                    files = [str(p) for p in path_obj.rglob(pattern)]
                else:
                    files = [str(p) for p in path_obj.rglob("*")]
            else:
                if pattern:
                    files = [str(p) for p in path_obj.glob(pattern)]
                else:
                    files = [str(p) for p in path_obj.iterdir()]

            # Sort files
            files.sort()

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return self._create_success_result(
                result={
                    "path": path,
                    "files": files,
                    "count": len(files),
                    "recursive": recursive,
                    "pattern": pattern,
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Failed to list directory '{path}': {str(e)}",
                execution_time_ms=execution_time_ms,
            )


class CreateDirectoryTool(BaseTool):
    """Tool for creating directories."""

    def __init__(self):
        """Initialize create directory tool."""
        super().__init__(
            name="create_directory",
            category=ToolCategory.FILE_SYSTEM,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Create a directory",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to directory to create",
                    required=True,
                ),
                ToolParameter(
                    name="exist_ok",
                    type="boolean",
                    description="Don't error if directory already exists",
                    required=False,
                    default=True,
                ),
            ],
            returns="Directory creation confirmation",
            timeout_seconds=5,
        )

    async def execute(self, path: str, exist_ok: bool = True, **kwargs) -> ToolResult:
        """
        Execute directory creation.

        Args:
            path: Directory path
            exist_ok: Allow existing directory

        Returns:
            ToolResult with creation confirmation
        """
        start_time = datetime.now()

        try:
            os.makedirs(path, exist_ok=exist_ok)

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return self._create_success_result(
                result={
                    "path": path,
                    "created": not os.path.exists(path),
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Failed to create directory '{path}': {str(e)}",
                execution_time_ms=execution_time_ms,
            )


class DeleteFileTool(BaseTool):
    """Tool for deleting files."""

    def __init__(self):
        """Initialize delete file tool."""
        super().__init__(
            name="delete_file",
            category=ToolCategory.FILE_SYSTEM,
        )

    def _build_schema(self) -> ToolSchema:
        """Build tool schema."""
        return ToolSchema(
            name=self.name,
            description="Delete a file",
            category=self.category,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to file to delete",
                    required=True,
                ),
            ],
            returns="Deletion confirmation",
            timeout_seconds=5,
            requires_confirmation=True,  # Safety measure
        )

    async def execute(self, path: str, **kwargs) -> ToolResult:
        """
        Execute file deletion.

        Args:
            path: File path

        Returns:
            ToolResult with deletion confirmation
        """
        start_time = datetime.now()

        try:
            if not os.path.exists(path):
                return self._create_error_result(
                    error=f"File '{path}' does not exist",
                    execution_time_ms=0,
                )

            os.remove(path)

            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            return self._create_success_result(
                result={
                    "path": path,
                    "deleted": True,
                },
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            return self._create_error_result(
                error=f"Failed to delete file '{path}': {str(e)}",
                execution_time_ms=execution_time_ms,
            )
