"""Base extractor interface."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, TypeVar, Generic

from ..models import FileInfo

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseExtractor(ABC, Generic[T]):
    """
    Abstract base class for code extractors.

    PATTERN: Template method for extraction pipeline
    """

    def __init__(self):
        """Initialize extractor."""
        self.logger = logger
        self._errors: List[str] = []

    @abstractmethod
    async def extract(self, *args, **kwargs) -> T:
        """
        Extract information from code.

        Returns:
            Extracted information
        """
        pass

    def get_errors(self) -> List[str]:
        """Get extraction errors."""
        return self._errors.copy()

    def clear_errors(self) -> None:
        """Clear extraction errors."""
        self._errors.clear()

    def _add_error(self, error: str) -> None:
        """Add an error message."""
        self._errors.append(error)
        self.logger.warning(error)

    def _read_file_content(self, file_path: str | Path) -> str | None:
        """
        Read file content safely.

        Args:
            file_path: Path to file

        Returns:
            File content or None on error
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            self._add_error(f"Failed to read {file_path}: {e}")
            return None

    def _read_file_bytes(self, file_path: str | Path) -> bytes | None:
        """
        Read file as bytes safely.

        Args:
            file_path: Path to file

        Returns:
            File bytes or None on error
        """
        try:
            with open(file_path, "rb") as f:
                return f.read()
        except Exception as e:
            self._add_error(f"Failed to read {file_path}: {e}")
            return None

    def _is_source_file(self, file_info: FileInfo) -> bool:
        """Check if file is a source code file."""
        return file_info.language not in ("unknown", "text", "markdown", "rst")
