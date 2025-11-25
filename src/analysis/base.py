"""Base analyzer class for all code analyzers."""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List
from pathlib import Path

from ..models.analysis_models import (
    Language,
    ASTNode,
    CodeEntity,
)

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """
    Abstract base class for all code analyzers.

    PATTERN: Abstract base for language-specific analyzers
    CRITICAL: Must be async and return AnalysisResult
    GOTCHA: All analyze methods must be async for agent integration
    """

    def __init__(self, language: Language):
        """
        Initialize base analyzer.

        Args:
            language: Programming language this analyzer handles
        """
        self.language = language
        self.logger = logging.getLogger(f"{__name__}.{language.value}")

    @abstractmethod
    async def analyze(self, file_path: str, ast: ASTNode, **kwargs) -> List[CodeEntity]:
        """
        Analyze code and extract entities.

        CRITICAL: Must be async and return list of CodeEntity
        CRITICAL: Results must be JSON-serializable

        Args:
            file_path: Path to file being analyzed
            ast: Parsed AST of the file
            **kwargs: Language-specific parameters

        Returns:
            List of code entities found
        """
        pass

    @abstractmethod
    def get_language(self) -> Language:
        """
        Get the language this analyzer handles.

        Returns:
            Language enum value
        """
        pass

    async def extract_functions(self, ast: ASTNode) -> List[CodeEntity]:
        """
        Extract function definitions from AST.

        Args:
            ast: Parsed AST

        Returns:
            List of function entities
        """
        # Default implementation - override in subclasses
        return []

    async def extract_classes(self, ast: ASTNode) -> List[CodeEntity]:
        """
        Extract class definitions from AST.

        Args:
            ast: Parsed AST

        Returns:
            List of class entities
        """
        # Default implementation - override in subclasses
        return []

    async def extract_imports(self, ast: ASTNode) -> List[str]:
        """
        Extract import statements from AST.

        Args:
            ast: Parsed AST

        Returns:
            List of imported module names
        """
        # Default implementation - override in subclasses
        return []

    async def extract_docstring(self, node: ASTNode) -> Optional[str]:
        """
        Extract docstring from a node.

        Args:
            node: AST node (function, class, etc.)

        Returns:
            Docstring text if found
        """
        # Default implementation - override in subclasses
        return None

    def _validate_file_path(self, file_path: str) -> bool:
        """
        Validate that file path exists and is a file.

        Args:
            file_path: Path to validate

        Returns:
            True if valid, False otherwise
        """
        path = Path(file_path)
        if not path.exists():
            self.logger.error(f"File does not exist: {file_path}")
            return False

        if not path.is_file():
            self.logger.error(f"Path is not a file: {file_path}")
            return False

        return True

    def _get_file_extension(self, file_path: str) -> str:
        """
        Get file extension.

        Args:
            file_path: Path to file

        Returns:
            File extension (e.g., ".py", ".js")
        """
        return Path(file_path).suffix
