"""Structural code search using AST patterns."""

import logging
from typing import List, Optional

from ...models.document_models import Chunk
from ...analysis.ast_parser import ASTParser
from ...models.analysis_models import Language

logger = logging.getLogger(__name__)


class StructuralSearch:
    """
    Search for code patterns using AST structure.

    PATTERN: AST-based pattern matching
    CRITICAL: Match code structure, not just text
    """

    def __init__(self, ast_parser: Optional[ASTParser] = None):
        """
        Initialize structural search.

        Args:
            ast_parser: Optional AST parser (creates if None)
        """
        self.ast_parser = ast_parser or ASTParser()
        self.logger = logger

    async def search_by_structure(
        self,
        chunks: List[Chunk],
        pattern: str,
        language: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Search chunks for structural code patterns.

        Args:
            chunks: List of code chunks to search
            pattern: Pattern to search for (e.g., "function_definition")
            language: Optional language filter

        Returns:
            Matching chunks
        """
        matching_chunks = []

        for chunk in chunks:
            # Filter by language if specified
            if language and chunk.language != language:
                continue

            # Only search code chunks
            if chunk.chunk_type != "code":
                continue

            # Check if chunk matches pattern
            if await self._matches_pattern(chunk, pattern):
                matching_chunks.append(chunk)

        self.logger.debug(
            f"Structural search found {len(matching_chunks)} matches for '{pattern}'"
        )

        return matching_chunks

    async def _matches_pattern(self, chunk: Chunk, pattern: str) -> bool:
        """
        Check if chunk matches structural pattern.

        Args:
            chunk: Code chunk
            pattern: Pattern to match

        Returns:
            True if chunk matches pattern
        """
        if not chunk.language:
            return False

        # Map language string to enum
        try:
            lang = Language(chunk.language.lower())
        except ValueError:
            return False

        # Parse chunk AST
        try:
            ast = await self.ast_parser.parse_code(
                chunk.content.encode("utf-8"),
                lang,
            )

            if not ast:
                return False

            # Search for pattern in AST
            nodes = self.ast_parser.find_nodes_by_type(ast, [pattern])
            return len(nodes) > 0

        except Exception as e:
            self.logger.warning(f"Failed to parse chunk {chunk.id}: {e}")
            return False

    async def find_function_definitions(
        self,
        chunks: List[Chunk],
        function_name: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Find chunks containing function definitions.

        Args:
            chunks: Chunks to search
            function_name: Optional specific function name

        Returns:
            Chunks containing function definitions
        """
        # Function definition node types across languages
        function_types = [
            "function_definition",
            "function_declaration",
            "method_definition",
            "method_declaration",
        ]

        matching_chunks = []

        for chunk in chunks:
            if chunk.chunk_type != "code":
                continue

            # If looking for specific function
            if function_name and chunk.function_name:
                if chunk.function_name.lower() == function_name.lower():
                    matching_chunks.append(chunk)
                    continue

            # Otherwise, search for any function definition
            for func_type in function_types:
                if await self._matches_pattern(chunk, func_type):
                    matching_chunks.append(chunk)
                    break

        return matching_chunks

    async def find_class_definitions(
        self,
        chunks: List[Chunk],
        class_name: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Find chunks containing class definitions.

        Args:
            chunks: Chunks to search
            class_name: Optional specific class name

        Returns:
            Chunks containing class definitions
        """
        class_types = [
            "class_definition",
            "class_declaration",
            "type_declaration",  # Go structs
        ]

        matching_chunks = []

        for chunk in chunks:
            if chunk.chunk_type != "code":
                continue

            # If looking for specific class
            if class_name and chunk.class_name:
                if chunk.class_name.lower() == class_name.lower():
                    matching_chunks.append(chunk)
                    continue

            # Otherwise, search for any class definition
            for class_type in class_types:
                if await self._matches_pattern(chunk, class_type):
                    matching_chunks.append(chunk)
                    break

        return matching_chunks
