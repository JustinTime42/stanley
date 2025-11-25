"""AST parsing engine using Tree-sitter."""

import logging
from pathlib import Path
from typing import Optional, Dict
import asyncio

from tree_sitter_language_pack import get_parser, get_language
from tree_sitter import Parser, Language

from ..models.analysis_models import ASTNode, Language as LangEnum

logger = logging.getLogger(__name__)


class ASTParser:
    """
    Tree-sitter based AST parser.

    PATTERN: Tree-sitter integration with caching
    CRITICAL: Load languages once, reuse parser
    GOTCHA: Tree-sitter uses bytes, not strings
    """

    # Language mapping from our enum to tree-sitter names
    LANGUAGE_MAPPING: Dict[LangEnum, str] = {
        LangEnum.PYTHON: "python",
        LangEnum.JAVASCRIPT: "javascript",
        LangEnum.TYPESCRIPT: "typescript",
        LangEnum.JAVA: "java",
        LangEnum.GO: "go",
    }

    # File extension to language mapping
    EXTENSION_MAPPING: Dict[str, LangEnum] = {
        ".py": LangEnum.PYTHON,
        ".js": LangEnum.JAVASCRIPT,
        ".jsx": LangEnum.JAVASCRIPT,
        ".ts": LangEnum.TYPESCRIPT,
        ".tsx": LangEnum.TYPESCRIPT,
        ".java": LangEnum.JAVA,
        ".go": LangEnum.GO,
    }

    def __init__(self):
        """Initialize AST parser and load language grammars."""
        self.parsers: Dict[str, Parser] = {}
        self.languages: Dict[str, Language] = {}
        self.logger = logger
        self._load_languages()

    def _load_languages(self):
        """
        Load Tree-sitter language grammars.

        PATTERN: Pre-load all supported languages for performance
        GOTCHA: tree-sitter-language-pack provides pre-compiled grammars
        """
        for lang_enum, lang_name in self.LANGUAGE_MAPPING.items():
            try:
                # Get pre-compiled parser from tree-sitter-language-pack
                parser = get_parser(lang_name)
                language = get_language(lang_name)

                self.parsers[lang_name] = parser
                self.languages[lang_name] = language
                self.logger.info(f"Loaded {lang_name} language grammar")

            except Exception as e:
                self.logger.error(f"Failed to load {lang_name} grammar: {e}")

    def detect_language(self, file_path: str) -> LangEnum:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to source file

        Returns:
            Detected language or UNKNOWN
        """
        extension = Path(file_path).suffix.lower()
        return self.EXTENSION_MAPPING.get(extension, LangEnum.UNKNOWN)

    async def parse_file(
        self, file_path: str, language: Optional[LangEnum] = None
    ) -> Optional[ASTNode]:
        """
        Parse file into AST.

        CRITICAL: Tree-sitter uses bytes, not strings
        GOTCHA: Must handle encoding errors gracefully

        Args:
            file_path: Path to source file
            language: Optional language override (auto-detect if None)

        Returns:
            ASTNode representing the parse tree, or None on error
        """
        # Detect language if not provided
        if language is None:
            language = self.detect_language(file_path)

        if language == LangEnum.UNKNOWN:
            self.logger.error(f"Unknown language for file: {file_path}")
            return None

        # Read file content
        try:
            with open(file_path, "rb") as f:
                content = f.read()
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return None

        # Parse using tree-sitter
        return await self.parse_code(content, language, file_path)

    async def parse_code(
        self,
        code: bytes,
        language: LangEnum,
        file_path: str = "<string>",
    ) -> Optional[ASTNode]:
        """
        Parse code string into AST.

        Args:
            code: Source code as bytes
            language: Programming language
            file_path: Optional file path for error reporting

        Returns:
            ASTNode representing the parse tree, or None on error
        """
        lang_name = self.LANGUAGE_MAPPING.get(language)

        if not lang_name or lang_name not in self.parsers:
            self.logger.error(f"Unsupported language: {language}")
            return None

        try:
            # Get parser for this language
            parser = self.parsers[lang_name]

            # Parse code (run in thread pool to avoid blocking)
            loop = asyncio.get_event_loop()
            tree = await loop.run_in_executor(None, parser.parse, code)

            # Convert to our AST format
            if tree and tree.root_node:
                return self._convert_tree_to_ast(tree.root_node, code)
            else:
                self.logger.error(f"Failed to parse {file_path}")
                return None

        except Exception as e:
            self.logger.error(f"Parse error in {file_path}: {e}")
            return None

    def _convert_tree_to_ast(
        self, node, source_code: bytes, max_depth: int = 100, current_depth: int = 0
    ) -> ASTNode:
        """
        Convert Tree-sitter node to our AST model.

        CRITICAL: Must be recursive but watch depth
        GOTCHA: Large ASTs can cause stack overflow

        Args:
            node: Tree-sitter node
            source_code: Original source code bytes
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth

        Returns:
            ASTNode model
        """
        # Prevent stack overflow on very deep trees
        if current_depth >= max_depth:
            self.logger.warning(f"Max AST depth {max_depth} reached")
            return ASTNode(
                node_type=node.type,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_point=(node.start_point[0], node.start_point[1]),
                end_point=(node.end_point[0], node.end_point[1]),
                text="<truncated>",
                children=[],
            )

        # Extract text safely
        try:
            text = source_code[node.start_byte : node.end_byte].decode("utf-8")
            # Limit text length to prevent huge nodes
            if len(text) > 5000:
                text = text[:5000] + "..."
        except Exception as e:
            self.logger.warning(f"Failed to decode node text: {e}")
            text = "<decode error>"

        # Convert children recursively
        children = []
        for child in node.children:
            try:
                child_ast = self._convert_tree_to_ast(
                    child, source_code, max_depth, current_depth + 1
                )
                children.append(child_ast)
            except Exception as e:
                self.logger.warning(f"Failed to convert child node: {e}")

        return ASTNode(
            node_type=node.type,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_point=(node.start_point[0], node.start_point[1]),
            end_point=(node.end_point[0], node.end_point[1]),
            text=text,
            children=children,
            metadata={
                "is_named": node.is_named,
                "has_error": node.has_error,
            },
        )

    def traverse_ast(
        self,
        node: ASTNode,
        node_type: Optional[str] = None,
        max_results: int = 1000,
    ) -> list[ASTNode]:
        """
        Traverse AST and collect nodes of a specific type.

        Args:
            node: Root AST node
            node_type: Optional node type to filter (None for all)
            max_results: Maximum number of results to return

        Returns:
            List of matching AST nodes
        """
        results = []

        def visit(n: ASTNode):
            if len(results) >= max_results:
                return

            if node_type is None or n.node_type == node_type:
                results.append(n)

            for child in n.children:
                visit(child)

        visit(node)
        return results

    def find_nodes_by_type(self, ast: ASTNode, node_types: list[str]) -> list[ASTNode]:
        """
        Find all nodes matching any of the given types.

        Args:
            ast: Root AST node
            node_types: List of node types to search for

        Returns:
            List of matching nodes
        """
        results = []
        node_type_set = set(node_types)

        def visit(node: ASTNode):
            if node.node_type in node_type_set:
                results.append(node)

            for child in node.children:
                visit(child)

        visit(ast)
        return results
