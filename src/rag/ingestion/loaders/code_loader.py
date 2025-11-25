"""Code file loader for source code documents."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import uuid

from ....models.document_models import Document, DocumentType
from ....analysis.ast_parser import ASTParser
from ....models.analysis_models import Language

logger = logging.getLogger(__name__)


class CodeLoader:
    """
    Loader for source code files.

    PATTERN: Extract structure using AST parser
    CRITICAL: Preserve language metadata for chunking
    """

    def __init__(self, ast_parser: Optional[ASTParser] = None):
        """
        Initialize code loader.

        Args:
            ast_parser: Optional AST parser (creates if None)
        """
        self.ast_parser = ast_parser or ASTParser()
        self.logger = logger

    async def load_file(self, file_path: str) -> Optional[Document]:
        """
        Load a source code file.

        Args:
            file_path: Path to code file

        Returns:
            Document with code content and metadata
        """
        try:
            path = Path(file_path)

            # Read file content
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Detect language
            language = self.ast_parser.detect_language(str(path))

            # Extract metadata using AST
            metadata = await self.extract_metadata(content, language, str(path))

            # Create document
            document = Document(
                id=str(uuid.uuid4()),
                content=content,
                type=DocumentType.CODE,
                source=str(path),
                metadata=metadata,
            )

            self.logger.info(f"Loaded code file: {path} ({language.value})")
            return document

        except Exception as e:
            self.logger.error(f"Failed to load code file {file_path}: {e}")
            return None

    async def extract_metadata(
        self,
        code: str,
        language: Language,
        file_path: str = "<string>",
    ) -> Dict[str, Any]:
        """
        Extract metadata from code using AST.

        Args:
            code: Source code content
            language: Programming language
            file_path: File path for context

        Returns:
            Metadata dictionary
        """
        metadata = {
            "language": language.value,
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "extension": Path(file_path).suffix,
            "line_count": len(code.splitlines()),
        }

        # Parse AST if supported language
        if language != Language.UNKNOWN:
            try:
                ast = await self.ast_parser.parse_code(
                    code.encode("utf-8"),
                    language,
                    file_path,
                )

                if ast:
                    # Extract function and class names
                    functions = self._extract_function_names(ast)
                    classes = self._extract_class_names(ast)

                    metadata.update(
                        {
                            "function_count": len(functions),
                            "class_count": len(classes),
                            "functions": functions,
                            "classes": classes,
                            "has_ast": True,
                        }
                    )
                else:
                    metadata["has_ast"] = False

            except Exception as e:
                self.logger.warning(f"Failed to parse AST for {file_path}: {e}")
                metadata["has_ast"] = False
        else:
            metadata["has_ast"] = False

        return metadata

    def _extract_function_names(self, ast_node) -> list[str]:
        """Extract function/method names from AST."""
        # Language-specific function node types
        function_types = [
            "function_definition",  # Python
            "function_declaration",  # JavaScript, TypeScript
            "method_definition",  # JavaScript, TypeScript
            "method_declaration",  # Java
            "function_declaration",  # Go
        ]

        functions = []
        for node in self.ast_parser.find_nodes_by_type(ast_node, function_types):
            # Try to extract name from node text
            name = self._extract_name_from_node(node)
            if name:
                functions.append(name)

        return functions

    def _extract_class_names(self, ast_node) -> list[str]:
        """Extract class names from AST."""
        class_types = [
            "class_definition",  # Python
            "class_declaration",  # JavaScript, TypeScript, Java
            "type_declaration",  # Go (struct)
        ]

        classes = []
        for node in self.ast_parser.find_nodes_by_type(ast_node, class_types):
            name = self._extract_name_from_node(node)
            if name:
                classes.append(name)

        return classes

    def _extract_name_from_node(self, node) -> Optional[str]:
        """Extract identifier name from AST node."""
        # Look for identifier child nodes
        for child in node.children:
            if child.node_type == "identifier":
                # Clean up the name
                name = child.text.strip()
                if name and len(name) < 100:  # Sanity check
                    return name

        return None

    def supports_file(self, file_path: str) -> bool:
        """
        Check if loader supports this file.

        Args:
            file_path: Path to file

        Returns:
            True if file extension is supported
        """
        extension = Path(file_path).suffix.lower()
        supported_extensions = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go"}
        return extension in supported_extensions
