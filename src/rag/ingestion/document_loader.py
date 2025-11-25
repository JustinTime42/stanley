"""Document loader orchestrator managing all format-specific loaders."""

import logging
from pathlib import Path
from typing import Optional, List
import asyncio

from ...models.document_models import Document, DocumentType
from .loaders.code_loader import CodeLoader
from .loaders.markdown_loader import MarkdownLoader
from .loaders.json_loader import JSONLoader

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Orchestrator for document loading across multiple formats.

    PATTERN: Factory pattern for loader selection
    CRITICAL: Auto-detect document type from extension
    """

    def __init__(self):
        """Initialize document loader with all format loaders."""
        self.logger = logger

        # Initialize format-specific loaders
        self.code_loader = CodeLoader()
        self.markdown_loader = MarkdownLoader()
        self.json_loader = JSONLoader()

        # Map extensions to loaders
        self.extension_map = {
            # Code files
            ".py": self.code_loader,
            ".js": self.code_loader,
            ".jsx": self.code_loader,
            ".ts": self.code_loader,
            ".tsx": self.code_loader,
            ".java": self.code_loader,
            ".go": self.code_loader,
            # Markdown
            ".md": self.markdown_loader,
            ".markdown": self.markdown_loader,
            # JSON/YAML
            ".json": self.json_loader,
            ".yaml": self.json_loader,
            ".yml": self.json_loader,
            # Plain text
            ".txt": "text",
            ".rst": "text",
        }

    async def load_document(
        self,
        file_path: str,
        document_type: Optional[DocumentType] = None,
    ) -> Optional[Document]:
        """
        Load a document from file.

        Args:
            file_path: Path to document
            document_type: Optional forced document type

        Returns:
            Loaded document or None on error
        """
        path = Path(file_path)

        if not path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None

        if not path.is_file():
            self.logger.error(f"Not a file: {file_path}")
            return None

        # Detect type from extension if not provided
        if document_type is None:
            document_type = self.detect_type(str(path))

        # Get appropriate loader
        extension = path.suffix.lower()
        loader = self.extension_map.get(extension)

        if loader == "text":
            # Handle plain text files directly
            return await self._load_text_file(str(path))
        elif loader:
            # Use format-specific loader
            return await loader.load_file(str(path))
        else:
            # Fallback to text
            self.logger.warning(f"No specific loader for {extension}, treating as text")
            return await self._load_text_file(str(path))

    async def load_directory(
        self,
        directory_path: str,
        recursive: bool = False,
        file_patterns: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load all documents from a directory.

        Args:
            directory_path: Path to directory
            recursive: Recursively process subdirectories
            file_patterns: Optional glob patterns to filter files

        Returns:
            List of loaded documents
        """
        path = Path(directory_path)

        if not path.exists() or not path.is_dir():
            self.logger.error(f"Invalid directory: {directory_path}")
            return []

        # Collect files
        files = []
        if recursive:
            if file_patterns:
                for pattern in file_patterns:
                    files.extend(path.rglob(pattern))
            else:
                files.extend(path.rglob("*"))
        else:
            if file_patterns:
                for pattern in file_patterns:
                    files.extend(path.glob(pattern))
            else:
                files.extend(path.glob("*"))

        # Filter to only files
        files = [f for f in files if f.is_file()]

        self.logger.info(f"Found {len(files)} files in {directory_path}")

        # Load documents concurrently
        tasks = [self.load_document(str(f)) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failures and exceptions
        documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Failed to load {files[i]}: {result}")
            elif result is not None:
                documents.append(result)

        self.logger.info(f"Successfully loaded {len(documents)}/{len(files)} documents")

        return documents

    def detect_type(self, file_path: str) -> DocumentType:
        """
        Detect document type from file extension.

        Args:
            file_path: Path to file

        Returns:
            Detected document type
        """
        extension = Path(file_path).suffix.lower()

        # Code files
        if extension in {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go"}:
            return DocumentType.CODE

        # Markdown
        elif extension in {".md", ".markdown"}:
            return DocumentType.MARKDOWN

        # JSON
        elif extension == ".json":
            return DocumentType.JSON

        # YAML
        elif extension in {".yaml", ".yml"}:
            return DocumentType.YAML

        # RestructuredText
        elif extension == ".rst":
            return DocumentType.RESTRUCTURED_TEXT

        # Default to text
        else:
            return DocumentType.TEXT

    async def _load_text_file(self, file_path: str) -> Optional[Document]:
        """
        Load a plain text file.

        Args:
            file_path: Path to text file

        Returns:
            Document with text content
        """
        try:
            import uuid

            path = Path(file_path)

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            metadata = {
                "file_path": str(path),
                "file_name": path.name,
                "extension": path.suffix,
                "line_count": len(content.splitlines()),
            }

            document = Document(
                id=str(uuid.uuid4()),
                content=content,
                type=self.detect_type(str(path)),
                source=str(path),
                metadata=metadata,
            )

            self.logger.info(f"Loaded text file: {path}")
            return document

        except Exception as e:
            self.logger.error(f"Failed to load text file {file_path}: {e}")
            return None

    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.

        Returns:
            List of extensions (with dots)
        """
        return list(self.extension_map.keys())
