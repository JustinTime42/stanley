"""Markdown file loader with structure extraction."""

import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import uuid

from ....models.document_models import Document, DocumentType

logger = logging.getLogger(__name__)


class MarkdownLoader:
    """
    Loader for Markdown documents.

    PATTERN: Extract structure (headers, code blocks, lists)
    CRITICAL: Preserve document hierarchy
    """

    def __init__(self):
        """Initialize markdown loader."""
        self.logger = logger

    async def load_file(self, file_path: str) -> Optional[Document]:
        """
        Load a markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            Document with markdown content and structure
        """
        try:
            path = Path(file_path)

            # Read file content
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse structure
            metadata = self.parse_structure(content)
            metadata.update(
                {
                    "file_path": str(path),
                    "file_name": path.name,
                    "extension": path.suffix,
                }
            )

            # Create document
            document = Document(
                id=str(uuid.uuid4()),
                content=content,
                type=DocumentType.MARKDOWN,
                source=str(path),
                metadata=metadata,
            )

            self.logger.info(f"Loaded markdown file: {path}")
            return document

        except Exception as e:
            self.logger.error(f"Failed to load markdown file {file_path}: {e}")
            return None

    def parse_structure(self, content: str) -> Dict[str, Any]:
        """
        Parse markdown structure.

        Args:
            content: Markdown content

        Returns:
            Metadata with structure information
        """
        metadata = {
            "line_count": len(content.splitlines()),
        }

        # Extract headers
        headers = self._extract_headers(content)
        metadata["headers"] = headers
        metadata["header_count"] = len(headers)

        # Extract code blocks
        code_blocks = self._extract_code_blocks(content)
        metadata["code_blocks"] = len(code_blocks)
        metadata["code_languages"] = list(set(cb[0] for cb in code_blocks if cb[0]))

        # Extract links
        links = self._extract_links(content)
        metadata["link_count"] = len(links)

        # Build table of contents
        if headers:
            metadata["toc"] = self._build_toc(headers)

        return metadata

    def _extract_headers(self, content: str) -> List[Dict[str, Any]]:
        """Extract headers with levels."""
        headers = []
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        for match in header_pattern.finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            headers.append(
                {
                    "level": level,
                    "text": text,
                    "position": match.start(),
                }
            )

        return headers

    def _extract_code_blocks(self, content: str) -> List[Tuple[Optional[str], str]]:
        """
        Extract code blocks with language.

        Returns:
            List of (language, code) tuples
        """
        code_blocks = []

        # Fenced code blocks with language
        fenced_pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)

        for match in fenced_pattern.finditer(content):
            language = match.group(1)
            code = match.group(2)
            code_blocks.append((language, code))

        return code_blocks

    def _extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extract markdown links."""
        links = []
        link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

        for match in link_pattern.finditer(content):
            links.append(
                {
                    "text": match.group(1),
                    "url": match.group(2),
                }
            )

        return links

    def _build_toc(self, headers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build hierarchical table of contents.

        Args:
            headers: List of headers

        Returns:
            Hierarchical TOC structure
        """
        toc = []
        stack = []

        for header in headers:
            level = header["level"]
            text = header["text"]

            # Pop stack until we find parent level
            while stack and stack[-1]["level"] >= level:
                stack.pop()

            toc_entry = {
                "level": level,
                "text": text,
                "children": [],
            }

            if stack:
                # Add as child of current parent
                stack[-1]["children"].append(toc_entry)
            else:
                # Top level entry
                toc.append(toc_entry)

            stack.append(toc_entry)

        return toc

    def supports_file(self, file_path: str) -> bool:
        """
        Check if loader supports this file.

        Args:
            file_path: Path to file

        Returns:
            True if file is markdown
        """
        extension = Path(file_path).suffix.lower()
        return extension in {".md", ".markdown"}
