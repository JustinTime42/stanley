"""Documentation extraction."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import BaseExtractor
from ..models import FileInfo, Symbol

logger = logging.getLogger(__name__)


class DocumentationExtractor(BaseExtractor[Dict[str, List[str]]]):
    """
    Extract documentation from code.

    PATTERN: Extract docstrings, comments, README content
    """

    # Patterns for special comments
    TODO_PATTERN = re.compile(r"#\s*TODO[:\s]*(.+)$|//\s*TODO[:\s]*(.+)$", re.MULTILINE | re.IGNORECASE)
    FIXME_PATTERN = re.compile(r"#\s*FIXME[:\s]*(.+)$|//\s*FIXME[:\s]*(.+)$", re.MULTILINE | re.IGNORECASE)
    NOTE_PATTERN = re.compile(r"#\s*NOTE[:\s]*(.+)$|//\s*NOTE[:\s]*(.+)$", re.MULTILINE | re.IGNORECASE)
    HACK_PATTERN = re.compile(r"#\s*HACK[:\s]*(.+)$|//\s*HACK[:\s]*(.+)$", re.MULTILINE | re.IGNORECASE)

    async def extract(
        self,
        files: List[FileInfo],
        symbols: Dict[str, Symbol],
    ) -> Dict[str, List[str]]:
        """
        Extract documentation from codebase.

        Args:
            files: List of files to analyze
            symbols: Extracted symbols

        Returns:
            Dict with documentation categories
        """
        docs = {
            "readme": await self._extract_readme(files),
            "todos": [],
            "fixmes": [],
            "notes": [],
            "hacks": [],
            "api_docs": [],
            "inline_comments": [],
        }

        # Extract special comments from files
        for file_info in files:
            content = self._read_file_content(file_info.path)
            if not content:
                continue

            # Extract TODOs
            for match in self.TODO_PATTERN.finditer(content):
                text = match.group(1) or match.group(2)
                if text:
                    docs["todos"].append({
                        "text": text.strip(),
                        "file": file_info.relative_path,
                        "line": content[:match.start()].count("\n") + 1,
                    })

            # Extract FIXMEs
            for match in self.FIXME_PATTERN.finditer(content):
                text = match.group(1) or match.group(2)
                if text:
                    docs["fixmes"].append({
                        "text": text.strip(),
                        "file": file_info.relative_path,
                        "line": content[:match.start()].count("\n") + 1,
                    })

            # Extract NOTEs
            for match in self.NOTE_PATTERN.finditer(content):
                text = match.group(1) or match.group(2)
                if text:
                    docs["notes"].append({
                        "text": text.strip(),
                        "file": file_info.relative_path,
                        "line": content[:match.start()].count("\n") + 1,
                    })

            # Extract HACKs
            for match in self.HACK_PATTERN.finditer(content):
                text = match.group(1) or match.group(2)
                if text:
                    docs["hacks"].append({
                        "text": text.strip(),
                        "file": file_info.relative_path,
                        "line": content[:match.start()].count("\n") + 1,
                    })

        # Extract API documentation from docstrings
        for symbol in symbols.values():
            if symbol.docstring and symbol.kind in ("function", "method", "class"):
                docs["api_docs"].append({
                    "symbol": symbol.qualified_name,
                    "kind": symbol.kind.value,
                    "docstring": symbol.docstring,
                    "file": symbol.file_path,
                    "line": symbol.line_start,
                })

        return docs

    async def _extract_readme(self, files: List[FileInfo]) -> Optional[str]:
        """Extract README content."""
        readme_names = ["README.md", "README.rst", "README.txt", "README"]

        for file_info in files:
            file_name = Path(file_info.path).name
            if file_name in readme_names:
                content = self._read_file_content(file_info.path)
                if content:
                    return content

        return None

    async def extract_module_docs(
        self,
        file_path: str,
    ) -> Optional[str]:
        """Extract module-level docstring."""
        content = self._read_file_content(file_path)
        if not content:
            return None

        # Look for module docstring at start of file
        lines = content.split("\n")

        # Skip shebang and encoding declarations
        start_line = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#!") or stripped.startswith("# -*-"):
                start_line = i + 1
            else:
                break

        # Look for docstring
        remaining = "\n".join(lines[start_line:]).strip()

        # Triple-quoted string at start
        for quote in ['"""', "'''"]:
            if remaining.startswith(quote):
                end = remaining.find(quote, 3)
                if end != -1:
                    return remaining[3:end].strip()

        return None

    async def extract_comments(
        self,
        file_path: str,
        language: str,
    ) -> List[Tuple[int, str]]:
        """
        Extract all comments from a file.

        Returns:
            List of (line_number, comment_text) tuples
        """
        content = self._read_file_content(file_path)
        if not content:
            return []

        comments = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            line_num = i + 1
            stripped = line.strip()

            # Single-line comments
            if language == "python":
                if "#" in line:
                    # Find comment start (not in string)
                    comment_start = self._find_comment_start_python(line)
                    if comment_start != -1:
                        comment = line[comment_start + 1:].strip()
                        if comment:
                            comments.append((line_num, comment))

            elif language in ("javascript", "typescript", "java", "go"):
                if "//" in line:
                    comment_start = line.find("//")
                    comment = line[comment_start + 2:].strip()
                    if comment:
                        comments.append((line_num, comment))

        return comments

    def _find_comment_start_python(self, line: str) -> int:
        """Find start of Python comment, ignoring # in strings."""
        in_string = False
        string_char = None

        for i, char in enumerate(line):
            if char in ('"', "'") and (i == 0 or line[i - 1] != "\\"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            elif char == "#" and not in_string:
                return i

        return -1

    def get_documentation_coverage(
        self,
        symbols: Dict[str, Symbol],
    ) -> Dict[str, float]:
        """
        Calculate documentation coverage.

        Returns:
            Dict with coverage metrics
        """
        function_count = 0
        documented_functions = 0
        class_count = 0
        documented_classes = 0

        for symbol in symbols.values():
            if symbol.kind == "function" or symbol.kind == "method":
                function_count += 1
                if symbol.docstring:
                    documented_functions += 1
            elif symbol.kind == "class":
                class_count += 1
                if symbol.docstring:
                    documented_classes += 1

        return {
            "function_coverage": documented_functions / function_count if function_count > 0 else 0,
            "class_coverage": documented_classes / class_count if class_count > 0 else 0,
            "total_functions": function_count,
            "documented_functions": documented_functions,
            "total_classes": class_count,
            "documented_classes": documented_classes,
        }
