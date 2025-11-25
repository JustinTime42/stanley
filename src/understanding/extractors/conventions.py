"""Coding convention detection."""

import logging
import re
from collections import Counter
from typing import Dict, List, Any

from .base import BaseExtractor
from ..models import FileInfo, Symbol

logger = logging.getLogger(__name__)


class ConventionExtractor(BaseExtractor[Dict[str, Any]]):
    """
    Detect coding conventions in the codebase.

    PATTERN: Statistical analysis of naming patterns
    """

    async def extract(
        self,
        symbols: Dict[str, Symbol],
        files: List[FileInfo],
    ) -> Dict[str, Any]:
        """
        Extract coding conventions from symbols.

        Args:
            symbols: Extracted symbols
            files: File information

        Returns:
            Dict of detected conventions
        """
        conventions = {
            "naming": self._detect_naming_convention(symbols),
            "file_naming": self._detect_file_naming(files),
            "indentation": await self._detect_indentation(files),
            "documentation": self._detect_documentation_style(symbols),
            "test_patterns": self._detect_test_patterns(symbols, files),
        }

        return conventions

    def _detect_naming_convention(self, symbols: Dict[str, Symbol]) -> str:
        """Detect primary naming convention."""
        patterns = {
            "snake_case": 0,
            "camelCase": 0,
            "PascalCase": 0,
            "kebab-case": 0,
            "SCREAMING_SNAKE_CASE": 0,
        }

        for symbol in symbols.values():
            name = symbol.name

            # Skip single character names
            if len(name) <= 1:
                continue

            # Check patterns
            if self._is_snake_case(name):
                patterns["snake_case"] += 1
            elif self._is_camel_case(name):
                patterns["camelCase"] += 1
            elif self._is_pascal_case(name):
                patterns["PascalCase"] += 1
            elif self._is_screaming_snake_case(name):
                patterns["SCREAMING_SNAKE_CASE"] += 1

        # Return most common pattern
        if not any(patterns.values()):
            return "unknown"

        return max(patterns, key=patterns.get)

    def _is_snake_case(self, name: str) -> bool:
        """Check if name is snake_case."""
        return bool(re.match(r"^[a-z][a-z0-9_]*$", name)) and "_" in name

    def _is_camel_case(self, name: str) -> bool:
        """Check if name is camelCase."""
        return bool(re.match(r"^[a-z][a-zA-Z0-9]*$", name)) and any(c.isupper() for c in name)

    def _is_pascal_case(self, name: str) -> bool:
        """Check if name is PascalCase."""
        return bool(re.match(r"^[A-Z][a-zA-Z0-9]*$", name))

    def _is_screaming_snake_case(self, name: str) -> bool:
        """Check if name is SCREAMING_SNAKE_CASE."""
        return bool(re.match(r"^[A-Z][A-Z0-9_]*$", name)) and "_" in name

    def _detect_file_naming(self, files: List[FileInfo]) -> str:
        """Detect file naming convention."""
        patterns = Counter()

        for file_info in files:
            name = file_info.relative_path.split("/")[-1]
            # Remove extension
            name = name.rsplit(".", 1)[0]

            if "-" in name:
                patterns["kebab-case"] += 1
            elif "_" in name:
                patterns["snake_case"] += 1
            elif any(c.isupper() for c in name):
                if name[0].isupper():
                    patterns["PascalCase"] += 1
                else:
                    patterns["camelCase"] += 1
            else:
                patterns["lowercase"] += 1

        if not patterns:
            return "unknown"

        return patterns.most_common(1)[0][0]

    async def _detect_indentation(self, files: List[FileInfo]) -> Dict[str, Any]:
        """Detect indentation style."""
        space_counts = Counter()
        tab_count = 0

        for file_info in files[:20]:  # Sample first 20 files
            content = self._read_file_content(file_info.path)
            if not content:
                continue

            for line in content.split("\n"):
                if not line.strip():
                    continue

                # Count leading whitespace
                stripped = line.lstrip()
                indent = line[: len(line) - len(stripped)]

                if indent.startswith("\t"):
                    tab_count += 1
                elif indent.startswith(" "):
                    # Count spaces
                    space_count = len(indent)
                    if space_count > 0:
                        space_counts[space_count] += 1

        # Determine style
        if tab_count > sum(space_counts.values()):
            return {"style": "tabs", "size": None}

        if space_counts:
            # Find most common indentation
            most_common = space_counts.most_common(3)
            # Look for smallest common increment
            sizes = [s for s, _ in most_common]
            if sizes:
                # Find GCD of common sizes
                from math import gcd
                from functools import reduce

                indent_size = reduce(gcd, sizes)
                return {"style": "spaces", "size": indent_size}

        return {"style": "unknown", "size": None}

    def _detect_documentation_style(self, symbols: Dict[str, Symbol]) -> Dict[str, Any]:
        """Detect documentation style."""
        docstring_count = 0
        total_documented = 0
        styles = Counter()

        for symbol in symbols.values():
            if symbol.kind in ("function", "method", "class"):
                if symbol.docstring:
                    docstring_count += 1

                    # Detect docstring style
                    doc = symbol.docstring

                    if ":param" in doc or ":return" in doc:
                        styles["sphinx"] += 1
                    elif "Args:" in doc or "Returns:" in doc:
                        styles["google"] += 1
                    elif "@param" in doc or "@return" in doc:
                        styles["jsdoc"] += 1
                    else:
                        styles["plain"] += 1

                total_documented += 1

        coverage = docstring_count / total_documented if total_documented > 0 else 0
        style = styles.most_common(1)[0][0] if styles else "none"

        return {
            "style": style,
            "coverage": round(coverage, 2),
            "documented_count": docstring_count,
            "total_functions": total_documented,
        }

    def _detect_test_patterns(
        self,
        symbols: Dict[str, Symbol],
        files: List[FileInfo],
    ) -> Dict[str, Any]:
        """Detect test patterns."""
        test_patterns = {
            "prefix_test": 0,  # test_function
            "suffix_test": 0,  # function_test
            "prefix_Test": 0,  # Test class
            "describe_it": 0,  # describe/it blocks
            "test_classes": 0,  # TestCase classes
        }

        # Analyze symbols
        for symbol in symbols.values():
            name = symbol.name

            if name.startswith("test_"):
                test_patterns["prefix_test"] += 1
            elif name.endswith("_test"):
                test_patterns["suffix_test"] += 1
            elif name.startswith("Test"):
                test_patterns["prefix_Test"] += 1
            elif name == "describe" or name == "it" or name == "test":
                test_patterns["describe_it"] += 1

        # Analyze files for test frameworks
        test_files = [f for f in files if "test" in f.relative_path.lower()]

        for file_info in test_files[:10]:
            content = self._read_file_content(file_info.path)
            if not content:
                continue

            if "unittest.TestCase" in content or "TestCase" in content:
                test_patterns["test_classes"] += 1
            if "describe(" in content and "it(" in content:
                test_patterns["describe_it"] += 1

        # Determine primary pattern
        primary_pattern = max(test_patterns, key=test_patterns.get)

        return {
            "primary_pattern": primary_pattern if test_patterns[primary_pattern] > 0 else "unknown",
            "patterns": test_patterns,
            "test_file_count": len(test_files),
        }
