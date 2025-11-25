"""Dependency graph extraction."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Optional

from .base import BaseExtractor
from ..models import FileInfo, Symbol, DependencyGraph

logger = logging.getLogger(__name__)


class DependencyExtractor(BaseExtractor[DependencyGraph]):
    """
    Extract module dependency graph.

    PATTERN: Parse imports, build graph, detect cycles
    CRITICAL: Handle circular dependencies gracefully
    """

    def __init__(self):
        """Initialize dependency extractor."""
        super().__init__()

    async def extract(
        self,
        files: List[FileInfo],
        symbols: Dict[str, Symbol],
        file_infos: Dict[str, FileInfo],
    ) -> DependencyGraph:
        """
        Extract dependency graph from files.

        Args:
            files: List of files to analyze
            symbols: Extracted symbols
            file_infos: File information

        Returns:
            DependencyGraph with all dependencies
        """
        graph = DependencyGraph()

        # Build module nodes
        for file_info in files:
            module_name = self._path_to_module(file_info.relative_path)
            graph.add_node(
                module_name,
                file_path=file_info.path,
                relative_path=file_info.relative_path,
                language=file_info.language,
                symbol_count=len(file_info.symbols),
            )

        # Extract imports for each file
        for file_info in files:
            imports = await self._extract_file_imports(file_info)
            file_info.imports = imports

            module_name = self._path_to_module(file_info.relative_path)

            for imp in imports:
                # Determine if import is internal or external
                is_internal = self._is_internal_import(imp, files)

                if is_internal:
                    graph.add_edge(module_name, imp, "import")

        # Identify entry points (modules with no incoming edges)
        all_targets = {e["to"] for e in graph.edges}
        all_sources = {e["from"] for e in graph.edges}
        graph.entry_points = list(all_sources - all_targets)

        # Identify leaf modules (no outgoing edges)
        graph.leaf_modules = list(all_targets - all_sources)

        # Detect cycles
        graph.cycles = self._detect_cycles(graph)

        return graph

    async def _extract_file_imports(self, file_info: FileInfo) -> List[str]:
        """Extract import statements from a file."""
        content = self._read_file_content(file_info.path)
        if not content:
            return []

        if file_info.language == "python":
            return self._extract_python_imports(content)
        elif file_info.language in ("javascript", "typescript"):
            return self._extract_js_imports(content)

        return []

    def _extract_python_imports(self, content: str) -> List[str]:
        """Extract Python import statements."""
        imports = []

        # import module
        import_pattern = re.compile(r"^import\s+([\w.]+)", re.MULTILINE)
        for match in import_pattern.finditer(content):
            imports.append(match.group(1))

        # from module import ...
        from_pattern = re.compile(r"^from\s+([\w.]+)\s+import", re.MULTILINE)
        for match in from_pattern.finditer(content):
            imports.append(match.group(1))

        return imports

    def _extract_js_imports(self, content: str) -> List[str]:
        """Extract JavaScript/TypeScript import statements."""
        imports = []

        # import ... from 'module'
        import_pattern = re.compile(r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE)
        for match in import_pattern.finditer(content):
            imports.append(match.group(1))

        # require('module')
        require_pattern = re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
        for match in require_pattern.finditer(content):
            imports.append(match.group(1))

        # Dynamic import
        dynamic_pattern = re.compile(r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
        for match in dynamic_pattern.finditer(content):
            imports.append(match.group(1))

        return imports

    def _path_to_module(self, relative_path: str) -> str:
        """Convert file path to module name."""
        # Remove extension
        path = Path(relative_path)
        module_path = path.with_suffix("")

        # Convert path separators to dots
        parts = module_path.parts

        # Remove __init__ from Python packages
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]

        # Remove index from JS/TS
        if parts and parts[-1] == "index":
            parts = parts[:-1]

        return ".".join(parts)

    def _is_internal_import(self, import_name: str, files: List[FileInfo]) -> bool:
        """Check if import is internal to the project."""
        # Relative imports are internal
        if import_name.startswith("."):
            return True

        # Check if any file matches this module
        for file_info in files:
            module_name = self._path_to_module(file_info.relative_path)
            if module_name == import_name or module_name.startswith(f"{import_name}."):
                return True
            if import_name.startswith(f"{module_name}."):
                return True

        # Common external packages
        external_packages = {
            # Python
            "os",
            "sys",
            "re",
            "json",
            "logging",
            "typing",
            "pathlib",
            "datetime",
            "asyncio",
            "collections",
            "functools",
            "itertools",
            "dataclasses",
            "enum",
            "abc",
            "hashlib",
            "copy",
            "time",
            "uuid",
            "threading",
            "multiprocessing",
            # Common Python packages
            "pydantic",
            "fastapi",
            "flask",
            "django",
            "requests",
            "httpx",
            "aiohttp",
            "pytest",
            "numpy",
            "pandas",
            "redis",
            "qdrant_client",
            "openai",
            "anthropic",
            "langchain",
            "langgraph",
            "rich",
            "click",
            "watchdog",
            # JavaScript/TypeScript
            "react",
            "vue",
            "angular",
            "express",
            "next",
            "axios",
            "lodash",
            "moment",
            "dayjs",
        }

        # Get base package name
        base_package = import_name.split(".")[0].split("/")[0]
        if base_package in external_packages:
            return False

        # Node built-in modules
        node_builtins = {"fs", "path", "http", "https", "crypto", "url", "util", "stream", "events"}
        if base_package in node_builtins:
            return False

        # NPM scoped packages (@org/package)
        if import_name.startswith("@"):
            return False

        # Default to internal if not recognized
        return True

    def _detect_cycles(self, graph: DependencyGraph) -> List[List[str]]:
        """
        Detect cycles in dependency graph.

        PATTERN: Tarjan's algorithm for SCC detection
        CRITICAL: Handle cycles gracefully, track visited nodes
        """
        cycles = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for dep in graph.get_dependencies(node):
                if dep not in visited:
                    dfs(dep)
                elif dep in rec_stack:
                    # Found cycle
                    cycle_start = path.index(dep)
                    cycle = path[cycle_start:] + [dep]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        for node in graph.nodes:
            if node not in visited:
                dfs(node)

        return cycles

    async def get_symbol_dependencies(
        self,
        symbol: Symbol,
        symbols: Dict[str, Symbol],
    ) -> Dict[str, List[str]]:
        """
        Get dependencies at the symbol level.

        Args:
            symbol: Symbol to analyze
            symbols: All symbols

        Returns:
            Dict with 'calls', 'called_by', 'imports' lists
        """
        # This would require deeper AST analysis
        # For now, return existing relationships
        return {
            "calls": symbol.calls,
            "called_by": symbol.called_by,
            "imports": symbol.imports,
        }
