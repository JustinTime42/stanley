"""Knowledge storage with vector embeddings."""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..models import (
    Symbol,
    FileInfo,
    CodebaseUnderstanding,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """
    Knowledge storage integrating with Memory Service.

    PATTERN: Facade for symbol storage and retrieval
    CRITICAL: Support fast lookup by qualified name
    """

    def __init__(
        self,
        memory_service: Optional[Any] = None,
        persistence_path: Optional[str] = None,
    ):
        """
        Initialize knowledge store.

        Args:
            memory_service: Optional Memory Orchestrator for vector storage
            persistence_path: Path to persist knowledge (e.g., .agent-swarm/understanding/)
        """
        self.memory_service = memory_service
        self.persistence_path = Path(persistence_path) if persistence_path else None

        # In-memory caches
        self._symbols: Dict[str, Symbol] = {}
        self._files: Dict[str, FileInfo] = {}
        self._symbol_by_name: Dict[str, str] = {}  # name -> symbol_id
        self._symbol_by_qualified_name: Dict[str, str] = {}  # qualified_name -> symbol_id

    async def store_understanding(
        self,
        understanding: CodebaseUnderstanding,
    ) -> None:
        """
        Store complete codebase understanding.

        Args:
            understanding: CodebaseUnderstanding to store
        """
        # Store in memory
        self._symbols = understanding.symbols.copy()
        self._files = understanding.files.copy()

        # Build name indexes
        self._symbol_by_name.clear()
        self._symbol_by_qualified_name.clear()

        for symbol_id, symbol in self._symbols.items():
            self._symbol_by_name[symbol.name] = symbol_id
            self._symbol_by_qualified_name[symbol.qualified_name] = symbol_id

        # Store in vector database if available
        if self.memory_service:
            await self._store_in_vector_db(understanding)

        # Persist to disk if path configured
        if self.persistence_path:
            await self._persist_to_disk(understanding)

        logger.info(
            f"Stored understanding: {len(self._symbols)} symbols, {len(self._files)} files"
        )

    async def _store_in_vector_db(
        self,
        understanding: CodebaseUnderstanding,
    ) -> None:
        """Store symbols in vector database for semantic search."""
        try:
            from ...models.memory_models import MemoryType

            # Store each symbol with its embedding
            for symbol in understanding.symbols.values():
                # Create content for embedding
                content = self._symbol_to_content(symbol)

                await self.memory_service.store_memory(
                    content=content,
                    agent_id="codebase_analyzer",
                    memory_type=MemoryType.PROJECT,
                    metadata={
                        "type": "symbol",
                        "symbol_id": symbol.id,
                        "symbol_kind": symbol.kind.value,
                        "qualified_name": symbol.qualified_name,
                        "file_path": symbol.file_path,
                        "line_start": symbol.line_start,
                    },
                    tags=["symbol", symbol.kind.value],
                )

        except Exception as e:
            logger.warning(f"Failed to store in vector DB: {e}")

    def _symbol_to_content(self, symbol: Symbol) -> str:
        """Convert symbol to searchable content."""
        parts = [
            f"{symbol.kind.value}: {symbol.qualified_name}",
        ]

        if symbol.signature:
            parts.append(f"Signature: {symbol.signature}")

        if symbol.docstring:
            parts.append(f"Documentation: {symbol.docstring}")

        if symbol.description:
            parts.append(f"Description: {symbol.description}")

        parts.append(f"File: {symbol.file_path}:{symbol.line_start}")

        return "\n".join(parts)

    async def _persist_to_disk(
        self,
        understanding: CodebaseUnderstanding,
    ) -> None:
        """Persist understanding to disk."""
        self.persistence_path.mkdir(parents=True, exist_ok=True)

        # Save symbols
        symbols_path = self.persistence_path / "symbols.json"
        symbols_data = {
            sid: s.model_dump(mode="json") for sid, s in understanding.symbols.items()
        }
        with open(symbols_path, "w") as f:
            json.dump(symbols_data, f, indent=2, default=str)

        # Save structure
        structure_path = self.persistence_path / "structure.json"
        with open(structure_path, "w") as f:
            json.dump(understanding.structure.model_dump(mode="json"), f, indent=2)

        # Save dependencies
        deps_path = self.persistence_path / "dependencies.json"
        with open(deps_path, "w") as f:
            json.dump(understanding.dependency_graph.model_dump(mode="json"), f, indent=2)

        # Save knowledge index
        index_path = self.persistence_path / "knowledge_index.json"
        index_data = {
            "project_id": understanding.project_id,
            "root_path": understanding.root_path,
            "symbol_count": len(understanding.symbols),
            "file_count": len(understanding.files),
            "created_at": understanding.created_at.isoformat(),
            "updated_at": understanding.updated_at.isoformat(),
            "analysis_mode": understanding.analysis_mode.value,
        }
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

        # Save human-readable summary
        summary_path = self.persistence_path / "project_summary.md"
        summary = self._generate_summary(understanding)
        with open(summary_path, "w") as f:
            f.write(summary)

        logger.info(f"Persisted understanding to {self.persistence_path}")

    def _generate_summary(self, understanding: CodebaseUnderstanding) -> str:
        """Generate human-readable project summary."""
        structure = understanding.structure

        lines = [
            f"# {structure.project_name}",
            "",
            f"**Type:** {structure.detected_type}",
        ]

        if structure.detected_framework:
            lines.append(f"**Framework:** {structure.detected_framework}")

        lines.extend([
            "",
            "## Statistics",
            f"- **Files:** {structure.total_files}",
            f"- **Lines of Code:** {structure.total_lines:,}",
            f"- **Symbols:** {len(understanding.symbols)}",
            "",
            "### Files by Language",
        ])

        for lang, count in sorted(
            structure.files_by_language.items(), key=lambda x: -x[1]
        ):
            lines.append(f"- {lang}: {count}")

        if structure.source_directories:
            lines.extend([
                "",
                "## Source Directories",
            ])
            for d in structure.source_directories:
                lines.append(f"- `{d}`")

        if structure.entry_points:
            lines.extend([
                "",
                "## Entry Points",
            ])
            for ep in structure.entry_points:
                lines.append(f"- `{ep}`")

        lines.extend([
            "",
            "---",
            f"*Generated: {datetime.now().isoformat()}*",
        ])

        return "\n".join(lines)

    async def load_from_disk(self) -> Optional[CodebaseUnderstanding]:
        """Load understanding from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return None

        try:
            # Load index
            index_path = self.persistence_path / "knowledge_index.json"
            if not index_path.exists():
                return None

            with open(index_path, "r") as f:
                index_data = json.load(f)

            # Load symbols
            symbols_path = self.persistence_path / "symbols.json"
            if symbols_path.exists():
                with open(symbols_path, "r") as f:
                    symbols_data = json.load(f)
                    self._symbols = {
                        sid: Symbol.model_validate(data)
                        for sid, data in symbols_data.items()
                    }

            # Load structure
            structure_path = self.persistence_path / "structure.json"
            if structure_path.exists():
                with open(structure_path, "r") as f:
                    from ..models import ProjectStructure
                    structure = ProjectStructure.model_validate(json.load(f))
            else:
                structure = None

            # Load dependencies
            deps_path = self.persistence_path / "dependencies.json"
            if deps_path.exists():
                with open(deps_path, "r") as f:
                    from ..models import DependencyGraph
                    deps = DependencyGraph.model_validate(json.load(f))
            else:
                deps = None

            # Build indexes
            for symbol_id, symbol in self._symbols.items():
                self._symbol_by_name[symbol.name] = symbol_id
                self._symbol_by_qualified_name[symbol.qualified_name] = symbol_id

            logger.info(f"Loaded understanding from {self.persistence_path}")

            # Return understanding if we have structure
            if structure:
                from ..models import AnalysisMode
                return CodebaseUnderstanding(
                    project_id=index_data["project_id"],
                    root_path=index_data["root_path"],
                    structure=structure,
                    dependency_graph=deps or DependencyGraph(),
                    symbols=self._symbols,
                    analysis_mode=AnalysisMode(index_data.get("analysis_mode", "deep")),
                )

            return None

        except Exception as e:
            logger.error(f"Failed to load understanding: {e}")
            return None

    async def get_symbol(self, name: str) -> Optional[Symbol]:
        """
        Get symbol by name or qualified name.

        Args:
            name: Symbol name or qualified name

        Returns:
            Symbol if found
        """
        # Try qualified name first
        symbol_id = self._symbol_by_qualified_name.get(name)
        if symbol_id:
            return self._symbols.get(symbol_id)

        # Try simple name
        symbol_id = self._symbol_by_name.get(name)
        if symbol_id:
            return self._symbols.get(symbol_id)

        return None

    async def get_file(self, file_path: str) -> Optional[FileInfo]:
        """Get file info by path."""
        return self._files.get(file_path)

    async def find_similar_symbols(
        self,
        name: str,
        kind: Optional[str] = None,
        limit: int = 5,
    ) -> List[Symbol]:
        """
        Find symbols with similar names.

        Args:
            name: Name to search for
            kind: Optional symbol kind filter
            limit: Maximum results

        Returns:
            List of similar symbols
        """
        results = []
        name_lower = name.lower()

        for symbol in self._symbols.values():
            if kind and symbol.kind.value != kind:
                continue

            # Check for substring match
            if name_lower in symbol.name.lower():
                results.append(symbol)
            elif name_lower in symbol.qualified_name.lower():
                results.append(symbol)

            if len(results) >= limit:
                break

        return results

    async def find_similar_files(
        self,
        file_path: str,
        limit: int = 5,
    ) -> List[str]:
        """Find files with similar paths."""
        results = []
        name = Path(file_path).name.lower()

        for path in self._files.keys():
            if name in Path(path).name.lower():
                results.append(path)

            if len(results) >= limit:
                break

        return results

    async def semantic_search(
        self,
        query: str,
        kind: Optional[str] = None,
        limit: int = 5,
    ) -> List[Tuple[Symbol, float]]:
        """
        Semantic search for symbols.

        Args:
            query: Search query
            kind: Optional symbol kind filter
            limit: Maximum results

        Returns:
            List of (symbol, score) tuples
        """
        if not self.memory_service:
            # Fallback to text search
            return [(s, 0.5) for s in await self.find_similar_symbols(query, kind, limit)]

        try:
            results = await self.memory_service.retrieve_relevant_memories(
                query=query,
                k=limit * 2,  # Get more to filter by kind
            )

            symbol_results = []
            for result in results:
                metadata = result.metadata or {}
                if metadata.get("type") != "symbol":
                    continue

                if kind and metadata.get("symbol_kind") != kind:
                    continue

                symbol_id = metadata.get("symbol_id")
                if symbol_id and symbol_id in self._symbols:
                    symbol_results.append((self._symbols[symbol_id], result.score))

                if len(symbol_results) >= limit:
                    break

            return symbol_results

        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return []

    async def find_by_signature_similarity(
        self,
        signature: str,
        kind: Optional[str] = None,
        limit: int = 5,
    ) -> List[Symbol]:
        """Find symbols with similar signatures."""
        results = []
        sig_parts = set(signature.replace("(", " ").replace(")", " ").replace(",", " ").split())

        for symbol in self._symbols.values():
            if kind and symbol.kind.value != kind:
                continue

            if not symbol.signature:
                continue

            # Count matching parts
            sym_parts = set(
                symbol.signature.replace("(", " ").replace(")", " ").replace(",", " ").split()
            )
            overlap = len(sig_parts & sym_parts)

            if overlap > 1:
                results.append((symbol, overlap))

        # Sort by overlap and return
        results.sort(key=lambda x: -x[1])
        return [s for s, _ in results[:limit]]

    async def update_symbols(
        self,
        symbols: Dict[str, Symbol],
        changed_files: List[str],
    ) -> None:
        """
        Update symbols from changed files.

        Args:
            symbols: New/updated symbols
            changed_files: Files that changed
        """
        # Remove old symbols from changed files
        for file_path in changed_files:
            old_symbol_ids = [
                sid for sid, s in self._symbols.items() if s.file_path == file_path
            ]
            for sid in old_symbol_ids:
                symbol = self._symbols.pop(sid, None)
                if symbol:
                    self._symbol_by_name.pop(symbol.name, None)
                    self._symbol_by_qualified_name.pop(symbol.qualified_name, None)

        # Add new symbols
        for symbol_id, symbol in symbols.items():
            self._symbols[symbol_id] = symbol
            self._symbol_by_name[symbol.name] = symbol_id
            self._symbol_by_qualified_name[symbol.qualified_name] = symbol_id

        logger.info(f"Updated {len(symbols)} symbols from {len(changed_files)} files")

    def get_all_symbols(self) -> Dict[str, Symbol]:
        """Get all stored symbols."""
        return self._symbols.copy()

    def get_symbol_count(self) -> int:
        """Get total symbol count."""
        return len(self._symbols)
