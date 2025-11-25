"""Find similar code functionality."""

import logging
from typing import List, Optional, Tuple

from ..knowledge import KnowledgeStore
from ..models import Symbol, ConfidenceLevel

logger = logging.getLogger(__name__)


class SimilaritySearchHandler:
    """
    Find similar code functionality.

    PATTERN: Vector similarity search for code
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
    ):
        """
        Initialize handler.

        Args:
            knowledge_store: Knowledge store to search
        """
        self.store = knowledge_store

    async def find_similar_functions(
        self,
        description: str,
        limit: int = 5,
    ) -> List[Tuple[Symbol, float]]:
        """
        Find functions with similar purpose.

        Args:
            description: Description of desired functionality
            limit: Maximum results

        Returns:
            List of (symbol, similarity_score) tuples
        """
        return await self.store.semantic_search(
            description,
            kind="function",
            limit=limit,
        )

    async def find_similar_classes(
        self,
        description: str,
        limit: int = 5,
    ) -> List[Tuple[Symbol, float]]:
        """
        Find classes with similar purpose.

        Args:
            description: Description of desired class
            limit: Maximum results

        Returns:
            List of (symbol, similarity_score) tuples
        """
        return await self.store.semantic_search(
            description,
            kind="class",
            limit=limit,
        )

    async def find_by_signature(
        self,
        signature: str,
        limit: int = 5,
    ) -> List[Symbol]:
        """
        Find functions with similar signatures.

        Args:
            signature: Function signature to match
            limit: Maximum results

        Returns:
            List of matching symbols
        """
        return await self.store.find_by_signature_similarity(
            signature,
            kind="function",
            limit=limit,
        )

    async def find_implementations(
        self,
        interface_name: str,
        limit: int = 10,
    ) -> List[Symbol]:
        """
        Find implementations of an interface/base class.

        Args:
            interface_name: Name of interface or base class
            limit: Maximum results

        Returns:
            List of implementing symbols
        """
        # Search for classes that might implement/extend
        all_symbols = self.store.get_all_symbols()
        implementations = []

        for symbol in all_symbols.values():
            if symbol.kind.value != "class":
                continue

            # Check if parent matches
            if symbol.parent_symbol and interface_name.lower() in symbol.parent_symbol.lower():
                implementations.append(symbol)
                continue

            # Check docstring for "implements" or "extends"
            if symbol.docstring:
                doc_lower = symbol.docstring.lower()
                if interface_name.lower() in doc_lower and (
                    "implement" in doc_lower or "extend" in doc_lower
                ):
                    implementations.append(symbol)

        return implementations[:limit]

    async def find_callers(
        self,
        symbol_name: str,
        limit: int = 10,
    ) -> List[Symbol]:
        """
        Find symbols that call a given symbol.

        Args:
            symbol_name: Name of symbol to find callers for
            limit: Maximum results

        Returns:
            List of calling symbols
        """
        target = await self.store.get_symbol(symbol_name)
        if not target:
            return []

        callers = []
        all_symbols = self.store.get_all_symbols()

        for symbol in all_symbols.values():
            if target.id in symbol.calls or symbol_name in symbol.calls:
                callers.append(symbol)

            if len(callers) >= limit:
                break

        return callers

    async def find_callees(
        self,
        symbol_name: str,
        limit: int = 10,
    ) -> List[Symbol]:
        """
        Find symbols that are called by a given symbol.

        Args:
            symbol_name: Name of calling symbol
            limit: Maximum results

        Returns:
            List of called symbols
        """
        caller = await self.store.get_symbol(symbol_name)
        if not caller:
            return []

        callees = []
        for called_id in caller.calls[:limit]:
            callee = await self.store.get_symbol(called_id)
            if callee:
                callees.append(callee)

        return callees

    async def find_related(
        self,
        symbol_name: str,
        limit: int = 10,
    ) -> dict:
        """
        Find all related symbols.

        Args:
            symbol_name: Symbol to find relations for
            limit: Maximum results per category

        Returns:
            Dict with callers, callees, siblings, etc.
        """
        target = await self.store.get_symbol(symbol_name)
        if not target:
            return {"error": f"Symbol '{symbol_name}' not found"}

        result = {
            "symbol": target.qualified_name,
            "kind": target.kind.value,
            "callers": [],
            "callees": [],
            "siblings": [],
            "imports": [],
        }

        # Find callers
        result["callers"] = [
            s.qualified_name for s in await self.find_callers(symbol_name, limit)
        ]

        # Find callees
        result["callees"] = [
            s.qualified_name for s in await self.find_callees(symbol_name, limit)
        ]

        # Find siblings (same parent)
        if target.parent_symbol:
            all_symbols = self.store.get_all_symbols()
            siblings = [
                s.name for s in all_symbols.values()
                if s.parent_symbol == target.parent_symbol and s.id != target.id
            ]
            result["siblings"] = siblings[:limit]

        # Find imports used
        result["imports"] = target.imports[:limit]

        return result

    async def search_code(
        self,
        query: str,
        file_filter: Optional[str] = None,
        kind_filter: Optional[str] = None,
        min_confidence: ConfidenceLevel = ConfidenceLevel.UNCERTAIN,
        limit: int = 10,
    ) -> List[Tuple[Symbol, float]]:
        """
        General code search.

        Args:
            query: Search query
            file_filter: Optional file pattern filter
            kind_filter: Optional symbol kind filter
            min_confidence: Minimum confidence level
            limit: Maximum results

        Returns:
            List of (symbol, score) tuples
        """
        # Get semantic search results
        results = await self.store.semantic_search(query, kind=kind_filter, limit=limit * 2)

        # Filter by criteria
        filtered = []
        for symbol, score in results:
            # Check confidence
            if symbol.confidence < min_confidence:
                continue

            # Check file filter
            if file_filter and file_filter not in symbol.file_path:
                continue

            filtered.append((symbol, score))

            if len(filtered) >= limit:
                break

        return filtered
