"""'What do you know about X?' query handler."""

import logging
import re
from typing import List, Optional, Tuple

from ..knowledge import KnowledgeStore, KnowledgeVerifier
from ..models import (
    KnowledgeQuery,
    KnowledgeResponse,
    ConfidenceLevel,
    Symbol,
)

logger = logging.getLogger(__name__)


class KnowledgeQueryHandler:
    """
    Handle 'What do you know about X?' queries.

    PATTERN: Natural language query parsing with confidence
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        verifier: Optional[KnowledgeVerifier] = None,
    ):
        """
        Initialize handler.

        Args:
            knowledge_store: Knowledge store to query
            verifier: Optional verifier for confidence
        """
        self.store = knowledge_store
        self.verifier = verifier or KnowledgeVerifier(knowledge_store)

    async def query(self, query: KnowledgeQuery) -> KnowledgeResponse:
        """
        Handle knowledge query.

        Args:
            query: KnowledgeQuery object

        Returns:
            KnowledgeResponse with answer and confidence
        """
        query_text = query.query.lower().strip()

        # Determine query type
        query_type = query.query_type
        if query_type == "auto":
            query_type = self._detect_query_type(query_text)

        # Handle different query types
        if query_type == "symbol":
            return await self._query_symbol(query)
        elif query_type == "file":
            return await self._query_file(query)
        elif query_type == "concept":
            return await self._query_concept(query)
        elif query_type == "relationship":
            return await self._query_relationship(query)
        else:
            return await self._query_general(query)

    def _detect_query_type(self, query: str) -> str:
        """Detect query type from text."""
        # File patterns
        if re.search(r"\.(py|js|ts|tsx|jsx|java|go)\b", query):
            return "file"

        # Symbol patterns
        if re.search(r"function|method|class|def\s+\w+", query):
            return "symbol"

        # Qualified names
        if re.search(r"\w+\.\w+", query):
            return "symbol"

        # Relationship patterns
        if any(word in query for word in ["calls", "uses", "depends", "imports"]):
            return "relationship"

        return "concept"

    async def _query_symbol(self, query: KnowledgeQuery) -> KnowledgeResponse:
        """Query for symbol information."""
        query_text = query.query

        # Extract symbol name
        symbol_name = self._extract_symbol_name(query_text)
        if not symbol_name:
            return KnowledgeResponse(
                query=query.query,
                answer="Could not identify a symbol name in your query.",
                confidence=ConfidenceLevel.UNKNOWN,
            )

        # Look up symbol
        symbol = await self.store.get_symbol(symbol_name)

        if symbol:
            answer = self._format_symbol_info(symbol)
            return KnowledgeResponse(
                query=query.query,
                answer=answer,
                confidence=symbol.confidence,
                sources=[f"{symbol.file_path}:{symbol.line_start}"],
                verified_claims=[f"{symbol.kind.value} '{symbol.name}' exists"],
            )

        # Try fuzzy search
        similar = await self.store.find_similar_symbols(symbol_name, limit=3)
        if similar:
            suggestions = ", ".join(s.name for s in similar)
            return KnowledgeResponse(
                query=query.query,
                answer=f"Symbol '{symbol_name}' not found. Did you mean: {suggestions}?",
                confidence=ConfidenceLevel.VERIFIED,
                knowledge_gaps=[f"No symbol named '{symbol_name}' in indexed codebase"],
                suggested_investigation=[f"Search for similar: {suggestions}"],
            )

        return KnowledgeResponse(
            query=query.query,
            answer=f"I don't have information about '{symbol_name}'.",
            confidence=ConfidenceLevel.UNKNOWN,
            knowledge_gaps=[f"Symbol '{symbol_name}' not found"],
        )

    async def _query_file(self, query: KnowledgeQuery) -> KnowledgeResponse:
        """Query for file information."""
        # Extract file path
        file_match = re.search(r"[\w/\\.-]+\.(py|js|ts|tsx|jsx|java|go)", query.query)
        if not file_match:
            return KnowledgeResponse(
                query=query.query,
                answer="Could not identify a file path in your query.",
                confidence=ConfidenceLevel.UNKNOWN,
            )

        file_path = file_match.group()
        file_info = await self.store.get_file(file_path)

        if not file_info:
            # Try partial match
            similar = await self.store.find_similar_files(file_path, limit=3)
            if similar:
                return KnowledgeResponse(
                    query=query.query,
                    answer=f"File '{file_path}' not found. Similar files: {', '.join(similar)}",
                    confidence=ConfidenceLevel.VERIFIED,
                    suggested_investigation=similar,
                )

            return KnowledgeResponse(
                query=query.query,
                answer=f"I don't have information about file '{file_path}'.",
                confidence=ConfidenceLevel.UNKNOWN,
            )

        # Build file info response
        answer_parts = [
            f"**File**: `{file_info.relative_path}`",
            f"- **Language**: {file_info.language}",
            f"- **Lines**: {file_info.line_count}",
            f"- **Size**: {file_info.size_bytes:,} bytes",
            f"- **Symbols**: {len(file_info.symbols)}",
        ]

        # List symbols in file
        if file_info.symbols:
            all_symbols = self.store.get_all_symbols()
            symbols_in_file = [
                all_symbols[sid] for sid in file_info.symbols
                if sid in all_symbols
            ]
            if symbols_in_file:
                answer_parts.append("\n**Symbols**:")
                for sym in symbols_in_file[:10]:
                    answer_parts.append(f"- `{sym.name}` ({sym.kind.value})")
                if len(symbols_in_file) > 10:
                    answer_parts.append(f"... and {len(symbols_in_file) - 10} more")

        return KnowledgeResponse(
            query=query.query,
            answer="\n".join(answer_parts),
            confidence=file_info.confidence,
            sources=[file_info.path],
        )

    async def _query_concept(self, query: KnowledgeQuery) -> KnowledgeResponse:
        """Query for concept/semantic search."""
        # Use semantic search
        results = await self.store.semantic_search(
            query.query,
            limit=query.max_results,
        )

        if not results:
            return KnowledgeResponse(
                query=query.query,
                answer="I couldn't find relevant code for that concept.",
                confidence=ConfidenceLevel.UNKNOWN,
                suggested_investigation=["Try more specific keywords", "Check spelling"],
            )

        answer_parts = [f"Found {len(results)} relevant items:\n"]

        for symbol, score in results:
            answer_parts.append(
                f"- **{symbol.qualified_name}** ({symbol.kind.value}) "
                f"- similarity: {score:.2f}"
            )
            if symbol.docstring:
                doc_preview = symbol.docstring[:100].replace("\n", " ")
                answer_parts.append(f"  {doc_preview}...")

        # Calculate aggregate confidence
        min_confidence = min(s.confidence for s, _ in results)

        return KnowledgeResponse(
            query=query.query,
            answer="\n".join(answer_parts),
            confidence=min_confidence,
            sources=[s.file_path for s, _ in results],
        )

    async def _query_relationship(self, query: KnowledgeQuery) -> KnowledgeResponse:
        """Query for code relationships."""
        # Extract symbol names
        names = re.findall(r"\b([A-Z][a-zA-Z0-9_]*|[a-z_][a-z0-9_]*)\b", query.query)

        if not names:
            return KnowledgeResponse(
                query=query.query,
                answer="Could not identify symbols to find relationships for.",
                confidence=ConfidenceLevel.UNKNOWN,
            )

        # Find symbols
        found_symbols = []
        for name in names[:5]:  # Limit to 5
            symbol = await self.store.get_symbol(name)
            if symbol:
                found_symbols.append(symbol)

        if not found_symbols:
            return KnowledgeResponse(
                query=query.query,
                answer=f"Could not find symbols: {', '.join(names[:5])}",
                confidence=ConfidenceLevel.UNKNOWN,
            )

        # Build relationship info
        answer_parts = ["**Relationships**:\n"]

        for symbol in found_symbols:
            answer_parts.append(f"\n**{symbol.qualified_name}**:")

            if symbol.calls:
                answer_parts.append(f"- Calls: {', '.join(symbol.calls[:5])}")
            if symbol.called_by:
                answer_parts.append(f"- Called by: {', '.join(symbol.called_by[:5])}")
            if symbol.imports:
                answer_parts.append(f"- Imports: {', '.join(symbol.imports[:5])}")
            if symbol.parent_symbol:
                answer_parts.append(f"- Parent: {symbol.parent_symbol}")

        return KnowledgeResponse(
            query=query.query,
            answer="\n".join(answer_parts),
            confidence=min(s.confidence for s in found_symbols),
            sources=[s.file_path for s in found_symbols],
        )

    async def _query_general(self, query: KnowledgeQuery) -> KnowledgeResponse:
        """Handle general queries."""
        # Try semantic search as fallback
        return await self._query_concept(query)

    def _extract_symbol_name(self, query: str) -> Optional[str]:
        """Extract symbol name from query text."""
        # Look for quoted names
        quoted = re.search(r"[`'\"](\w+(?:\.\w+)*)[`'\"]", query)
        if quoted:
            return quoted.group(1)

        # Look for qualified names
        qualified = re.search(r"\b(\w+(?:\.\w+)+)\b", query)
        if qualified:
            return qualified.group(1)

        # Look for function/class patterns
        func_pattern = re.search(
            r"(?:function|method|class|def)\s+(\w+)", query.lower()
        )
        if func_pattern:
            return func_pattern.group(1)

        # Last resort: extract likely symbol names
        words = re.findall(r"\b([A-Z][a-zA-Z0-9_]*|[a-z_][a-z0-9_]{2,})\b", query)
        # Filter out common words
        stopwords = {
            "the", "about", "what", "how", "where", "function", "class",
            "method", "file", "know", "tell", "me", "does", "is", "are",
        }
        candidates = [w for w in words if w.lower() not in stopwords]

        return candidates[0] if candidates else None

    def _format_symbol_info(self, symbol: Symbol) -> str:
        """Format symbol information for response."""
        lines = [
            f"**{symbol.kind.value}**: `{symbol.qualified_name}`",
            f"- **File**: `{symbol.file_path}:{symbol.line_start}`",
        ]

        if symbol.signature:
            lines.append(f"- **Signature**: `{symbol.signature}`")

        if symbol.docstring:
            lines.append(f"- **Documentation**: {symbol.docstring[:300]}")

        if symbol.parameters:
            params = ", ".join(p["name"] for p in symbol.parameters)
            lines.append(f"- **Parameters**: {params}")

        if symbol.return_type:
            lines.append(f"- **Returns**: {symbol.return_type}")

        lines.append(f"- **Confidence**: {symbol.confidence.value}")

        return "\n".join(lines)
