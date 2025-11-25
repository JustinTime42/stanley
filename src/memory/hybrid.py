"""Hybrid search combining vector and keyword search with RRF."""

import logging
from typing import List, Dict, Any
from collections import defaultdict
import re

from ..models.memory_models import MemorySearchRequest, MemorySearchResult, MemoryItem

logger = logging.getLogger(__name__)


class HybridSearchManager:
    """
    Hybrid search manager combining vector and keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine results from
    vector similarity search and keyword/BM25 search.
    """

    def __init__(
        self,
        vector_search_fn,
        keyword_search_fn=None,
        k: int = 60,
    ):
        """
        Initialize hybrid search manager.

        Args:
            vector_search_fn: Async function for vector search
            keyword_search_fn: Optional async function for keyword search
            k: RRF constant (default 60, standard value)
        """
        self.vector_search_fn = vector_search_fn
        self.keyword_search_fn = keyword_search_fn
        self.rrf_k = k

    async def hybrid_search(
        self,
        request: MemorySearchRequest,
        **kwargs: Any,
    ) -> List[MemorySearchResult]:
        """
        Perform hybrid search combining vector and keyword results.

        Args:
            request: Search request
            **kwargs: Additional parameters

        Returns:
            Combined and re-ranked search results
        """
        # If alpha=0.0 (pure keyword), use keyword search only
        if request.alpha == 0.0 and self.keyword_search_fn is not None:
            keyword_results = await self.keyword_search_fn(request, **kwargs)
            return keyword_results

        # Get vector search results
        vector_results = await self.vector_search_fn(request, **kwargs)

        # If no keyword search function or alpha=1.0 (pure vector), return vector results
        if self.keyword_search_fn is None or request.alpha >= 1.0:
            return vector_results

        # Perform keyword search
        keyword_results = await self.keyword_search_fn(request, **kwargs)

        # Combine using RRF
        combined_results = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            alpha=request.alpha,
        )

        # Apply score threshold and limit
        filtered_results = [
            r for r in combined_results if r.score >= request.score_threshold
        ]

        return filtered_results[: request.k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[MemorySearchResult],
        keyword_results: List[MemorySearchResult],
        alpha: float = 0.7,
    ) -> List[MemorySearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.

        RRF formula: score = 1 / (k + rank)
        Final score: alpha * vector_score + (1-alpha) * keyword_score

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            alpha: Weight for vector search (0=keyword only, 1=vector only)

        Returns:
            Combined and re-ranked results
        """
        # Calculate RRF scores for each result set
        memory_scores: Dict[str, float] = defaultdict(float)
        memory_objects: Dict[str, MemorySearchResult] = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            rrf_score = 1.0 / (self.rrf_k + rank)
            memory_scores[result.memory.id] += alpha * rrf_score
            memory_objects[result.memory.id] = result

        # Process keyword results
        for rank, result in enumerate(keyword_results, start=1):
            rrf_score = 1.0 / (self.rrf_k + rank)
            memory_scores[result.memory.id] += (1.0 - alpha) * rrf_score
            if result.memory.id not in memory_objects:
                memory_objects[result.memory.id] = result

        # Create combined results
        combined_results = []
        for memory_id, score in memory_scores.items():
            result = memory_objects[memory_id]
            combined_results.append(
                MemorySearchResult(
                    memory=result.memory,
                    score=score,
                    source="hybrid",
                    highlights=result.highlights,
                )
            )

        # Sort by combined score descending
        combined_results.sort(key=lambda x: x.score, reverse=True)

        return combined_results


class BM25KeywordSearch:
    """
    Simple BM25-like keyword search implementation.

    This is a simplified version for in-memory keyword search.
    For production, consider using a dedicated search engine.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 search.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b

    async def search(
        self,
        query: str,
        documents: List[MemoryItem],
        k: int = 5,
    ) -> List[MemorySearchResult]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            documents: List of memory items to search
            k: Number of results

        Returns:
            Ranked search results
        """
        # Tokenize query
        query_terms = self._tokenize(query)

        # Calculate document stats
        avg_doc_length = (
            sum(len(self._tokenize(doc.content)) for doc in documents) / len(documents)
            if documents
            else 0
        )

        # Score each document
        scores = []
        for doc in documents:
            score = self._calculate_bm25_score(
                query_terms=query_terms,
                document=doc.content,
                avg_doc_length=avg_doc_length,
                total_docs=len(documents),
                documents=documents,
            )
            if score > 0:
                scores.append((doc, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Convert to MemorySearchResult
        results = []
        for doc, score in scores[:k]:
            results.append(
                MemorySearchResult(
                    memory=doc,
                    score=score,
                    source="keyword",
                    highlights=self._extract_highlights(query_terms, doc.content),
                )
            )

        return results

    def _calculate_bm25_score(
        self,
        query_terms: List[str],
        document: str,
        avg_doc_length: float,
        total_docs: int,
        documents: List[MemoryItem],
    ) -> float:
        """Calculate BM25 score for a document."""
        doc_terms = self._tokenize(document)
        doc_length = len(doc_terms)

        if doc_length == 0:
            return 0.0

        score = 0.0
        for term in query_terms:
            # Term frequency in document
            tf = doc_terms.count(term)
            if tf == 0:
                continue

            # Document frequency (how many docs contain the term)
            df = sum(1 for doc in documents if term in self._tokenize(doc.content))

            # IDF calculation
            idf = self._calculate_idf(df, total_docs)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / avg_doc_length)
            )

            score += idf * (numerator / denominator)

        return score

    def _calculate_idf(self, df: int, total_docs: int) -> float:
        """Calculate Inverse Document Frequency."""
        import math

        return math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _extract_highlights(
        self,
        query_terms: List[str],
        content: str,
        context_words: int = 5,
    ) -> List[str]:
        """Extract relevant excerpts containing query terms."""
        highlights = []
        content_lower = content.lower()

        for term in query_terms:
            # Find occurrences of term
            start = 0
            while True:
                idx = content_lower.find(term, start)
                if idx == -1:
                    break

                # Extract context around term
                words = content.split()
                word_idx = len(content[:idx].split())

                context_start = max(0, word_idx - context_words)
                context_end = min(len(words), word_idx + context_words + 1)

                excerpt = " ".join(words[context_start:context_end])
                if excerpt and excerpt not in highlights:
                    highlights.append(f"...{excerpt}...")

                start = idx + len(term)

                if len(highlights) >= 3:  # Limit highlights
                    break

            if len(highlights) >= 3:
                break

        return highlights
