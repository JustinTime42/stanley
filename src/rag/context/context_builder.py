"""Context builder with optimization and U-shaped attention."""

import logging
from typing import List, Optional

from ...models.rag_models import RetrievalResult, ContextOptimization
from .window_optimizer import WindowOptimizer

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Build optimized context from retrieved chunks.

    PATTERN: U-shaped attention with relevance ordering
    CRITICAL: Most relevant content at beginning and end
    GOTCHA: "Lost in the Middle" problem - middle content often ignored
    """

    def __init__(self, window_optimizer: Optional[WindowOptimizer] = None):
        """
        Initialize context builder.

        Args:
            window_optimizer: Optional window optimizer (creates if None)
        """
        self.window_optimizer = window_optimizer or WindowOptimizer()
        self.logger = logger

    def build_context(
        self,
        results: List[RetrievalResult],
        max_tokens: int = 4000,
        optimization: Optional[ContextOptimization] = None,
    ) -> str:
        """
        Build optimized context from retrieval results.

        PATTERN: U-shaped relevance ordering for better attention
        CRITICAL: Most relevant chunks at start and end

        Args:
            results: Retrieved and scored chunks
            max_tokens: Maximum context tokens
            optimization: Optional optimization settings

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        if optimization is None:
            optimization = ContextOptimization()

        # Sort results by relevance
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)

        # Apply U-shaped ordering
        ordered_results = self._apply_u_shaped_ordering(
            sorted_results,
            optimization.relevance_ordering,
        )

        # Format chunks with metadata
        formatted_chunks = []
        for result in ordered_results:
            formatted = self._format_chunk(result, optimization.include_metadata)
            formatted_chunks.append(formatted)

        # Optimize to fit within token limit
        if optimization.optimization_strategy == "truncate":
            context_parts = self.window_optimizer.optimize_window(
                formatted_chunks,
                max_tokens,
                optimization.reserve_tokens,
            )
        elif optimization.optimization_strategy == "compress":
            context_parts = self._compress_context(
                formatted_chunks,
                max_tokens,
                optimization.reserve_tokens,
            )
        else:  # summarize
            # For now, fall back to truncate
            # TODO: Implement LLM-based summarization
            context_parts = self.window_optimizer.optimize_window(
                formatted_chunks,
                max_tokens,
                optimization.reserve_tokens,
            )

        # Join with separators
        context = "\n\n---\n\n".join(context_parts)

        self.logger.info(
            f"Built context from {len(results)} chunks "
            f"({self.window_optimizer.estimate_tokens(context)} tokens)"
        )

        return context

    def _apply_u_shaped_ordering(
        self,
        results: List[RetrievalResult],
        ordering: str,
    ) -> List[RetrievalResult]:
        """
        Apply U-shaped ordering to results.

        PATTERN: Most relevant at beginning, second-most at end, rest in middle
        REASON: LLM attention is strongest at start and end of context

        Args:
            results: Sorted results
            ordering: Ordering strategy

        Returns:
            Reordered results
        """
        if len(results) <= 2:
            return results

        if ordering == "score":
            # U-shaped ordering by relevance
            # Top third at beginning
            # Middle third in middle
            # Last third at end (for recency bias)

            third = len(results) // 3

            top_chunks = results[:third]  # Highest relevance
            middle_chunks = results[third : 2 * third]  # Medium relevance
            bottom_chunks = results[2 * third :]  # Lower but still relevant

            # Reorder: top at start, bottom at end, middle in between
            reordered = top_chunks + middle_chunks + bottom_chunks

        elif ordering == "diversity":
            # Interleave diverse chunks
            # TODO: Implement diversity-based reordering
            reordered = results

        elif ordering == "recency":
            # Order by recency (chunk position in document)
            reordered = sorted(results, key=lambda x: x.chunk.chunk_index)

        else:
            reordered = results

        return reordered

    def _format_chunk(
        self,
        result: RetrievalResult,
        include_metadata: bool = True,
    ) -> str:
        """
        Format chunk with optional metadata.

        Args:
            result: Retrieval result
            include_metadata: Whether to include metadata

        Returns:
            Formatted chunk string
        """
        chunk = result.chunk

        if not include_metadata:
            return chunk.content

        # Build metadata header
        metadata_parts = []

        # Source information
        source = chunk.metadata.get("file_name", chunk.document_id)
        metadata_parts.append(f"Source: {source}")

        # Relevance score
        metadata_parts.append(f"Score: {result.score:.3f}")

        # Code-specific metadata
        if chunk.chunk_type == "code":
            if chunk.function_name:
                metadata_parts.append(f"Function: {chunk.function_name}")
            if chunk.class_name:
                metadata_parts.append(f"Class: {chunk.class_name}")
            if chunk.language:
                metadata_parts.append(f"Language: {chunk.language}")

        # Document structure metadata
        if chunk.section:
            metadata_parts.append(f"Section: {chunk.section}")
        if chunk.page_number:
            metadata_parts.append(f"Page: {chunk.page_number}")

        metadata_str = " | ".join(metadata_parts)

        # Format with markdown-style header
        return f"""[{metadata_str}]
{chunk.content}"""

    def _compress_context(
        self,
        formatted_chunks: List[str],
        max_tokens: int,
        reserve_tokens: int,
    ) -> List[str]:
        """
        Compress context to fit within token limit.

        Args:
            formatted_chunks: List of formatted chunk strings
            max_tokens: Maximum total tokens
            reserve_tokens: Tokens to reserve

        Returns:
            Compressed chunks
        """
        available_tokens = max_tokens - reserve_tokens

        # Calculate current usage
        total_tokens = sum(
            self.window_optimizer.estimate_tokens(chunk) for chunk in formatted_chunks
        )

        # If within limit, return as-is
        if total_tokens <= available_tokens:
            return formatted_chunks

        # Calculate compression ratio needed
        compression_ratio = available_tokens / total_tokens

        # Compress each chunk
        compressed = []
        for chunk in formatted_chunks:
            chunk_tokens = self.window_optimizer.estimate_tokens(chunk)
            target_tokens = int(chunk_tokens * compression_ratio)

            if target_tokens < 50:  # Skip very small chunks
                continue

            compressed_chunk = self.window_optimizer.truncate_to_tokens(
                chunk,
                target_tokens,
            )

            if compressed_chunk:
                compressed.append(compressed_chunk)

        return compressed

    def build_simple_context(
        self,
        results: List[RetrievalResult],
        max_chunks: int = 5,
    ) -> str:
        """
        Build simple context without advanced optimization.

        Args:
            results: Retrieval results
            max_chunks: Maximum number of chunks

        Returns:
            Simple formatted context
        """
        if not results:
            return ""

        # Take top results
        top_results = sorted(results, key=lambda x: x.score, reverse=True)[:max_chunks]

        # Format chunks
        chunks = []
        for i, result in enumerate(top_results, 1):
            chunks.append(f"[Result {i}]")
            chunks.append(result.chunk.content)

        return "\n\n".join(chunks)
