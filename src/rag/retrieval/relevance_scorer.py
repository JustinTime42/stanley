"""Dynamic relevance scorer with intent-based adjustments."""

import logging
from typing import Dict, List

from ...models.document_models import Chunk
from ...models.rag_models import QueryAnalysis, QueryIntent

logger = logging.getLogger(__name__)


class RelevanceScorer:
    """
    Score chunk relevance with intent-based weight adjustments.

    PATTERN: Multi-factor weighted scoring
    CRITICAL: Different intents need different ranking signals
    """

    # Scoring weight profiles for each intent
    WEIGHT_PROFILES: Dict[QueryIntent, Dict[str, float]] = {
        QueryIntent.CODE_SEARCH: {
            "semantic": 0.3,
            "keyword": 0.2,
            "structural": 0.4,
            "recency": 0.1,
        },
        QueryIntent.DOCUMENTATION: {
            "semantic": 0.5,
            "keyword": 0.3,
            "structural": 0.1,
            "recency": 0.1,
        },
        QueryIntent.DEBUGGING: {
            "semantic": 0.4,
            "keyword": 0.4,
            "structural": 0.1,
            "recency": 0.1,
        },
        QueryIntent.EXAMPLE_SEARCH: {
            "semantic": 0.3,
            "keyword": 0.2,
            "structural": 0.3,
            "recency": 0.2,
        },
        QueryIntent.DEFINITION: {
            "semantic": 0.5,
            "keyword": 0.4,
            "structural": 0.05,
            "recency": 0.05,
        },
        QueryIntent.EXPLANATION: {
            "semantic": 0.6,
            "keyword": 0.3,
            "structural": 0.05,
            "recency": 0.05,
        },
        QueryIntent.QUESTION_ANSWER: {
            "semantic": 0.4,
            "keyword": 0.3,
            "structural": 0.2,
            "recency": 0.1,
        },
    }

    def __init__(self):
        """Initialize relevance scorer."""
        self.logger = logger

    def score_chunk(
        self,
        chunk: Chunk,
        query_analysis: QueryAnalysis,
        base_score: float,
    ) -> float:
        """
        Calculate relevance score with intent-based adjustments.

        PATTERN: Combine multiple scoring signals with weighted sum
        GOTCHA: Normalize all scores to 0-1 range

        Args:
            chunk: Chunk to score
            query_analysis: Analyzed query with intent
            base_score: Base similarity score (from vector search)

        Returns:
            Final relevance score (0-1)
        """
        # Get weight profile for this intent
        weights = self._get_weights_for_intent(query_analysis.intent)

        # Calculate component scores
        semantic_score = base_score  # Vector similarity
        keyword_score = self._calculate_keyword_overlap(
            chunk.keywords,
            query_analysis.keywords,
        )
        structural_score = self._calculate_structural_relevance(
            chunk,
            query_analysis,
        )
        recency_score = self._calculate_recency_score(chunk)

        # Apply weighted combination
        final_score = (
            weights["semantic"] * semantic_score
            + weights["keyword"] * keyword_score
            + weights["structural"] * structural_score
            + weights["recency"] * recency_score
        )

        # Apply content type boosting
        final_score = self._apply_content_boosting(
            final_score,
            chunk,
            query_analysis,
        )

        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, final_score))

        return final_score

    def _get_weights_for_intent(self, intent: QueryIntent) -> Dict[str, float]:
        """Get scoring weights based on query intent."""
        return self.WEIGHT_PROFILES.get(
            intent,
            {
                "semantic": 0.4,
                "keyword": 0.3,
                "structural": 0.2,
                "recency": 0.1,
            },
        )

    def _calculate_keyword_overlap(
        self,
        chunk_keywords: List[str],
        query_keywords: List[str],
    ) -> float:
        """
        Calculate keyword overlap score.

        PATTERN: Jaccard similarity with normalization

        Args:
            chunk_keywords: Keywords from chunk
            query_keywords: Keywords from query

        Returns:
            Overlap score (0-1)
        """
        if not chunk_keywords or not query_keywords:
            return 0.0

        chunk_set = set(kw.lower() for kw in chunk_keywords)
        query_set = set(kw.lower() for kw in query_keywords)

        # Calculate Jaccard similarity
        intersection = len(chunk_set & query_set)
        union = len(chunk_set | query_set)

        if union == 0:
            return 0.0

        return intersection / union

    def _calculate_structural_relevance(
        self,
        chunk: Chunk,
        query_analysis: QueryAnalysis,
    ) -> float:
        """
        Calculate structural relevance score.

        PATTERN: Boost chunks with matching structural features

        Args:
            chunk: Chunk to score
            query_analysis: Query analysis

        Returns:
            Structural score (0-1)
        """
        score = 0.0

        # Check for code structure matches
        if chunk.chunk_type == "code":
            # Boost if query mentions function/class names
            if chunk.function_name:
                for entity in query_analysis.entities:
                    if entity.lower() == chunk.function_name.lower():
                        score += 0.5

            if chunk.class_name:
                for entity in query_analysis.entities:
                    if entity.lower() == chunk.class_name.lower():
                        score += 0.5

            # Boost for programming language match
            if chunk.language:
                for term in query_analysis.programming_terms:
                    if term.lower() == chunk.language.lower():
                        score += 0.3

        # Check for document structure matches
        if chunk.section:
            for keyword in query_analysis.keywords:
                if keyword in chunk.section.lower():
                    score += 0.2

        return min(1.0, score)

    def _calculate_recency_score(self, chunk: Chunk) -> float:
        """
        Calculate recency score based on chunk position.

        PATTERN: Prefer earlier chunks (closer to document start)
        GOTCHA: This is simplistic, could use actual timestamps

        Args:
            chunk: Chunk to score

        Returns:
            Recency score (0-1)
        """
        # For now, use inverse of chunk index as proxy for recency
        # Earlier chunks get higher score
        # This assumes documents are organized with important info first

        # Use exponential decay
        decay_factor = 0.95
        score = decay_factor**chunk.chunk_index

        return score

    def _apply_content_boosting(
        self,
        score: float,
        chunk: Chunk,
        query_analysis: QueryAnalysis,
    ) -> float:
        """
        Apply content type boosting based on query requirements.

        Args:
            score: Current score
            chunk: Chunk being scored
            query_analysis: Query analysis

        Returns:
            Boosted score
        """
        # Boost code chunks if query requires code
        if query_analysis.requires_code and chunk.chunk_type == "code":
            score *= 1.2

        # Boost documentation chunks if query requires docs
        elif query_analysis.requires_docs and chunk.chunk_type == "text":
            score *= 1.1

        # Exact function/class name match gets significant boost
        if chunk.function_name:
            for entity in query_analysis.entities:
                if entity.lower() == chunk.function_name.lower():
                    score *= 1.5

        if chunk.class_name:
            for entity in query_analysis.entities:
                if entity.lower() == chunk.class_name.lower():
                    score *= 1.5

        return score

    def rerank_results(
        self,
        chunks: List[tuple[Chunk, float]],
        query_analysis: QueryAnalysis,
        top_k: int = 10,
    ) -> List[tuple[Chunk, float]]:
        """
        Re-rank results using full scoring model.

        Args:
            chunks: List of (chunk, base_score) tuples
            query_analysis: Query analysis
            top_k: Number of top results to return

        Returns:
            Re-ranked list of (chunk, final_score) tuples
        """
        # Score all chunks
        scored_chunks = []
        for chunk, base_score in chunks:
            final_score = self.score_chunk(chunk, query_analysis, base_score)
            scored_chunks.append((chunk, final_score))

        # Sort by final score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        return scored_chunks[:top_k]
