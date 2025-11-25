"""Semantic code search with embeddings."""

import logging
from typing import List

from ..models.analysis_models import (
    CodeEntity,
    SemanticSearchRequest,
    SemanticSearchResult,
)

logger = logging.getLogger(__name__)


class SemanticSearch:
    """
    Semantic code search using embeddings.

    PATTERN: Code embeddings with vector similarity
    GOTCHA: Requires LLM service for embeddings
    NOTE: Simplified implementation - full version would use vector database
    """

    def __init__(self, llm_service=None):
        """
        Initialize semantic search.

        Args:
            llm_service: Optional LLM service for embeddings
        """
        self.llm_service = llm_service
        self.logger = logger
        self.index: List[tuple[CodeEntity, List[float]]] = []

    async def index_code(self, entities: List[CodeEntity]):
        """
        Index code entities for semantic search.

        CRITICAL: Batch embeddings for efficiency

        Args:
            entities: List of code entities to index
        """
        if not self.llm_service:
            self.logger.warning("No LLM service available for embeddings")
            return

        # Prepare code snippets for embedding
        snippets = []
        for entity in entities:
            snippet = self._entity_to_text(entity)
            snippets.append(snippet)

        try:
            # Generate embeddings (placeholder - would use actual LLM service)
            # embeddings = await self.llm_service.generate_embeddings(snippets)

            # For now, store entities without actual embeddings
            for entity in entities:
                # Placeholder embedding
                embedding: List[float] = []
                self.index.append((entity, embedding))

            self.logger.info(f"Indexed {len(entities)} entities")

        except Exception as e:
            self.logger.error(f"Failed to index entities: {e}")

    async def search(
        self, request: SemanticSearchRequest
    ) -> List[SemanticSearchResult]:
        """
        Search for code using semantic similarity.

        PATTERN: Embed query, find similar vectors

        Args:
            request: Search request

        Returns:
            List of search results
        """
        if not self.llm_service:
            self.logger.warning("No LLM service available for search")
            return []

        # For now, return simple text-based matching
        # Full implementation would use vector similarity
        results = []

        query_lower = request.query.lower()

        for entity, _embedding in self.index:
            # Simple text matching as fallback
            score = self._calculate_text_similarity(query_lower, entity)

            if score >= request.similarity_threshold:
                result = SemanticSearchResult(
                    code_snippet=entity.signature or entity.name,
                    file_path=entity.file_path,
                    line_start=entity.line_start,
                    line_end=entity.line_end,
                    similarity_score=score,
                    entity=entity,
                )
                results.append(result)

        # Sort by similarity
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        # Return top results
        return results[: request.max_results]

    def _entity_to_text(self, entity: CodeEntity) -> str:
        """Convert entity to text for embedding."""
        parts = [
            f"Type: {entity.type.value}",
            f"Name: {entity.name}",
        ]

        if entity.signature:
            parts.append(f"Signature: {entity.signature}")

        if entity.docstring:
            parts.append(f"Doc: {entity.docstring}")

        return "\n".join(parts)

    def _calculate_text_similarity(self, query: str, entity: CodeEntity) -> float:
        """
        Calculate simple text similarity.

        This is a placeholder - full version would use embedding similarity.
        """
        # Simple keyword matching
        entity_text = self._entity_to_text(entity).lower()

        # Count matching words
        query_words = set(query.split())
        entity_words = set(entity_text.split())

        if not query_words:
            return 0.0

        matches = query_words.intersection(entity_words)
        similarity = len(matches) / len(query_words)

        return similarity

    def clear_index(self):
        """Clear the search index."""
        self.index = []
        self.logger.info("Search index cleared")
