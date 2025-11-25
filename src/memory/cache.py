"""Semantic caching layer for reducing duplicate work."""

import logging
import hashlib
import json
from typing import Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with semantic key and expiration."""

    key: str
    semantic_key: List[float]  # Embedding of the query
    value: Any
    created_at: datetime
    ttl: int  # seconds
    access_count: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)


class SemanticCache:
    """
    Semantic caching layer using embedding similarity.

    Caches results based on semantic similarity rather than exact matches.
    Useful for reducing redundant LLM calls and memory searches.
    """

    def __init__(
        self,
        embedding_function: Optional[Callable] = None,
        max_size: int = 1000,
        default_ttl: int = 3600,
        similarity_threshold: float = 0.95,
    ):
        """
        Initialize semantic cache.

        Args:
            embedding_function: Function to generate embeddings
            max_size: Maximum cache size (LRU eviction)
            default_ttl: Default TTL in seconds
            similarity_threshold: Minimum similarity for cache hit (0-1)
        """
        self.embedding_function = embedding_function
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        self._cache: List[CacheEntry] = []

    async def get(
        self,
        query: str,
        exact_match: bool = False,
    ) -> Optional[Any]:
        """
        Get cached value for query.

        Args:
            query: Query string
            exact_match: If True, require exact match; if False, use semantic similarity

        Returns:
            Cached value if found, None otherwise
        """
        if exact_match:
            # Fast path: exact match
            key = self._hash_query(query)
            for entry in self._cache:
                if entry.key == key and not entry.is_expired():
                    entry.access_count += 1
                    logger.debug(f"Cache hit (exact): {query[:50]}...")
                    return entry.value
            return None

        # Semantic search
        if self.embedding_function is None:
            logger.debug("No embedding function, falling back to exact match")
            return await self.get(query, exact_match=True)

        query_embedding = await self._generate_embedding(query)

        # Find most similar non-expired entry
        best_match = None
        best_similarity = 0.0

        for entry in self._cache:
            if entry.is_expired():
                continue

            similarity = self._cosine_similarity(query_embedding, entry.semantic_key)
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = entry

        if best_match:
            best_match.access_count += 1
            logger.debug(
                f"Cache hit (semantic, similarity={best_similarity:.3f}): {query[:50]}..."
            )
            return best_match.value

        logger.debug(f"Cache miss: {query[:50]}...")
        return None

    async def set(
        self,
        query: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Cache a value for query.

        Args:
            query: Query string
            value: Value to cache
            ttl: Time to live in seconds
        """
        key = self._hash_query(query)
        ttl = ttl or self.default_ttl

        # Generate semantic key
        if self.embedding_function:
            semantic_key = await self._generate_embedding(query)
        else:
            semantic_key = []

        # Create entry
        entry = CacheEntry(
            key=key,
            semantic_key=semantic_key,
            value=value,
            created_at=datetime.now(),
            ttl=ttl,
        )

        # Add to cache
        self._cache.append(entry)

        # Evict if over size limit
        if len(self._cache) > self.max_size:
            self._evict_lru()

        logger.debug(f"Cached: {query[:50]}...")

    async def invalidate(
        self,
        query: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            query: Specific query to invalidate
            pattern: Pattern to match for invalidation

        Returns:
            Number of entries invalidated
        """
        if query:
            key = self._hash_query(query)
            original_size = len(self._cache)
            self._cache = [e for e in self._cache if e.key != key]
            count = original_size - len(self._cache)
            logger.debug(f"Invalidated {count} entries for query")
            return count

        if pattern:
            # Simple pattern matching
            original_size = len(self._cache)
            # For simplicity, we'll match against a JSON representation
            # In production, you might want more sophisticated pattern matching
            self._cache = [e for e in self._cache if pattern not in json.dumps(e.value)]
            count = original_size - len(self._cache)
            logger.debug(f"Invalidated {count} entries matching pattern")
            return count

        return 0

    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cache entries")

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        original_size = len(self._cache)
        self._cache = [e for e in self._cache if not e.is_expired()]
        count = original_size - len(self._cache)
        if count > 0:
            logger.debug(f"Cleaned up {count} expired entries")
        return count

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_entries = len(self._cache)
        expired_entries = sum(1 for e in self._cache if e.is_expired())
        total_accesses = sum(e.access_count for e in self._cache)

        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_entries,
            "expired_entries": expired_entries,
            "max_size": self.max_size,
            "utilization": total_entries / self.max_size if self.max_size > 0 else 0,
            "total_accesses": total_accesses,
        }

    def _hash_query(self, query: str) -> str:
        """Generate hash for exact matching."""
        return hashlib.sha256(query.encode()).hexdigest()

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        import asyncio

        if asyncio.iscoroutinefunction(self.embedding_function):
            return await self.embedding_function(text)
        else:
            return self.embedding_function(text)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0-1)
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        # Calculate dot product and magnitudes
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Remove expired entries first
        self.cleanup_expired()

        # If still over limit, remove least accessed
        if len(self._cache) > self.max_size:
            self._cache.sort(key=lambda e: e.access_count)
            removed = self._cache.pop(0)
            logger.debug(f"Evicted LRU entry (access_count={removed.access_count})")
