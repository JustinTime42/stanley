"""LLM response caching layer for cost optimization."""

import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from ..models.llm_models import LLMResponse

logger = logging.getLogger(__name__)


class LLMResponseCache:
    """
    In-memory cache for LLM responses with TTL support.

    PATTERN: Simple dict-based cache with TTL and size limits
    CRITICAL: Cache key must include request parameters
    GOTCHA: Don't cache responses with different temperatures
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,
    ):
        """
        Initialize LLM response cache.

        Args:
            max_size: Maximum number of cached entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    async def get(self, cache_key: str) -> Optional[LLMResponse]:
        """
        Retrieve cached response.

        PATTERN: Check expiration before returning
        CRITICAL: Return None if expired or not found

        Args:
            cache_key: Cache key

        Returns:
            Cached LLMResponse if found and valid, None otherwise
        """
        if cache_key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[cache_key]

        # Check expiration
        if datetime.now() > entry["expires_at"]:
            self.logger.debug(f"Cache entry expired: {cache_key[:16]}...")
            del self.cache[cache_key]
            self.misses += 1
            return None

        # Update access time for LRU
        entry["last_accessed"] = datetime.now()
        entry["access_count"] += 1

        self.hits += 1
        self.logger.debug(
            f"Cache hit: {cache_key[:16]}... "
            f"(hit rate: {self.get_hit_rate():.2%})"
        )

        # Return cached response with cache_hit flag set
        response = entry["response"]
        response.cache_hit = True
        return response

    async def set(
        self,
        cache_key: str,
        response: LLMResponse,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store response in cache.

        PATTERN: Evict oldest if at capacity
        CRITICAL: Don't cache error responses

        Args:
            cache_key: Cache key
            response: LLM response to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        # Check cache size
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        # Calculate expiration
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)

        # Store entry
        self.cache[cache_key] = {
            "response": response,
            "created_at": datetime.now(),
            "expires_at": expires_at,
            "last_accessed": datetime.now(),
            "access_count": 0,
        }

        self.logger.debug(
            f"Cached response: {cache_key[:16]}... "
            f"(size: {len(self.cache)}/{self.max_size})"
        )

    def _evict_lru(self) -> None:
        """
        Evict least recently used entry.

        PATTERN: LRU eviction based on last_accessed time
        CRITICAL: Always evict when at capacity
        """
        if not self.cache:
            return

        # Find LRU entry
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k]["last_accessed"],
        )

        # Remove it
        del self.cache[lru_key]
        self.evictions += 1

        self.logger.debug(f"Evicted LRU entry: {lru_key[:16]}...")

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        PATTERN: Periodic cleanup to free memory
        CRITICAL: Run periodically to prevent unbounded growth

        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now > entry["expires_at"]
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired entries")

        return len(expired_keys)

    def clear(self) -> None:
        """Clear all cached entries."""
        count = len(self.cache)
        self.cache.clear()
        self.logger.info(f"Cleared {count} cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }

    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate.

        Returns:
            Hit rate (0-1)
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0

    def get_entry_info(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a cache entry.

        Args:
            cache_key: Cache key

        Returns:
            Entry information if found
        """
        if cache_key not in self.cache:
            return None

        entry = self.cache[cache_key]
        return {
            "created_at": entry["created_at"].isoformat(),
            "expires_at": entry["expires_at"].isoformat(),
            "last_accessed": entry["last_accessed"].isoformat(),
            "access_count": entry["access_count"],
            "model_used": entry["response"].model_used,
            "cost_saved": entry["response"].total_cost * entry["access_count"],
        }

    def estimate_cost_savings(self) -> float:
        """
        Estimate total cost savings from caching.

        PATTERN: Sum cost * (access_count - 1) for all entries
        CRITICAL: Actual savings from avoiding redundant API calls

        Returns:
            Estimated cost savings in USD
        """
        total_savings = 0.0

        for entry in self.cache.values():
            response = entry["response"]
            access_count = entry["access_count"]

            # Savings = cost per call * number of avoided calls
            if access_count > 0:
                avoided_calls = access_count  # We cached after first call
                total_savings += response.total_cost * avoided_calls

        return total_savings


class RedisLLMCache:
    """
    Redis-based LLM cache for distributed caching.

    PATTERN: Use Redis for shared cache across workers
    CRITICAL: Requires Redis connection
    """

    def __init__(
        self,
        redis_client,
        prefix: str = "llm_cache:",
        default_ttl: int = 3600,
    ):
        """
        Initialize Redis-based cache.

        Args:
            redis_client: Redis client instance
            prefix: Key prefix for namespacing
            default_ttl: Default TTL in seconds
        """
        self.redis = redis_client
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.logger = logging.getLogger(__name__)

    async def get(self, cache_key: str) -> Optional[LLMResponse]:
        """
        Retrieve from Redis cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached response if found
        """
        try:
            key = f"{self.prefix}{cache_key}"
            data = await self.redis.get(key)

            if data:
                response_dict = json.loads(data)
                response = LLMResponse(**response_dict)
                response.cache_hit = True
                return response

            return None

        except Exception as e:
            self.logger.error(f"Redis cache get error: {e}")
            return None

    async def set(
        self,
        cache_key: str,
        response: LLMResponse,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store in Redis cache.

        Args:
            cache_key: Cache key
            response: Response to cache
            ttl: TTL in seconds
        """
        try:
            key = f"{self.prefix}{cache_key}"
            ttl = ttl or self.default_ttl

            # Serialize response
            data = response.model_dump_json()

            # Store with TTL
            await self.redis.setex(key, ttl, data)

        except Exception as e:
            self.logger.error(f"Redis cache set error: {e}")
