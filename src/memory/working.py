"""Redis-based working memory for immediate context storage."""

import json
import logging
from typing import Dict, Any, List, Optional
import asyncio
from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from .base import BaseMemory
from ..models.memory_models import MemoryItem, MemorySearchRequest, MemorySearchResult
from ..config.memory_config import MemoryConfig

logger = logging.getLogger(__name__)


class RedisWorkingMemory(BaseMemory):
    """Redis-based working memory for immediate context storage with TTL."""

    def __init__(self, config: MemoryConfig):
        """
        Initialize Redis working memory.

        Args:
            config: Memory system configuration
        """
        self.config = config
        self.ttl = config.memory_ttl_working
        self._pool: Optional[ConnectionPool] = None
        self._redis: Optional[Redis] = None

    async def _get_redis(self) -> Redis:
        """
        Get Redis client with connection pooling and retry logic.

        Returns:
            Redis async client
        """
        if self._redis is None:
            self._pool = ConnectionPool.from_url(
                self.config.redis_url,
                max_connections=self.config.connection_pool_size,
                decode_responses=True,
            )
            self._redis = Redis(connection_pool=self._pool)

            # Test connection with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await self._redis.ping()
                    logger.info("Redis connection established successfully")
                    break
                except (RedisConnectionError, RedisError) as e:
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed to connect to Redis after {max_retries} attempts: {e}"
                        )
                        raise
                    logger.warning(
                        f"Redis connection attempt {attempt + 1} failed, retrying..."
                    )
                    await asyncio.sleep(1)

        return self._redis

    def _generate_key(self, memory: MemoryItem) -> str:
        """
        Generate Redis key following naming convention.

        Pattern: {tier}:{agent_id}:{session_id}:{memory_id}

        Args:
            memory: Memory item

        Returns:
            Redis key string
        """
        session = memory.session_id or "default"
        return f"working:{memory.agent_id}:{session}:{memory.id}"

    async def add_memory(self, memory: MemoryItem, **kwargs: Any) -> str:
        """
        Add memory to Redis with TTL.

        Args:
            memory: Memory item to store
            **kwargs: Additional parameters (ttl override)

        Returns:
            Memory ID
        """
        redis = await self._get_redis()
        key = self._generate_key(memory)
        ttl = kwargs.get("ttl", self.ttl)

        # Convert memory to JSON, excluding embedding to save space
        # Use mode='json' to properly serialize datetime objects
        memory_dict = memory.model_dump(mode="json")
        if "embedding" in memory_dict:
            # Store embedding reference only, not full vector
            memory_dict["embedding"] = None

        try:
            # Store with TTL
            await redis.setex(
                key,
                ttl,
                json.dumps(memory_dict),
            )
            logger.debug(f"Stored memory {memory.id} with TTL {ttl}s")
            return memory.id

        except RedisError as e:
            logger.error(f"Failed to store memory {memory.id}: {e}")
            raise

    async def get_memory(
        self,
        memory_id: str,
        agent_id: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[MemoryItem]:
        """
        Retrieve memory by ID.

        Args:
            memory_id: Memory identifier
            agent_id: Agent ID for key generation
            session_id: Session ID for key generation
            **kwargs: Additional parameters

        Returns:
            Memory item if found
        """
        redis = await self._get_redis()
        session = session_id or "default"
        key = f"working:{agent_id}:{session}:{memory_id}"

        try:
            data = await redis.get(key)
            if data:
                memory_dict = json.loads(data)
                # Update access count
                memory = MemoryItem(**memory_dict)
                memory.access_count += 1

                # Preserve the original TTL by getting the remaining time
                remaining_ttl = await redis.ttl(key)
                # If TTL is -1 (no expiry) or -2 (key doesn't exist), use default TTL
                ttl_to_use = remaining_ttl if remaining_ttl > 0 else self.ttl

                await redis.setex(
                    key, ttl_to_use, json.dumps(memory.model_dump(mode="json"))
                )
                return memory
            return None

        except RedisError as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            raise

    async def search_memories(
        self,
        request: MemorySearchRequest,
        agent_id: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[MemorySearchResult]:
        """
        Search memories by pattern matching.

        Note: Working memory uses simple pattern matching, not vector search.

        Args:
            request: Search request
            agent_id: Agent ID for filtering
            session_id: Session ID for filtering
            **kwargs: Additional parameters

        Returns:
            List of matching memories
        """
        redis = await self._get_redis()
        session = session_id or "default"
        pattern = f"working:{agent_id}:{session}:*"

        try:
            results = []
            async for key in redis.scan_iter(match=pattern, count=100):
                data = await redis.get(key)
                if data:
                    memory_dict = json.loads(data)
                    memory = MemoryItem(**memory_dict)

                    # Simple text matching
                    score = self._calculate_relevance(request.query, memory.content)
                    if score >= request.score_threshold:
                        results.append(
                            MemorySearchResult(
                                memory=memory,
                                score=score,
                                source="working",
                            )
                        )

            # Sort by score descending and limit to k
            results.sort(key=lambda x: x.score, reverse=True)
            return results[: request.k]

        except RedisError as e:
            logger.error(f"Failed to search memories: {e}")
            raise

    def _calculate_relevance(self, query: str, content: str) -> float:
        """
        Calculate simple text relevance score.

        Args:
            query: Search query
            content: Memory content

        Returns:
            Relevance score (0-1)
        """
        query_lower = query.lower()
        content_lower = content.lower()

        # Exact match
        if query_lower == content_lower:
            return 1.0

        # Contains query
        if query_lower in content_lower:
            return 0.8

        # Word overlap
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        overlap = len(query_words & content_words)
        if overlap > 0:
            return overlap / max(len(query_words), len(content_words))

        return 0.0

    async def delete_memory(
        self,
        memory_id: str,
        agent_id: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Delete a memory item.

        Args:
            memory_id: Memory ID to delete
            agent_id: Agent ID for key generation
            session_id: Session ID for key generation
            **kwargs: Additional parameters

        Returns:
            True if deleted, False if not found
        """
        redis = await self._get_redis()
        session = session_id or "default"
        key = f"working:{agent_id}:{session}:{memory_id}"

        try:
            result = await redis.delete(key)
            return result > 0

        except RedisError as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise

    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any],
        agent_id: str,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[MemoryItem]:
        """
        Update a memory item.

        Args:
            memory_id: Memory ID to update
            updates: Fields to update
            agent_id: Agent ID for key generation
            session_id: Session ID for key generation
            **kwargs: Additional parameters

        Returns:
            Updated memory item
        """
        memory = await self.get_memory(memory_id, agent_id, session_id)
        if not memory:
            return None

        # Apply updates
        memory_dict = memory.model_dump()
        memory_dict.update(updates)
        updated_memory = MemoryItem(**memory_dict)

        # Store with new TTL
        await self.add_memory(updated_memory, **kwargs)
        return updated_memory

    async def clear_all(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        """
        Clear all working memories.

        Args:
            agent_id: Optional agent ID filter
            session_id: Optional session ID filter
            **kwargs: Additional parameters

        Returns:
            Number of memories cleared
        """
        redis = await self._get_redis()

        # Build pattern based on filters
        if agent_id and session_id:
            pattern = f"working:{agent_id}:{session_id}:*"
        elif agent_id:
            pattern = f"working:{agent_id}:*"
        else:
            pattern = "working:*"

        try:
            count = 0
            async for key in redis.scan_iter(match=pattern, count=100):
                await redis.delete(key)
                count += 1

            logger.info(f"Cleared {count} working memories")
            return count

        except RedisError as e:
            logger.error(f"Failed to clear memories: {e}")
            raise

    async def set_state(
        self, agent_id: str, session_id: str, state: Dict[str, Any]
    ) -> None:
        """
        Set agent state (convenience method for checkpointing).

        Args:
            agent_id: Agent identifier
            session_id: Session identifier
            state: State dictionary to store
        """
        redis = await self._get_redis()
        key = f"working:{agent_id}:{session_id}:state"

        try:
            await redis.setex(key, self.ttl, json.dumps(state))
            logger.debug(f"Stored state for {agent_id}/{session_id}")

        except RedisError as e:
            logger.error(f"Failed to store state: {e}")
            raise

    async def get_state(
        self, agent_id: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get agent state.

        Args:
            agent_id: Agent identifier
            session_id: Session identifier

        Returns:
            State dictionary if found
        """
        redis = await self._get_redis()
        key = f"working:{agent_id}:{session_id}:state"

        try:
            data = await redis.get(key)
            return json.loads(data) if data else None

        except RedisError as e:
            logger.error(f"Failed to retrieve state: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
        if self._pool:
            await self._pool.aclose()
            self._pool = None
        logger.info("Redis connection closed")
