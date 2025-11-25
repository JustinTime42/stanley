"""Memory orchestration service coordinating all memory tiers."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..memory import (
    RedisWorkingMemory,
    QdrantProjectMemory,
    QdrantGlobalMemory,
    HybridSearchManager,
    SemanticCache,
)
from ..models.memory_models import (
    MemoryItem,
    MemorySearchRequest,
    MemorySearchResult,
    MemoryType,
)
from ..config.memory_config import MemoryConfig

logger = logging.getLogger(__name__)


class MemoryOrchestrator:
    """
    Memory orchestration service providing unified interface to all memory tiers.

    Implements the Facade pattern to simplify memory operations across
    working, project, and global memory tiers.
    """

    def __init__(
        self,
        config: MemoryConfig,
        embedding_function=None,
    ):
        """
        Initialize memory orchestrator.

        Args:
            config: Memory system configuration
            embedding_function: Function to generate embeddings
        """
        self.config = config
        self.embedding_function = embedding_function

        # Initialize memory tiers
        self.working_memory = RedisWorkingMemory(config=config)
        self.project_memory = QdrantProjectMemory(
            config=config,
            embedding_function=embedding_function,
        )
        self.global_memory = QdrantGlobalMemory(
            config=config,
            embedding_function=embedding_function,
        )

        # Initialize semantic cache
        self.cache = SemanticCache(
            embedding_function=embedding_function,
            max_size=config.memory_cache_size,
            default_ttl=config.memory_ttl_working,
        )

        # Initialize hybrid search for project and global memories
        self.project_hybrid_search = None
        self.global_hybrid_search = None

    async def initialize_hybrid_search(self):
        """Initialize hybrid search managers."""

        # Create BM25 search functions
        async def project_keyword_search(request: MemorySearchRequest, **kwargs):
            # Get all project memories for BM25
            # In production, use a dedicated search engine like BM25KeywordSearch
            # This is simplified; in production, you'd maintain an index
            return []

        async def project_vector_search(request: MemorySearchRequest, **kwargs):
            return await self.project_memory.search_memories(request, **kwargs)

        self.project_hybrid_search = HybridSearchManager(
            vector_search_fn=project_vector_search,
            keyword_search_fn=project_keyword_search,
        )

    async def store_memory(
        self,
        content: str,
        agent_id: str,
        memory_type: MemoryType = MemoryType.PROJECT,
        session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Store a memory in the appropriate tier.

        Args:
            content: Memory content
            agent_id: Agent creating the memory
            memory_type: Memory tier (working/project/global)
            session_id: Session identifier
            project_id: Project identifier
            importance: Memory importance (0-1)
            tags: Memory tags
            metadata: Additional metadata
            **kwargs: Additional parameters

        Returns:
            Memory ID
        """
        # Create memory item
        import uuid

        memory = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            agent_id=agent_id,
            memory_type=memory_type,
            session_id=session_id,
            project_id=project_id,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            timestamp=datetime.now(),
        )

        # Store in appropriate tier
        if memory_type == MemoryType.WORKING:
            memory_id = await self.working_memory.add_memory(memory, **kwargs)
        elif memory_type == MemoryType.PROJECT:
            memory_id = await self.project_memory.add_memory(memory, **kwargs)
        elif memory_type == MemoryType.GLOBAL:
            memory_id = await self.global_memory.add_memory(memory, **kwargs)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")

        logger.info(f"Stored memory {memory_id} in {memory_type.value} tier")
        return memory_id

    async def retrieve_memory(
        self,
        memory_id: str,
        memory_type: MemoryType,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[MemoryItem]:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: Memory identifier
            memory_type: Memory tier to retrieve from
            agent_id: Agent ID (required for working memory)
            session_id: Session ID (for working memory)
            **kwargs: Additional parameters

        Returns:
            Memory item if found
        """
        if memory_type == MemoryType.WORKING:
            if not agent_id:
                raise ValueError("agent_id required for working memory retrieval")
            return await self.working_memory.get_memory(
                memory_id=memory_id,
                agent_id=agent_id,
                session_id=session_id,
                **kwargs,
            )
        elif memory_type == MemoryType.PROJECT:
            return await self.project_memory.get_memory(memory_id, **kwargs)
        elif memory_type == MemoryType.GLOBAL:
            return await self.global_memory.get_memory(memory_id, **kwargs)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")

    async def retrieve_relevant_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> List[MemorySearchResult]:
        """
        Retrieve relevant memories across specified tiers.

        Args:
            query: Search query
            memory_types: Memory tiers to search (default: project only)
            k: Number of results per tier
            filters: Metadata filters
            use_hybrid: Use hybrid search
            use_cache: Use semantic cache
            **kwargs: Additional parameters

        Returns:
            Combined search results from all tiers
        """
        memory_types = memory_types or [MemoryType.PROJECT]

        # Check cache first
        if use_cache:
            cached_results = await self.cache.get(query)
            if cached_results is not None:
                logger.info("Retrieved results from semantic cache")
                return cached_results

        # Create search request
        request = MemorySearchRequest(
            query=query,
            memory_types=memory_types,
            k=k,
            filters=filters or {},
            use_hybrid=use_hybrid,
            alpha=self.config.hybrid_search_alpha,
        )

        # Search across tiers
        all_results = []

        for memory_type in memory_types:
            try:
                if memory_type == MemoryType.WORKING:
                    # Working memory doesn't support vector search
                    results = await self.working_memory.search_memories(
                        request,
                        agent_id=filters.get("agent_id", "unknown"),
                        session_id=filters.get("session_id"),
                        **kwargs,
                    )
                elif memory_type == MemoryType.PROJECT:
                    if use_hybrid and self.project_hybrid_search:
                        results = await self.project_hybrid_search.hybrid_search(
                            request, **kwargs
                        )
                    else:
                        results = await self.project_memory.search_memories(
                            request, **kwargs
                        )
                elif memory_type == MemoryType.GLOBAL:
                    if use_hybrid and self.global_hybrid_search:
                        results = await self.global_hybrid_search.hybrid_search(
                            request, **kwargs
                        )
                    else:
                        results = await self.global_memory.search_memories(
                            request, **kwargs
                        )
                else:
                    logger.warning(f"Unknown memory type: {memory_type}")
                    continue

                all_results.extend(results)
                logger.debug(
                    f"Retrieved {len(results)} results from {memory_type.value}"
                )

            except Exception as e:
                # Don't log as error if it's just missing embedding function (expected in demo)
                if "embedding_function required" in str(e):
                    logger.debug(f"Skipping {memory_type.value} memory search: {e}")
                else:
                    logger.error(f"Error searching {memory_type.value} memory: {e}")
                # Continue with other tiers

        # Sort all results by score
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Limit to k results
        final_results = all_results[:k]

        # Cache results
        if use_cache:
            await self.cache.set(query, final_results)

        return final_results

    async def checkpoint_state(
        self,
        agent_id: str,
        session_id: str,
        state: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """
        Checkpoint agent state to working memory.

        Args:
            agent_id: Agent identifier
            session_id: Session identifier
            state: State dictionary
            **kwargs: Additional parameters
        """
        await self.working_memory.set_state(agent_id, session_id, state)
        logger.info(f"Checkpointed state for {agent_id}/{session_id}")

    async def restore_state(
        self,
        agent_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Restore agent state from working memory.

        Args:
            agent_id: Agent identifier
            session_id: Session identifier
            **kwargs: Additional parameters

        Returns:
            State dictionary if found
        """
        state = await self.working_memory.get_state(agent_id, session_id)
        if state:
            logger.info(f"Restored state for {agent_id}/{session_id}")
        else:
            logger.warning(f"No state found for {agent_id}/{session_id}")
        return state

    async def cleanup(self):
        """Clean up expired cache entries and connections."""
        # Cleanup cache
        expired = self.cache.cleanup_expired()
        logger.info(f"Cleaned up {expired} expired cache entries")

        # Close connections
        await self.working_memory.close()
        await self.project_memory.close()
        await self.global_memory.close()

    async def close(self):
        """Alias for cleanup() to match common close() pattern."""
        await self.cleanup()

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory usage.

        Returns:
            Dictionary with memory stats
        """
        return {
            "cache": self.cache.get_stats(),
            "timestamp": datetime.now().isoformat(),
        }
