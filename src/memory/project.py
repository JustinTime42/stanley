"""Qdrant-based project memory for project-specific vector storage."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from qdrant_client.http.exceptions import UnexpectedResponse

from .base import BaseMemory
from ..models.memory_models import MemoryItem, MemorySearchRequest, MemorySearchResult
from ..config.memory_config import MemoryConfig

logger = logging.getLogger(__name__)


class QdrantProjectMemory(BaseMemory):
    """Qdrant-based project memory for vector search."""

    def __init__(self, config: MemoryConfig, embedding_function=None):
        """
        Initialize Qdrant project memory.

        Args:
            config: Memory system configuration
            embedding_function: Function to generate embeddings
        """
        self.config = config
        self.collection_name = config.project_collection_name
        self.embedding_function = embedding_function
        self._client: Optional[AsyncQdrantClient] = None
        self._qdrant_available: Optional[bool] = None  # None = not yet checked

    async def _get_client(self) -> Optional[AsyncQdrantClient]:
        """
        Get Qdrant client and ensure collection exists.

        Returns:
            Async Qdrant client, or None if Qdrant is not available
        """
        # If we already know Qdrant is unavailable, return None immediately
        if self._qdrant_available is False:
            return None

        if self._client is None:
            try:
                # Create client (suppress the insecure connection warning)
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")
                    self._client = AsyncQdrantClient(
                        url=self.config.qdrant_url,
                        api_key=self.config.qdrant_api_key,
                    )

                # Ensure collection exists with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Check if collection exists
                        collections = await self._client.get_collections()
                        collection_names = [c.name for c in collections.collections]

                        if self.collection_name not in collection_names:
                            # Create collection
                            await self._client.create_collection(
                                collection_name=self.collection_name,
                                vectors_config=VectorParams(
                                    size=self.config.vector_size,
                                    distance=Distance.COSINE,
                                ),
                            )
                            logger.debug(
                                f"Created Qdrant collection: {self.collection_name}"
                            )
                        else:
                            logger.debug(
                                f"Using existing collection: {self.collection_name}"
                            )
                        self._qdrant_available = True
                        break

                    except (UnexpectedResponse, Exception) as e:
                        if attempt == max_retries - 1:
                            logger.debug(
                                f"Qdrant not available after {max_retries} attempts, continuing without it"
                            )
                            self._qdrant_available = False
                            self._client = None
                            return None
                        await asyncio.sleep(0.5)  # Shorter retry delay

            except Exception as e:
                logger.debug(f"Qdrant not available: {e}")
                self._qdrant_available = False
                self._client = None
                return None

        return self._client

    async def add_memory(self, memory: MemoryItem, **kwargs: Any) -> str:
        """
        Add memory with vector embedding.

        Args:
            memory: Memory item to store
            **kwargs: Additional parameters

        Returns:
            Memory ID
        """
        client = await self._get_client()

        # If Qdrant is not available, return memory ID without storing
        if client is None:
            logger.debug(f"Qdrant not available, skipping memory storage for {memory.id}")
            return memory.id

        # Generate embedding if not provided
        if memory.embedding is None and self.embedding_function:
            memory.embedding = await self._generate_embedding(memory.content)

        if memory.embedding is None:
            raise ValueError(
                "Memory must have embedding or embedding_function must be provided"
            )

        # Prepare point
        point = PointStruct(
            id=memory.id,
            vector=memory.embedding,
            payload={
                "content": memory.content,
                "agent_id": memory.agent_id,
                "session_id": memory.session_id,
                "project_id": memory.project_id,
                "memory_type": memory.memory_type.value,
                "importance": memory.importance,
                "access_count": memory.access_count,
                "tags": memory.tags,
                "timestamp": memory.timestamp.isoformat(),
                **memory.metadata,
            },
        )

        try:
            await client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )
            logger.debug(f"Stored memory {memory.id} in Qdrant")
            return memory.id

        except Exception as e:
            logger.debug(f"Failed to store memory {memory.id}: {e}")
            return memory.id  # Return ID even on failure

    async def add_memories(
        self, memories: List[MemoryItem], **kwargs: Any
    ) -> List[str]:
        """
        Batch add memories for efficiency.

        Args:
            memories: List of memory items
            **kwargs: Additional parameters

        Returns:
            List of memory IDs
        """
        client = await self._get_client()

        # Generate embeddings for memories that don't have them
        for memory in memories:
            if memory.embedding is None and self.embedding_function:
                memory.embedding = await self._generate_embedding(memory.content)

        # Prepare points
        points = []
        for memory in memories:
            if memory.embedding is None:
                logger.warning(f"Skipping memory {memory.id} without embedding")
                continue

            points.append(
                PointStruct(
                    id=memory.id,
                    vector=memory.embedding,
                    payload={
                        "content": memory.content,
                        "agent_id": memory.agent_id,
                        "session_id": memory.session_id,
                        "project_id": memory.project_id,
                        "memory_type": memory.memory_type.value,
                        "importance": memory.importance,
                        "access_count": memory.access_count,
                        "tags": memory.tags,
                        "timestamp": memory.timestamp.isoformat(),
                        **memory.metadata,
                    },
                )
            )

        try:
            # Batch upsert
            await client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info(f"Batch stored {len(points)} memories in Qdrant")
            return [p.id for p in points]

        except Exception as e:
            logger.error(f"Failed to batch store memories: {e}")
            raise

    async def get_memory(self, memory_id: str, **kwargs: Any) -> Optional[MemoryItem]:
        """
        Retrieve memory by ID.

        Args:
            memory_id: Memory identifier
            **kwargs: Additional parameters

        Returns:
            Memory item if found
        """
        client = await self._get_client()

        try:
            points = await client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
                with_vectors=True,
                with_payload=True,
            )

            if not points:
                return None

            point = points[0]
            return self._point_to_memory(point)

        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            raise

    async def search_memories(
        self, request: MemorySearchRequest, **kwargs: Any
    ) -> List[MemorySearchResult]:
        """
        Vector search for relevant memories.

        Args:
            request: Search request
            **kwargs: Additional parameters

        Returns:
            List of search results
        """
        client = await self._get_client()

        # If Qdrant is not available, return empty results
        if client is None:
            logger.debug("Qdrant not available, returning empty search results")
            return []

        # Generate query embedding
        if self.embedding_function is None:
            raise ValueError("embedding_function required for search")

        query_embedding = await self._generate_embedding(request.query)

        # Build filter
        filter_conditions = []
        for key, value in request.filters.items():
            filter_conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )

        search_filter = Filter(must=filter_conditions) if filter_conditions else None

        try:
            # Perform vector search
            results = await client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=search_filter,
                limit=request.k,
                score_threshold=request.score_threshold,
                with_vectors=True,
                with_payload=True,
            )

            # Convert to MemorySearchResult
            search_results = []
            # query_points returns QueryResponse with .points attribute
            points = results.points if hasattr(results, "points") else results
            for result in points:
                memory = self._point_to_memory(result)
                search_results.append(
                    MemorySearchResult(
                        memory=memory,
                        score=result.score,
                        source="project",
                    )
                )

            return search_results

        except Exception as e:
            logger.debug(f"Failed to search memories: {e}")
            return []  # Return empty list on failure

    async def search_with_filters(
        self,
        query_embedding: List[float],
        filters: Dict[str, Any],
        k: int = 5,
        **kwargs: Any,
    ) -> List[MemorySearchResult]:
        """
        Advanced search with custom filters.

        Args:
            query_embedding: Query vector
            filters: Metadata filters
            k: Number of results
            **kwargs: Additional parameters

        Returns:
            List of search results
        """
        client = await self._get_client()

        # Build filter
        filter_conditions = []
        for key, value in filters.items():
            filter_conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )

        search_filter = Filter(must=filter_conditions) if filter_conditions else None

        try:
            results = await client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=search_filter,
                limit=k,
                with_vectors=True,
                with_payload=True,
            )

            search_results = []
            # query_points returns QueryResponse with .points attribute
            points = results.points if hasattr(results, "points") else results
            for result in points:
                memory = self._point_to_memory(result)
                search_results.append(
                    MemorySearchResult(
                        memory=memory,
                        score=result.score,
                        source="project",
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Failed to search with filters: {e}")
            raise

    async def delete_memory(self, memory_id: str, **kwargs: Any) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory ID to delete
            **kwargs: Additional parameters

        Returns:
            True if deleted
        """
        client = await self._get_client()

        try:
            await client.delete(
                collection_name=self.collection_name,
                points_selector=[memory_id],
            )
            logger.debug(f"Deleted memory {memory_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise

    async def update_memory(
        self, memory_id: str, updates: Dict[str, Any], **kwargs: Any
    ) -> Optional[MemoryItem]:
        """
        Update memory metadata.

        Args:
            memory_id: Memory ID
            updates: Fields to update
            **kwargs: Additional parameters

        Returns:
            Updated memory
        """
        # Get existing memory
        memory = await self.get_memory(memory_id)
        if not memory:
            return None

        # Apply updates
        memory_dict = memory.model_dump()
        memory_dict.update(updates)
        updated_memory = MemoryItem(**memory_dict)

        # Re-add (upsert)
        await self.add_memory(updated_memory)
        return updated_memory

    async def clear_all(self, **kwargs: Any) -> int:
        """
        Clear all memories in collection.

        Args:
            **kwargs: Optional filters

        Returns:
            Number of memories cleared
        """
        client = await self._get_client()

        try:
            # Get count before deletion
            collection_info = await client.get_collection(self.collection_name)
            count = collection_info.points_count

            # Delete collection and recreate
            await client.delete_collection(self.collection_name)
            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.config.vector_size,
                    distance=Distance.COSINE,
                ),
            )

            logger.info(f"Cleared {count} memories from {self.collection_name}")
            return count

        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            raise

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.embedding_function is None:
            raise ValueError("No embedding function provided")

        if asyncio.iscoroutinefunction(self.embedding_function):
            return await self.embedding_function(text)
        else:
            return self.embedding_function(text)

    def _point_to_memory(self, point: Any) -> MemoryItem:
        """
        Convert Qdrant point to MemoryItem.

        Args:
            point: Qdrant point/result

        Returns:
            MemoryItem
        """
        payload = point.payload
        return MemoryItem(
            id=point.id,
            content=payload["content"],
            embedding=point.vector if hasattr(point, "vector") else None,
            metadata={
                k: v
                for k, v in payload.items()
                if k
                not in [
                    "content",
                    "agent_id",
                    "session_id",
                    "project_id",
                    "memory_type",
                    "importance",
                    "access_count",
                    "tags",
                    "timestamp",
                ]
            },
            memory_type=payload["memory_type"],
            agent_id=payload["agent_id"],
            session_id=payload.get("session_id"),
            project_id=payload.get("project_id"),
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            importance=payload["importance"],
            access_count=payload["access_count"],
            tags=payload["tags"],
        )

    async def close(self) -> None:
        """Close Qdrant client."""
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("Qdrant client closed")
