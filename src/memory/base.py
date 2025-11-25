"""Abstract base class for memory implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ..models.memory_models import MemoryItem, MemorySearchRequest, MemorySearchResult


class BaseMemory(ABC):
    """Abstract base class defining memory interface."""

    @abstractmethod
    async def add_memory(self, memory: MemoryItem, **kwargs: Any) -> str:
        """
        Add a memory item to storage.

        Args:
            memory: Memory item to store
            **kwargs: Additional storage-specific parameters

        Returns:
            Memory ID of stored item
        """
        pass

    @abstractmethod
    async def get_memory(self, memory_id: str, **kwargs: Any) -> Optional[MemoryItem]:
        """
        Retrieve a specific memory by ID.

        Args:
            memory_id: Unique memory identifier
            **kwargs: Additional retrieval parameters

        Returns:
            Memory item if found, None otherwise
        """
        pass

    @abstractmethod
    async def search_memories(
        self, request: MemorySearchRequest, **kwargs: Any
    ) -> List[MemorySearchResult]:
        """
        Search for relevant memories.

        Args:
            request: Search request parameters
            **kwargs: Additional search parameters

        Returns:
            List of search results with scores
        """
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: str, **kwargs: Any) -> bool:
        """
        Delete a memory item.

        Args:
            memory_id: Memory ID to delete
            **kwargs: Additional deletion parameters

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def update_memory(
        self, memory_id: str, updates: Dict[str, Any], **kwargs: Any
    ) -> Optional[MemoryItem]:
        """
        Update a memory item.

        Args:
            memory_id: Memory ID to update
            updates: Fields to update
            **kwargs: Additional update parameters

        Returns:
            Updated memory item if found, None otherwise
        """
        pass

    @abstractmethod
    async def clear_all(self, **kwargs: Any) -> int:
        """
        Clear all memories (use with caution).

        Args:
            **kwargs: Additional parameters (e.g., filters)

        Returns:
            Number of memories cleared
        """
        pass
