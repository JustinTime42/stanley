"""Memory subsystem for hierarchical memory management."""

from .base import BaseMemory
from .working import RedisWorkingMemory
from .project import QdrantProjectMemory
from .global_memory import QdrantGlobalMemory
from .hybrid import HybridSearchManager, BM25KeywordSearch
from .cache import SemanticCache
from ..config.memory_config import MemoryConfig

__all__ = [
    "BaseMemory",
    "RedisWorkingMemory",
    "QdrantProjectMemory",
    "QdrantGlobalMemory",
    "HybridSearchManager",
    "BM25KeywordSearch",
    "SemanticCache",
    "MemoryConfig",
]


def create_memory_manager(
    config: MemoryConfig,
    embedding_function=None,
):
    """
    Factory function to create memory managers.

    Args:
        config: Memory configuration
        embedding_function: Optional embedding function for vector stores

    Returns:
        Dictionary with memory tier instances
    """
    return {
        "working": RedisWorkingMemory(config=config),
        "project": QdrantProjectMemory(
            config=config, embedding_function=embedding_function
        ),
        "global": QdrantGlobalMemory(
            config=config, embedding_function=embedding_function
        ),
        "cache": SemanticCache(
            embedding_function=embedding_function,
            max_size=config.memory_cache_size,
            default_ttl=config.memory_ttl_working,
        ),
    }
