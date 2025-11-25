"""Qdrant-based global memory for cross-project knowledge storage."""

import logging

from .project import QdrantProjectMemory
from ..config.memory_config import MemoryConfig

logger = logging.getLogger(__name__)


class QdrantGlobalMemory(QdrantProjectMemory):
    """
    Qdrant-based global memory for cross-project knowledge.

    Inherits from QdrantProjectMemory but uses a different collection
    and optionally enables on-disk storage for larger datasets.
    """

    def __init__(self, config: MemoryConfig, embedding_function=None):
        """
        Initialize Qdrant global memory.

        Args:
            config: Memory system configuration
            embedding_function: Function to generate embeddings
        """
        super().__init__(config, embedding_function)
        # Override collection name for global memory
        self.collection_name = config.global_collection_name

    async def _get_client(self):
        """
        Get Qdrant client and ensure global collection exists.

        Global memory collection is configured for larger datasets
        and may use on-disk storage.
        """
        if self._client is None:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import (
                Distance,
                VectorParams,
                OptimizersConfigDiff,
            )
            import asyncio

            # Create client
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
                        # Create collection optimized for large datasets
                        await self._client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(
                                size=self.config.vector_size,
                                distance=Distance.COSINE,
                                on_disk=True,  # Use disk storage for global memory
                            ),
                            optimizers_config=OptimizersConfigDiff(
                                indexing_threshold=10000,  # Optimize for larger datasets
                            ),
                        )
                        logger.info(
                            f"Created global Qdrant collection: {self.collection_name}"
                        )
                    else:
                        logger.info(
                            f"Using existing global collection: {self.collection_name}"
                        )
                    break

                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed to initialize global Qdrant after {max_retries} attempts: {e}"
                        )
                        raise
                    logger.warning(
                        f"Global Qdrant initialization attempt {attempt + 1} failed, retrying..."
                    )
                    await asyncio.sleep(1)

        return self._client
