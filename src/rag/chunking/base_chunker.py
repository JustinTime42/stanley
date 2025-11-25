"""Base abstract chunker class."""

from abc import ABC, abstractmethod
from typing import List

from ...models.document_models import Document, Chunk


class BaseChunker(ABC):
    """
    Abstract base class for document chunking strategies.

    PATTERN: Strategy pattern for different chunking approaches
    """

    @abstractmethod
    async def chunk_document(
        self,
        document: Document,
        target_chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[Chunk]:
        """
        Chunk a document into smaller pieces.

        Args:
            document: Document to chunk
            target_chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens

        Returns:
            List of document chunks
        """
        pass
