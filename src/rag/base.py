"""Base RAG abstract class defining RAG interface."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models.document_models import Document
from ..models.rag_models import (
    RetrievalRequest,
    RetrievalResult,
    IngestionRequest,
)


class BaseRAG(ABC):
    """
    Abstract base class for RAG implementations.

    PATTERN: Define common interface for all RAG operations
    CRITICAL: All implementations must be async-compatible
    """

    @abstractmethod
    async def ingest(
        self,
        request: IngestionRequest,
    ) -> Document:
        """
        Ingest a document or directory.

        Args:
            request: Ingestion request with source and options

        Returns:
            Processed document with chunks
        """
        pass

    @abstractmethod
    async def retrieve(
        self,
        request: RetrievalRequest,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            request: Retrieval request with query and options

        Returns:
            List of retrieved and ranked chunks
        """
        pass

    @abstractmethod
    async def generate(
        self,
        query: str,
        context: str,
        **kwargs,
    ) -> str:
        """
        Generate response using retrieved context.

        Args:
            query: User query
            context: Retrieved context string
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        pass

    @abstractmethod
    async def retrieve_and_generate(
        self,
        query: str,
        retrieval_request: Optional[RetrievalRequest] = None,
        **kwargs,
    ) -> str:
        """
        End-to-end retrieval and generation.

        Args:
            query: User query
            retrieval_request: Optional retrieval configuration
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        pass
