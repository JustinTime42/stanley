"""Document management service using RAG pipeline."""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..models.document_models import Document, DocumentType, ChunkingStrategy
from ..models.rag_models import IngestionRequest, RetrievalRequest, RetrievalResult
from ..rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class DocumentService:
    """
    High-level service for document management.

    PATTERN: Service facade over RAG pipeline
    CRITICAL: Single entry point for document operations
    """

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        memory_service=None,
        llm_service=None,
    ):
        """
        Initialize document service.

        Args:
            rag_pipeline: Optional RAG pipeline (creates if None)
            memory_service: Memory service for storage
            llm_service: LLM service for embeddings/generation
        """
        self.rag_pipeline = rag_pipeline or RAGPipeline(
            memory_service=memory_service,
            llm_service=llm_service,
        )
        self.logger = logger

    async def ingest_file(
        self,
        file_path: str,
        document_type: Optional[DocumentType] = None,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        generate_embeddings: bool = True,
    ) -> Document:
        """
        Ingest a single file.

        Args:
            file_path: Path to file
            document_type: Optional document type override
            chunking_strategy: Chunking strategy to use
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
            generate_embeddings: Whether to generate embeddings

        Returns:
            Processed document

        Raises:
            ValueError: If file not found or processing fails
        """
        if not Path(file_path).exists():
            raise ValueError(f"File not found: {file_path}")

        request = IngestionRequest(
            source_path=file_path,
            document_type=document_type,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            generate_embeddings=generate_embeddings,
        )

        document = await self.rag_pipeline.ingest(request)

        self.logger.info(f"Ingested file: {file_path} -> {len(document.chunks)} chunks")

        return document

    async def ingest_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        chunk_size: int = 500,
        generate_embeddings: bool = True,
    ) -> List[Document]:
        """
        Ingest all files from a directory.

        Args:
            directory_path: Path to directory
            recursive: Process subdirectories
            chunking_strategy: Chunking strategy
            chunk_size: Target chunk size
            generate_embeddings: Whether to generate embeddings

        Returns:
            List of processed documents
        """
        if not Path(directory_path).exists():
            raise ValueError(f"Directory not found: {directory_path}")

        request = IngestionRequest(
            source_path=directory_path,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            generate_embeddings=generate_embeddings,
            recursive=recursive,
        )

        # Ingest will handle directory processing
        await self.rag_pipeline.ingest(request)

        # Get all documents
        documents = list(self.rag_pipeline.documents.values())

        self.logger.info(
            f"Ingested directory: {directory_path} -> {len(documents)} documents"
        )

        return documents

    async def search(
        self,
        query: str,
        k: int = 5,
        document_types: Optional[List[DocumentType]] = None,
        use_hybrid: bool = True,
        use_reranking: bool = True,
        max_tokens: int = 4000,
    ) -> List[RetrievalResult]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            k: Number of results
            document_types: Filter by document types
            use_hybrid: Use hybrid search
            use_reranking: Apply re-ranking
            max_tokens: Maximum context tokens

        Returns:
            Ranked search results
        """
        request = RetrievalRequest(
            query=query,
            k=k,
            document_types=document_types,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking,
            max_tokens=max_tokens,
        )

        results = await self.rag_pipeline.retrieve(request)

        self.logger.info(f"Search for '{query}' returned {len(results)} results")

        return results

    async def search_and_generate(
        self,
        query: str,
        k: int = 5,
        max_tokens: int = 4000,
        **generation_kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Search for relevant context and generate response.

        Args:
            query: User query
            k: Number of chunks to retrieve
            max_tokens: Maximum context tokens
            **generation_kwargs: Additional LLM parameters

        Returns:
            Dictionary with response and metadata
        """
        # Retrieve
        request = RetrievalRequest(
            query=query,
            k=k,
            max_tokens=max_tokens,
        )
        results = await self.rag_pipeline.retrieve(request)

        # Build context
        from ..models.rag_models import ContextOptimization

        optimization = ContextOptimization(max_tokens=max_tokens)
        context = self.rag_pipeline.context_builder.build_context(
            results,
            max_tokens,
            optimization,
        )

        # Generate
        response = await self.rag_pipeline.generate(
            query,
            context,
            **generation_kwargs,
        )

        return {
            "response": response,
            "context": context,
            "chunks_used": len(results),
            "chunk_sources": [
                r.chunk.metadata.get("file_name", r.chunk.document_id) for r in results
            ],
        }

    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Get document by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document or None if not found
        """
        return self.rag_pipeline.documents.get(document_id)

    def list_documents(self) -> List[Document]:
        """
        Get all documents.

        Returns:
            List of all documents
        """
        return list(self.rag_pipeline.documents.values())

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Statistics dictionary
        """
        documents = list(self.rag_pipeline.documents.values())
        chunks = list(self.rag_pipeline.chunks.values())

        # Count by type
        doc_types = {}
        for doc in documents:
            doc_types[doc.type.value] = doc_types.get(doc.type.value, 0) + 1

        # Count embeddings
        chunks_with_embeddings = sum(1 for c in chunks if c.embedding is not None)

        return {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "documents_by_type": doc_types,
            "chunks_with_embeddings": chunks_with_embeddings,
            "average_chunks_per_document": (
                len(chunks) / len(documents) if documents else 0
            ),
        }
