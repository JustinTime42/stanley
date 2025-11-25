"""RAG pipeline orchestrating all RAG components."""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..models.document_models import Document, Chunk, DocumentType, ChunkingStrategy
from ..models.rag_models import (
    RetrievalRequest,
    RetrievalResult,
    IngestionRequest,
    ContextOptimization,
)
from .base import BaseRAG
from .ingestion.document_loader import DocumentLoader
from .chunking.semantic_chunker import SemanticChunker
from .chunking.code_chunker import CodeChunker
from .retrieval.query_analyzer import QueryAnalyzer
from .retrieval.relevance_scorer import RelevanceScorer
from .retrieval.structural_search import StructuralSearch
from .context.context_builder import ContextBuilder
from .context.window_optimizer import WindowOptimizer

logger = logging.getLogger(__name__)


class RAGPipeline(BaseRAG):
    """
    End-to-end RAG pipeline orchestrating all components.

    PATTERN: Pipeline pattern with configurable stages
    CRITICAL: Coordinates ingestion, chunking, retrieval, and generation
    """

    def __init__(
        self,
        memory_service=None,
        llm_service=None,
    ):
        """
        Initialize RAG pipeline.

        Args:
            memory_service: Memory service for storing/retrieving chunks
            llm_service: LLM service for embeddings and generation
        """
        self.memory_service = memory_service
        self.llm_service = llm_service
        self.logger = logger

        # Initialize components
        self.document_loader = DocumentLoader()
        self.semantic_chunker = SemanticChunker(llm_service)
        self.code_chunker = CodeChunker()
        self.query_analyzer = QueryAnalyzer()
        self.relevance_scorer = RelevanceScorer()
        self.structural_search = StructuralSearch()
        self.context_builder = ContextBuilder()
        self.window_optimizer = WindowOptimizer()

        # Document and chunk storage
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}

        self.logger.info("RAG Pipeline initialized")

    async def ingest(
        self,
        request: IngestionRequest,
    ) -> Document:
        """
        Ingest a document or directory.

        PATTERN: Load -> Chunk -> Generate Embeddings -> Store
        CRITICAL: All operations should be async for large files

        Args:
            request: Ingestion request

        Returns:
            Processed document with chunks
        """
        source_path = Path(request.source_path)

        # Handle directory vs file
        if source_path.is_dir():
            documents = await self._ingest_directory(request)
            # Return first document (or create summary document)
            return documents[0] if documents else None

        # Load document
        document = await self.document_loader.load_document(
            str(source_path),
            request.document_type,
        )

        if not document:
            raise ValueError(f"Failed to load document: {source_path}")

        # Chunk document
        chunks = await self._chunk_document(
            document,
            request.chunking_strategy,
            request.chunk_size,
            request.chunk_overlap,
        )

        document.chunks = chunks

        # Extract keywords for chunks
        for chunk in chunks:
            chunk.keywords = self._extract_keywords(chunk.content)

        # Generate embeddings if requested
        if request.generate_embeddings and self.llm_service:
            await self._generate_embeddings(chunks)
            document.embeddings_generated = True

        document.processed = True

        # Store in memory
        self.documents[document.id] = document
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

        # Store in vector database if memory service available
        if self.memory_service:
            await self._store_chunks_in_memory(chunks, document)

        self.logger.info(
            f"Ingested document {document.source}: "
            f"{len(chunks)} chunks, "
            f"embeddings={document.embeddings_generated}"
        )

        return document

    async def retrieve(
        self,
        request: RetrievalRequest,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.

        PATTERN: Analyze -> Search -> Score -> Rerank -> Filter
        CRITICAL: Combine multiple search strategies

        Args:
            request: Retrieval request

        Returns:
            Ranked retrieval results
        """
        # Analyze query
        query_analysis = self.query_analyzer.analyze_query(request.query)

        self.logger.debug(
            f"Query analysis: intent={query_analysis.intent.value}, "
            f"requires_code={query_analysis.requires_code}"
        )

        # Get candidate chunks
        if self.memory_service and request.use_hybrid:
            # Use hybrid search from memory service
            candidates = await self._hybrid_search(request, query_analysis)
        else:
            # Use simple search over stored chunks
            candidates = await self._simple_search(request.query, request.k)

        # Apply relevance scoring
        scored_results = []
        for chunk, base_score in candidates:
            final_score = self.relevance_scorer.score_chunk(
                chunk,
                query_analysis,
                base_score,
            )
            result = RetrievalResult(
                chunk=chunk,
                score=final_score,
                search_type="hybrid" if request.use_hybrid else "simple",
            )
            scored_results.append(result)

        # Re-rank if requested
        if request.use_reranking:
            scored_pairs = [(r.chunk, r.score) for r in scored_results]
            reranked = self.relevance_scorer.rerank_results(
                scored_pairs,
                query_analysis,
                request.k,
            )
            results = [
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    search_type="reranked",
                )
                for chunk, score in reranked
            ]
        else:
            # Sort by score and take top k
            results = sorted(scored_results, key=lambda x: x.score, reverse=True)[
                : request.k
            ]

        self.logger.info(f"Retrieved {len(results)} chunks for query")

        return results

    async def generate(
        self,
        query: str,
        context: str,
        **kwargs: Any,
    ) -> str:
        """
        Generate response using retrieved context.

        Args:
            query: User query
            context: Retrieved context
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        if not self.llm_service:
            return "No LLM service configured"

        # Build prompt
        prompt = f"""Based on the following context, please answer the query.

CONTEXT:
{context}

QUERY:
{query}

ANSWER:"""

        # Generate response
        try:
            # Assuming llm_service has a generate method
            response = await self.llm_service.generate_response(prompt, **kwargs)
            return (
                response.get("content", "")
                if isinstance(response, dict)
                else str(response)
            )
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return f"Error generating response: {e}"

    async def retrieve_and_generate(
        self,
        query: str,
        retrieval_request: Optional[RetrievalRequest] = None,
        **kwargs: Any,
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
        # Create default retrieval request if not provided
        if retrieval_request is None:
            retrieval_request = RetrievalRequest(query=query)

        # Retrieve relevant chunks
        results = await self.retrieve(retrieval_request)

        # Build context
        optimization = ContextOptimization(
            max_tokens=retrieval_request.max_tokens,
        )
        context = self.context_builder.build_context(
            results,
            retrieval_request.max_tokens,
            optimization,
        )

        # Generate response
        response = await self.generate(query, context, **kwargs)

        return response

    async def _ingest_directory(
        self,
        request: IngestionRequest,
    ) -> List[Document]:
        """Ingest all documents from a directory."""
        documents = await self.document_loader.load_directory(
            request.source_path,
            request.recursive,
        )

        processed_documents = []
        for doc in documents:
            # Process each document
            chunks = await self._chunk_document(
                doc,
                request.chunking_strategy,
                request.chunk_size,
                request.chunk_overlap,
            )
            doc.chunks = chunks
            doc.processed = True

            # Extract keywords
            for chunk in chunks:
                chunk.keywords = self._extract_keywords(chunk.content)

            # Generate embeddings
            if request.generate_embeddings and self.llm_service:
                await self._generate_embeddings(chunks)
                doc.embeddings_generated = True

            # Store
            self.documents[doc.id] = doc
            for chunk in chunks:
                self.chunks[chunk.id] = chunk

            if self.memory_service:
                await self._store_chunks_in_memory(chunks, doc)

            processed_documents.append(doc)

        self.logger.info(
            f"Ingested {len(processed_documents)} documents from directory"
        )
        return processed_documents

    async def _chunk_document(
        self,
        document: Document,
        strategy: ChunkingStrategy,
        chunk_size: int,
        overlap: int,
    ) -> List[Chunk]:
        """Chunk document using specified strategy."""
        if (
            strategy == ChunkingStrategy.CODE_AWARE
            and document.type == DocumentType.CODE
        ):
            chunker = self.code_chunker
        elif strategy == ChunkingStrategy.SEMANTIC:
            chunker = self.semantic_chunker
        else:
            # Default to semantic chunker
            chunker = self.semantic_chunker

        chunks = await chunker.chunk_document(document, chunk_size, overlap)
        return chunks

    async def _generate_embeddings(self, chunks: List[Chunk]) -> None:
        """Generate embeddings for chunks."""
        if not self.llm_service:
            return

        for chunk in chunks:
            try:
                # Generate embedding
                # TODO: Batch embeddings for efficiency
                embedding = await self.llm_service.generate_embedding(chunk.content)
                chunk.embedding = embedding
            except Exception as e:
                self.logger.warning(
                    f"Failed to generate embedding for chunk {chunk.id}: {e}"
                )

    async def _store_chunks_in_memory(
        self,
        chunks: List[Chunk],
        document: Document,
    ) -> None:
        """Store chunks in memory service."""
        # TODO: Implement memory service integration
        # This would store chunks in vector database
        pass

    async def _hybrid_search(
        self,
        request: RetrievalRequest,
        query_analysis,
    ) -> List[tuple[Chunk, float]]:
        """Perform hybrid search using memory service."""
        # TODO: Implement memory service integration
        # For now, fall back to simple search
        return await self._simple_search(request.query, request.k)

    async def _simple_search(
        self,
        query: str,
        k: int,
    ) -> List[tuple[Chunk, float]]:
        """Simple keyword-based search over stored chunks."""
        query_lower = query.lower()
        query_keywords = set(self._extract_keywords(query))

        results = []
        for chunk in self.chunks.values():
            # Calculate keyword overlap score
            chunk_keywords = set(chunk.keywords)
            if not chunk_keywords:
                score = 0.0
            else:
                overlap = len(query_keywords & chunk_keywords)
                score = overlap / max(len(query_keywords), len(chunk_keywords))

            # Boost if query appears in content
            if query_lower in chunk.content.lower():
                score += 0.5

            results.append((chunk, min(1.0, score)))

        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[: k * 2]  # Return more candidates for reranking

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        import re

        # Simple keyword extraction
        words = re.findall(r"\b\w+\b", text.lower())
        # Filter short words and common words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
        }
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]

        # Return unique keywords
        return list(set(keywords))[:50]  # Limit to 50 keywords
