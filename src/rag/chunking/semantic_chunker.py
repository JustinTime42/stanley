"""Semantic chunker using sentence embeddings for intelligent splitting."""

import logging
import re
from typing import List
import numpy as np

from ...models.document_models import Document, Chunk
from .base_chunker import BaseChunker

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """
    Chunk documents by semantic similarity between sentences.

    PATTERN: Sliding window with semantic boundary detection
    CRITICAL: Maintain context at chunk boundaries with overlap
    """

    def __init__(
        self,
        llm_service=None,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize semantic chunker.

        Args:
            llm_service: LLM service for embeddings (optional for now)
            similarity_threshold: Cosine similarity threshold for splitting
        """
        self.llm_service = llm_service
        self.similarity_threshold = similarity_threshold
        self.logger = logger

    async def chunk_document(
        self,
        document: Document,
        target_chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[Chunk]:
        """
        Chunk document by semantic similarity.

        PATTERN: Split into sentences, group by similarity
        GOTCHA: Must handle sentence boundaries carefully

        Args:
            document: Document to chunk
            target_chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens

        Returns:
            List of semantically coherent chunks
        """
        # Split into sentences
        sentences = self._split_sentences(document.content)

        if not sentences:
            return []

        # For now, use simple chunking if no LLM service
        # TODO: Generate embeddings when LLM service available
        if not self.llm_service:
            return await self._simple_chunk(
                document, sentences, target_chunk_size, overlap
            )

        # Generate embeddings for sentences
        try:
            embeddings = await self._generate_sentence_embeddings(sentences)
        except Exception as e:
            self.logger.warning(
                f"Failed to generate embeddings, using simple chunking: {e}"
            )
            return await self._simple_chunk(
                document, sentences, target_chunk_size, overlap
            )

        chunks = []
        current_chunk = []
        current_tokens = 0
        start_index = 0

        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            # Add sentence to current chunk
            current_chunk.append(sentence)
            sentence_tokens = self._estimate_tokens(sentence)
            current_tokens += sentence_tokens

            # Check if we should start new chunk
            should_split = False

            # Size-based split
            if current_tokens >= target_chunk_size:
                should_split = True

            # Semantic boundary split
            elif i < len(sentences) - 1:
                next_embedding = embeddings[i + 1]
                similarity = self._cosine_similarity(embedding, next_embedding)
                if similarity < self.similarity_threshold:
                    should_split = True

            if should_split or i == len(sentences) - 1:
                # Create chunk
                chunk_content = " ".join(current_chunk)
                chunk_id = f"{document.id}_chunk_{len(chunks)}"

                chunk = Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=chunk_content,
                    start_index=start_index,
                    end_index=start_index + len(chunk_content),
                    chunk_index=len(chunks),
                    token_count=current_tokens,
                    metadata=document.metadata.copy(),
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                if overlap > 0 and i < len(sentences) - 1:
                    # Calculate how many sentences to overlap
                    overlap_sentences = []
                    overlap_tokens = 0
                    for sent in reversed(current_chunk):
                        sent_tokens = self._estimate_tokens(sent)
                        if overlap_tokens + sent_tokens > overlap:
                            break
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens

                    current_chunk = overlap_sentences
                    current_tokens = overlap_tokens
                else:
                    current_chunk = []
                    current_tokens = 0

                start_index += len(chunk_content) + 1

        self.logger.info(
            f"Semantically chunked document {document.id} into {len(chunks)} chunks"
        )
        return chunks

    async def _simple_chunk(
        self,
        document: Document,
        sentences: List[str],
        target_chunk_size: int,
        overlap: int,
    ) -> List[Chunk]:
        """
        Simple chunking by sentence count without embeddings.

        Args:
            document: Document to chunk
            sentences: List of sentences
            target_chunk_size: Target size in tokens
            overlap: Overlap in tokens

        Returns:
            List of chunks
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        start_index = 0

        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            sentence_tokens = self._estimate_tokens(sentence)
            current_tokens += sentence_tokens

            # Split when we reach target size
            if current_tokens >= target_chunk_size or i == len(sentences) - 1:
                chunk_content = " ".join(current_chunk)
                chunk_id = f"{document.id}_chunk_{len(chunks)}"

                chunk = Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    content=chunk_content,
                    start_index=start_index,
                    end_index=start_index + len(chunk_content),
                    chunk_index=len(chunks),
                    token_count=current_tokens,
                    metadata=document.metadata.copy(),
                )
                chunks.append(chunk)

                # Calculate overlap
                if overlap > 0 and i < len(sentences) - 1:
                    overlap_sentences = []
                    overlap_tokens = 0
                    for sent in reversed(current_chunk):
                        sent_tokens = self._estimate_tokens(sent)
                        if overlap_tokens + sent_tokens > overlap:
                            break
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens

                    current_chunk = overlap_sentences
                    current_tokens = overlap_tokens
                else:
                    current_chunk = []
                    current_tokens = 0

                start_index += len(chunk_content) + 1

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        PATTERN: Use regex for sentence boundary detection
        GOTCHA: Handle abbreviations and edge cases

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting with regex
        # Handles periods, question marks, exclamation marks
        sentence_pattern = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s")

        sentences = sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    async def _generate_sentence_embeddings(
        self, sentences: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for sentences.

        Args:
            sentences: List of sentences

        Returns:
            List of embedding vectors
        """
        if not self.llm_service:
            raise ValueError("LLM service required for embeddings")

        # Generate embeddings (batch if possible)
        # TODO: Implement batching for efficiency
        embeddings = []
        for sentence in sentences:
            # Use LLM service to generate embedding
            embedding = await self.llm_service.generate_embedding(sentence)
            embeddings.append(embedding)

        return embeddings

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        PATTERN: Rough estimation (1 token ≈ 4 characters)
        TODO: Use tiktoken for accurate counting

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token
        return max(1, len(text) // 4)
