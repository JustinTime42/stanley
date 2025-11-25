"""Tests for context optimization."""

import pytest

from ...models.document_models import Chunk
from ...models.rag_models import RetrievalResult, ContextOptimization
from ...rag.context.context_builder import ContextBuilder
from ...rag.context.window_optimizer import WindowOptimizer


def test_window_optimizer_estimation():
    """Test token estimation."""
    optimizer = WindowOptimizer()

    text = "This is a test sentence with some words."
    tokens = optimizer.estimate_tokens(text)

    assert tokens > 0
    # Rough check: should be roughly len/4
    assert tokens >= len(text) // 5
    assert tokens <= len(text) // 3


def test_window_optimizer_truncate():
    """Test text truncation."""
    optimizer = WindowOptimizer()

    long_text = " ".join(["word"] * 100)
    truncated = optimizer.truncate_to_tokens(long_text, max_tokens=10)

    assert truncated is not None
    assert len(truncated) < len(long_text)
    assert truncated.endswith("...")


def test_window_optimizer_optimize():
    """Test window optimization."""
    optimizer = WindowOptimizer()

    texts = [
        "Short text one.",
        "Short text two.",
        "Short text three.",
        "Short text four.",
    ]

    optimized = optimizer.optimize_window(texts, max_tokens=20, reserve_tokens=5)

    assert len(optimized) > 0
    assert len(optimized) <= len(texts)


def test_context_builder_basic():
    """Test basic context building."""
    builder = ContextBuilder()

    # Create test chunks
    chunks = [
        Chunk(
            id="chunk1",
            document_id="doc1",
            content="First chunk content",
            start_index=0,
            end_index=18,
            chunk_index=0,
        ),
        Chunk(
            id="chunk2",
            document_id="doc1",
            content="Second chunk content",
            start_index=19,
            end_index=39,
            chunk_index=1,
        ),
    ]

    results = [
        RetrievalResult(chunk=chunks[0], score=0.9, search_type="test"),
        RetrievalResult(chunk=chunks[1], score=0.7, search_type="test"),
    ]

    context = builder.build_context(results, max_tokens=1000)

    assert len(context) > 0
    assert "First chunk content" in context
    assert "Second chunk content" in context


def test_context_builder_with_optimization():
    """Test context building with optimization."""
    builder = ContextBuilder()

    # Create test chunks with more content
    chunks = [
        Chunk(
            id=f"chunk{i}",
            document_id="doc1",
            content=f"This is chunk number {i} with substantial content that needs to be included in the context. It has multiple sentences and enough text to make compression worthwhile.",
            start_index=i*150,
            end_index=(i+1)*150,
            chunk_index=i,
        )
        for i in range(5)
    ]

    results = [
        RetrievalResult(chunk=chunk, score=0.9 - i*0.1, search_type="test")
        for i, chunk in enumerate(chunks)
    ]

    optimization = ContextOptimization(
        max_tokens=2000,  # More realistic token limit
        optimization_strategy="truncate",  # Use truncate for more predictable behavior
    )

    context = builder.build_context(results, max_tokens=2000, optimization=optimization)

    assert len(context) > 0
    assert "chunk number" in context


def test_context_builder_u_shaped_ordering():
    """Test U-shaped attention ordering."""
    builder = ContextBuilder()

    # Create chunks with different scores
    chunks = []
    for i in range(9):
        chunks.append(
            Chunk(
                id=f"chunk{i}",
                document_id="doc1",
                content=f"Content {i}",
                start_index=i*10,
                end_index=(i+1)*10,
                chunk_index=i,
            )
        )

    # Scores in descending order
    results = [
        RetrievalResult(chunk=chunks[i], score=0.9 - i*0.1, search_type="test")
        for i in range(9)
    ]

    # Apply U-shaped ordering
    ordered = builder._apply_u_shaped_ordering(results, "score")

    # First third should be highest scoring
    # Last third should be lower but present
    assert len(ordered) == len(results)
