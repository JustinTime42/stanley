"""Unit tests for hybrid search functionality."""

import pytest
from unittest.mock import AsyncMock

from src.memory.hybrid import HybridSearchManager, BM25KeywordSearch
from src.models.memory_models import (
    MemoryItem,
    MemorySearchRequest,
    MemorySearchResult,
    MemoryType,
)


@pytest.fixture
def sample_memories():
    """Create sample memory items."""
    return [
        MemoryItem(
            id="mem-1",
            content="Python programming language is great for data science",
            agent_id="agent-1",
            memory_type=MemoryType.PROJECT,
            tags=["python", "data"],
        ),
        MemoryItem(
            id="mem-2",
            content="JavaScript is essential for web development",
            agent_id="agent-1",
            memory_type=MemoryType.PROJECT,
            tags=["javascript", "web"],
        ),
        MemoryItem(
            id="mem-3",
            content="Python and JavaScript are popular programming languages",
            agent_id="agent-1",
            memory_type=MemoryType.PROJECT,
            tags=["python", "javascript"],
        ),
    ]


@pytest.fixture
def sample_results(sample_memories):
    """Create sample search results."""
    return [
        MemorySearchResult(
            memory=sample_memories[0],
            score=0.95,
            source="vector",
        ),
        MemorySearchResult(
            memory=sample_memories[2],
            score=0.85,
            source="vector",
        ),
    ]


@pytest.mark.asyncio
class TestHybridSearchManager:
    """Test suite for HybridSearchManager."""

    async def test_init(self):
        """Test initialization."""
        vector_fn = AsyncMock()
        keyword_fn = AsyncMock()

        manager = HybridSearchManager(
            vector_search_fn=vector_fn,
            keyword_search_fn=keyword_fn,
            k=60,
        )

        assert manager.vector_search_fn == vector_fn
        assert manager.keyword_search_fn == keyword_fn
        assert manager.rrf_k == 60

    async def test_hybrid_search_vector_only(self, sample_results):
        """Test hybrid search with alpha=1.0 (vector only)."""
        vector_fn = AsyncMock(return_value=sample_results)
        keyword_fn = AsyncMock()

        manager = HybridSearchManager(
            vector_search_fn=vector_fn,
            keyword_search_fn=keyword_fn,
        )

        request = MemorySearchRequest(
            query="test query",
            k=5,
            alpha=1.0,  # Vector only
        )

        results = await manager.hybrid_search(request)

        # Verify
        assert results == sample_results
        vector_fn.assert_called_once()
        keyword_fn.assert_not_called()

    async def test_hybrid_search_keyword_only(self, sample_results):
        """Test hybrid search with alpha=0.0 (keyword only)."""
        vector_fn = AsyncMock()
        keyword_fn = AsyncMock(return_value=sample_results)

        manager = HybridSearchManager(
            vector_search_fn=vector_fn,
            keyword_search_fn=keyword_fn,
        )

        request = MemorySearchRequest(
            query="test query",
            k=5,
            alpha=0.0,  # Keyword only
        )

        results = await manager.hybrid_search(request)

        # Verify
        assert results == sample_results
        vector_fn.assert_not_called()
        keyword_fn.assert_called_once()

    async def test_hybrid_search_combined(self, sample_memories):
        """Test hybrid search combining both methods."""
        # Create different results from vector and keyword search
        vector_results = [
            MemorySearchResult(
                memory=sample_memories[0],
                score=0.95,
                source="vector",
            ),
            MemorySearchResult(
                memory=sample_memories[1],
                score=0.85,
                source="vector",
            ),
        ]

        keyword_results = [
            MemorySearchResult(
                memory=sample_memories[0],  # Same as vector (should be combined)
                score=0.90,
                source="keyword",
            ),
            MemorySearchResult(
                memory=sample_memories[2],
                score=0.80,
                source="keyword",
            ),
        ]

        vector_fn = AsyncMock(return_value=vector_results)
        keyword_fn = AsyncMock(return_value=keyword_results)

        manager = HybridSearchManager(
            vector_search_fn=vector_fn,
            keyword_search_fn=keyword_fn,
        )

        request = MemorySearchRequest(
            query="test query",
            k=5,
            alpha=0.7,  # Balanced
        )

        results = await manager.hybrid_search(request)

        # Verify both searches were called
        vector_fn.assert_called_once()
        keyword_fn.assert_called_once()

        # Verify results are combined and sorted
        assert len(results) > 0
        assert all(r.source == "hybrid" for r in results)

        # First result should be mem-1 (appears in both searches)
        assert results[0].memory.id == "mem-1"

    async def test_reciprocal_rank_fusion(self, sample_memories):
        """Test RRF score calculation."""
        vector_results = [
            MemorySearchResult(
                memory=sample_memories[0],
                score=0.95,
                source="vector",
            ),
        ]

        keyword_results = [
            MemorySearchResult(
                memory=sample_memories[0],
                score=0.90,
                source="keyword",
            ),
        ]

        manager = HybridSearchManager(
            vector_search_fn=AsyncMock(),
            keyword_search_fn=AsyncMock(),
            k=60,
        )

        # Test RRF fusion
        results = manager._reciprocal_rank_fusion(
            vector_results=vector_results,
            keyword_results=keyword_results,
            alpha=0.5,
        )

        # Verify combined score
        assert len(results) == 1
        assert results[0].source == "hybrid"
        # RRF score should be: 0.5 * (1/(60+1)) + 0.5 * (1/(60+1))
        expected_score = 0.5 * (1 / 61) + 0.5 * (1 / 61)
        assert abs(results[0].score - expected_score) < 0.001


class TestBM25KeywordSearch:
    """Test suite for BM25KeywordSearch."""

    def test_init(self):
        """Test initialization."""
        bm25 = BM25KeywordSearch(k1=1.5, b=0.75)
        assert bm25.k1 == 1.5
        assert bm25.b == 0.75

    def test_tokenize(self, sample_memories):
        """Test text tokenization."""
        bm25 = BM25KeywordSearch()

        # Test basic tokenization
        tokens = bm25._tokenize("Hello World, this is a test!")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

        # Test empty string
        tokens = bm25._tokenize("")
        assert tokens == []

    @pytest.mark.asyncio
    async def test_search(self, sample_memories):
        """Test BM25 search."""
        bm25 = BM25KeywordSearch()

        # Search for "python"
        results = await bm25.search(
            query="python programming",
            documents=sample_memories,
            k=3,
        )

        # Verify results
        assert len(results) > 0
        assert all(isinstance(r, MemorySearchResult) for r in results)
        assert all(r.source == "keyword" for r in results)

        # First result should contain "python"
        assert "python" in results[0].memory.content.lower()

    @pytest.mark.asyncio
    async def test_search_no_results(self, sample_memories):
        """Test BM25 search with no matching documents."""
        bm25 = BM25KeywordSearch()

        # Search for term not in documents
        results = await bm25.search(
            query="quantum mechanics",
            documents=sample_memories,
            k=3,
        )

        # Verify no results
        assert len(results) == 0

    def test_calculate_idf(self):
        """Test IDF calculation."""
        bm25 = BM25KeywordSearch()

        # Common term (appears in many docs)
        idf_common = bm25._calculate_idf(df=8, total_docs=10)

        # Rare term (appears in few docs)
        idf_rare = bm25._calculate_idf(df=1, total_docs=10)

        # Rare terms should have higher IDF
        assert idf_rare > idf_common

    def test_extract_highlights(self, sample_memories):
        """Test highlight extraction."""
        bm25 = BM25KeywordSearch()

        query_terms = ["python", "programming"]
        content = sample_memories[0].content

        highlights = bm25._extract_highlights(
            query_terms=query_terms,
            content=content,
            context_words=3,
        )

        # Verify highlights
        assert len(highlights) > 0
        assert all("..." in h for h in highlights)
        assert any("python" in h.lower() for h in highlights)

    @pytest.mark.asyncio
    async def test_search_with_highlights(self, sample_memories):
        """Test that search results include highlights."""
        bm25 = BM25KeywordSearch()

        results = await bm25.search(
            query="python data science",
            documents=sample_memories,
            k=3,
        )

        # Verify results have highlights
        if len(results) > 0:
            assert results[0].highlights is not None
            assert len(results[0].highlights) > 0
