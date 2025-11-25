"""Tests for knowledge system."""

import pytest
from datetime import datetime


class TestKnowledgeStore:
    """Tests for KnowledgeStore."""

    @pytest.fixture
    def knowledge_store(self):
        """Create a knowledge store for testing."""
        from src.understanding.knowledge import KnowledgeStore

        return KnowledgeStore()

    @pytest.fixture
    def sample_symbols(self):
        """Create sample symbols."""
        from src.understanding.models import Symbol, SymbolKind, ConfidenceLevel

        return [
            Symbol(
                id="sym-1",
                name="foo",
                qualified_name="module.foo",
                kind=SymbolKind.FUNCTION,
                file_path="/path/to/file.py",
                line_start=10,
                line_end=20,
                signature="def foo(x: int) -> str",
                docstring="Does foo things.",
                parameters=[{"name": "x", "type": "int"}],
                return_type="str",
                content_hash="abc123",
                confidence=ConfidenceLevel.VERIFIED,
            ),
            Symbol(
                id="sym-2",
                name="Bar",
                qualified_name="module.Bar",
                kind=SymbolKind.CLASS,
                file_path="/path/to/file.py",
                line_start=25,
                line_end=50,
                signature="class Bar",
                docstring="A Bar class.",
                content_hash="def456",
                confidence=ConfidenceLevel.VERIFIED,
            ),
        ]

    @pytest.mark.asyncio
    async def test_get_symbol(self, knowledge_store, sample_symbols):
        """Test getting a symbol by name."""
        # Store symbols manually
        for sym in sample_symbols:
            knowledge_store._symbols[sym.id] = sym
            knowledge_store._symbol_by_name[sym.name] = sym.id
            knowledge_store._symbol_by_qualified_name[sym.qualified_name] = sym.id

        # Test simple name lookup
        result = await knowledge_store.get_symbol("foo")
        assert result is not None
        assert result.name == "foo"

        # Test qualified name lookup
        result = await knowledge_store.get_symbol("module.Bar")
        assert result is not None
        assert result.name == "Bar"

        # Test non-existent
        result = await knowledge_store.get_symbol("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_similar_symbols(self, knowledge_store, sample_symbols):
        """Test finding similar symbols."""
        for sym in sample_symbols:
            knowledge_store._symbols[sym.id] = sym
            knowledge_store._symbol_by_name[sym.name] = sym.id

        # Find similar to "foo"
        similar = await knowledge_store.find_similar_symbols("fo", limit=5)
        assert len(similar) >= 1
        assert any(s.name == "foo" for s in similar)

    def test_get_all_symbols(self, knowledge_store, sample_symbols):
        """Test getting all symbols."""
        for sym in sample_symbols:
            knowledge_store._symbols[sym.id] = sym

        all_symbols = knowledge_store.get_all_symbols()
        assert len(all_symbols) == 2


class TestKnowledgeVerifier:
    """Tests for KnowledgeVerifier."""

    @pytest.fixture
    def verifier(self):
        """Create verifier with mock store."""
        from src.understanding.knowledge import KnowledgeStore, KnowledgeVerifier

        store = KnowledgeStore()
        return KnowledgeVerifier(store)

    @pytest.mark.asyncio
    async def test_verify_nonexistent_symbol(self, verifier):
        """Test verifying a symbol that doesn't exist."""
        result = await verifier.verify_symbol_exists("nonexistent_function")

        assert result.verified is False
        assert result.confidence.value == "unknown"

    @pytest.mark.asyncio
    async def test_verify_file_nonexistent(self, verifier):
        """Test verifying a file that doesn't exist."""
        result = await verifier.verify_file_exists("/nonexistent/path.py")

        assert result.verified is False


class TestConfidenceScorer:
    """Tests for ConfidenceScorer."""

    @pytest.fixture
    def scorer(self):
        """Create confidence scorer."""
        from src.understanding.knowledge import ConfidenceScorer

        return ConfidenceScorer()

    def test_confidence_to_score(self, scorer):
        """Test converting confidence to numeric score."""
        from src.understanding.models import ConfidenceLevel

        assert scorer.confidence_to_score(ConfidenceLevel.VERIFIED) == 1.0
        assert scorer.confidence_to_score(ConfidenceLevel.INFERRED) == 0.75
        assert scorer.confidence_to_score(ConfidenceLevel.UNCERTAIN) == 0.5
        assert scorer.confidence_to_score(ConfidenceLevel.UNKNOWN) == 0.0

    def test_score_to_confidence(self, scorer):
        """Test converting score to confidence."""
        from src.understanding.models import ConfidenceLevel

        assert scorer.score_to_confidence(0.95) == ConfidenceLevel.VERIFIED
        assert scorer.score_to_confidence(0.75) == ConfidenceLevel.INFERRED
        assert scorer.score_to_confidence(0.5) == ConfidenceLevel.UNCERTAIN
        assert scorer.score_to_confidence(0.1) == ConfidenceLevel.UNKNOWN
