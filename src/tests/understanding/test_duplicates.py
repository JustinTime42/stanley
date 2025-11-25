"""Tests for duplicate detection."""

import pytest


class TestDuplicateDetector:
    """Tests for DuplicateDetector."""

    @pytest.fixture
    def detector(self):
        """Create duplicate detector."""
        from src.understanding.knowledge import KnowledgeStore, DuplicateDetector

        store = KnowledgeStore()
        return DuplicateDetector(store, similarity_threshold=0.7)

    def test_name_similarity(self, detector):
        """Test name similarity calculation."""
        # Exact match
        assert detector._name_similarity("foo", "foo") == 1.0

        # Different
        assert detector._name_similarity("foo", "bar") < 0.5

        # Similar
        similarity = detector._name_similarity("get_user", "get_users")
        assert similarity > 0.7

    def test_signature_similarity(self, detector):
        """Test signature similarity calculation."""
        # Same parameters
        similarity = detector._signature_similarity(
            "def foo(x, y, z)",
            "def bar(x, y, z)"
        )
        assert similarity == 1.0

        # Some overlap
        similarity = detector._signature_similarity(
            "def foo(x, y)",
            "def bar(x, z)"
        )
        assert 0 < similarity < 1

        # No overlap
        similarity = detector._signature_similarity(
            "def foo(a, b)",
            "def bar(x, y)"
        )
        assert similarity == 0.0

    def test_recommendation(self, detector):
        """Test recommendation generation."""
        assert detector._recommend(0.98, "semantic") == "use_existing"
        assert detector._recommend(0.90, "semantic") == "use_existing"
        assert detector._recommend(0.85, "name") == "consider_existing"
        assert detector._recommend(0.75, "name") == "extend_existing"
        assert detector._recommend(0.50, "name") == "create_new"

    @pytest.mark.asyncio
    async def test_check_before_create_empty(self, detector):
        """Test duplicate check with empty store."""
        candidates = await detector.check_before_create(
            proposed_name="new_function",
            proposed_description="Does something new",
            proposed_kind="function",
        )

        # Should return empty list with empty store
        assert isinstance(candidates, list)
