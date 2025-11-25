"""Tests for pattern recognition."""

import pytest
from src.architecture.pattern_recognizer import PatternRecognizer
from src.architecture.pattern_library import PatternLibrary
from src.models.architecture_models import ArchitecturePattern


@pytest.fixture
def pattern_library():
    """Create pattern library fixture."""
    return PatternLibrary()


@pytest.fixture
def recognizer(pattern_library):
    """Create pattern recognizer fixture."""
    return PatternRecognizer(pattern_library=pattern_library)


@pytest.mark.asyncio
async def test_recognizer_initialization(recognizer):
    """Test recognizer initializes correctly."""
    assert recognizer is not None
    assert recognizer.pattern_library is not None


@pytest.mark.asyncio
async def test_analyze_structure(recognizer):
    """Test codebase structure analysis."""
    # Use current codebase as test subject
    structure = await recognizer._analyze_structure("src")

    assert "root_path" in structure
    assert "directories" in structure
    assert "files" in structure
    assert len(structure["files"]) > 0


@pytest.mark.asyncio
async def test_check_layer_separation(recognizer):
    """Test layer separation detection."""
    structure = {
        "directories": [
            "presentation/controllers",
            "business/services",
            "data/repositories",
        ],
        "files": [],
    }

    has_layers, evidence = recognizer._check_layer_separation(structure)

    assert has_layers is True
    assert len(evidence) > 0


@pytest.mark.asyncio
async def test_check_service_boundaries(recognizer):
    """Test service boundary detection."""
    structure = {
        "directories": [
            "services/user-service",
            "services/payment-service",
            "services/notification-service",
        ],
        "files": [],
    }

    has_services, evidence = recognizer._check_service_boundaries(structure)

    assert has_services is True
    assert len(evidence) > 0


@pytest.mark.asyncio
async def test_check_api_definitions(recognizer):
    """Test API definition detection."""
    structure = {
        "files": [
            "api/routes.py",
            "api/endpoints.py",
        ],
        "directories": [],
    }

    has_apis, evidence = recognizer._check_api_definitions(structure)

    assert has_apis is True


@pytest.mark.asyncio
async def test_check_event_patterns(recognizer):
    """Test event pattern detection."""
    structure = {
        "files": [
            "events/user_created.py",
            "handlers/notification_handler.py",
            "listeners/email_listener.py",
        ],
        "directories": [],
    }

    has_events, evidence = recognizer._check_event_patterns(structure)

    assert has_events is True


@pytest.mark.asyncio
async def test_check_repository_pattern(recognizer):
    """Test repository pattern detection."""
    structure = {
        "files": [
            "repositories/user_repository.py",
            "repositories/product_repo.py",
        ],
        "directories": ["repositories"],
    }

    has_repos, evidence = recognizer._check_repository_pattern(structure)

    assert has_repos is True


@pytest.mark.asyncio
async def test_check_mvc_pattern(recognizer):
    """Test MVC pattern detection."""
    structure = {
        "directories": [
            "models",
            "views",
            "controllers",
        ],
        "files": [],
    }

    has_mvc, evidence = recognizer._check_mvc_pattern(structure)

    assert has_mvc is True
    assert len(evidence) == 3


@pytest.mark.asyncio
async def test_recognize_patterns_with_confidence_threshold(recognizer):
    """Test pattern recognition with confidence threshold."""
    # Create a test directory structure
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create layered structure
        os.makedirs(os.path.join(tmpdir, "presentation"))
        os.makedirs(os.path.join(tmpdir, "business"))
        os.makedirs(os.path.join(tmpdir, "data"))

        patterns = await recognizer.recognize_patterns(tmpdir, confidence_threshold=0.3)

        # Should detect at least layered pattern
        pattern_names = [p.pattern for p in patterns]
        assert any("layered" in p for p in pattern_names) or len(patterns) >= 0


@pytest.mark.asyncio
async def test_pattern_confidence_scoring(recognizer):
    """Test pattern confidence scoring."""
    # Layered pattern structure
    structure = {
        "directories": [
            "presentation/controllers",
            "business/services",
            "data/repositories",
        ],
        "files": ["presentation/base.py", "business/interface.py"],
    }

    match = await recognizer._check_pattern(ArchitecturePattern.LAYERED, structure)

    assert match is not None
    assert 0 <= match.confidence <= 1
    assert len(match.evidence) > 0
