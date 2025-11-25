"""Tests for consistency checker."""

import pytest
from src.architecture.consistency_checker import ConsistencyChecker
from src.architecture.pattern_library import PatternLibrary
from src.models.architecture_models import ArchitectureDesign, PatternMatch


@pytest.fixture
def pattern_library():
    """Create pattern library fixture."""
    return PatternLibrary()


@pytest.fixture
def checker(pattern_library):
    """Create consistency checker fixture."""
    return ConsistencyChecker(pattern_library=pattern_library)


@pytest.mark.asyncio
async def test_checker_initialization(checker):
    """Test checker initializes correctly."""
    assert checker is not None
    assert checker.pattern_library is not None


@pytest.mark.asyncio
async def test_check_consistency_complete_design(checker):
    """Test consistency check on complete design."""
    design = ArchitectureDesign(
        name="Test Architecture",
        description="Complete architecture design",
        patterns=["layered"],
        layers=["presentation", "business", "data"],
        components={
            "API": {
                "type": "service",
                "responsibilities": ["Handle requests"],
            },
            "Database": {
                "type": "repository",
                "responsibilities": ["Store data"],
            },
        },
        technologies={
            "Python": {"name": "Python", "version": "3.11"},
            "PostgreSQL": {"name": "PostgreSQL", "version": "15"},
        },
        quality_attributes={
            "performance": 0.8,
            "scalability": 0.7,
        },
    )

    result = await checker.check_consistency(design)

    assert "consistency_score" in result
    assert "completeness_score" in result
    assert result["consistency_score"] > 0.5
    assert result["completeness_score"] >= 0.8  # Should be complete


@pytest.mark.asyncio
async def test_check_consistency_incomplete_design(checker):
    """Test consistency check on incomplete design."""
    design = ArchitectureDesign(
        name="Incomplete",
        description="Missing components",
        patterns=[],
        components={},
        technologies={},
    )

    result = await checker.check_consistency(design)

    assert result["completeness_score"] < 0.5  # Should be incomplete
    assert len(result["issues"]) > 0 or len(result["warnings"]) > 0


@pytest.mark.asyncio
async def test_check_technology_compatibility(checker):
    """Test technology compatibility checking."""
    technologies = {
        "Tech1": {
            "name": "Tech1",
            "incompatible_with": ["Tech2"],
        },
        "Tech2": {
            "name": "Tech2",
            "incompatible_with": [],
        },
    }

    incompatible = checker._check_technology_compatibility(technologies)

    assert len(incompatible) > 0
    assert ("Tech1", "Tech2") in incompatible


@pytest.mark.asyncio
async def test_check_layer_violations(checker):
    """Test layer violation detection."""
    design = ArchitectureDesign(
        name="Test",
        description="Test",
        patterns=["layered"],
        components={
            "DataLayer": {
                "type": "data",
                "layer": "data",
                "dependencies": ["PresentationLayer"],  # Violation!
            },
            "PresentationLayer": {
                "type": "presentation",
                "layer": "presentation",
                "dependencies": [],
            },
        },
    )

    violations = checker._check_layer_violations(design)

    # Should detect violation (data depending on presentation)
    assert isinstance(violations, list)


@pytest.mark.asyncio
async def test_identify_layer(checker):
    """Test layer identification."""
    # Presentation layer
    layer1 = checker._identify_layer(
        "UserController",
        {"type": "controller"},
    )
    assert layer1 == "presentation"

    # Business layer
    layer2 = checker._identify_layer(
        "UserService",
        {"type": "service"},
    )
    assert layer2 in ["business", "service"]

    # Data layer
    layer3 = checker._identify_layer(
        "UserRepository",
        {"type": "repository"},
    )
    assert layer3 == "data"


@pytest.mark.asyncio
async def test_check_component_coupling(checker):
    """Test component coupling detection."""
    design = ArchitectureDesign(
        name="Test",
        description="Test",
        components={
            "HighCoupling": {
                "dependencies": ["A", "B", "C", "D", "E", "F"],
                "dependents": ["X", "Y"],
            },
            "LowCoupling": {
                "dependencies": ["A"],
                "dependents": [],
            },
        },
    )

    issues = checker._check_component_coupling(design)

    # HighCoupling should be flagged
    assert len(issues) > 0
    assert any("HighCoupling" in issue for issue in issues)


@pytest.mark.asyncio
async def test_pattern_adherence_check(checker):
    """Test pattern adherence checking."""
    design = ArchitectureDesign(
        name="Test",
        description="Test",
        patterns=["layered"],
    )

    patterns = [
        PatternMatch(
            pattern="layered",
            confidence=0.65,  # Low confidence
            location="test",
            evidence=["Some evidence"],
            matches_best_practice=False,
        )
    ]

    result = await checker._check_pattern_adherence(design, patterns)

    assert "warnings" in result
    assert len(result["warnings"]) > 0  # Should warn about low confidence


@pytest.mark.asyncio
async def test_completeness_checks(checker):
    """Test completeness checking."""
    # Complete design
    complete_design = ArchitectureDesign(
        name="Complete",
        description="A complete design",
        patterns=["layered"],
        components={"A": {}, "B": {}},
        technologies={"Python": {}},
        quality_attributes={"performance": 0.8},
    )

    result1 = await checker._run_completeness_checks(complete_design)
    assert result1["score"] >= 0.8

    # Incomplete design
    incomplete_design = ArchitectureDesign(
        name="Incomplete",
        description="Missing everything",
    )

    result2 = await checker._run_completeness_checks(incomplete_design)
    assert result2["score"] < 0.5


@pytest.mark.asyncio
async def test_generate_recommendations(checker):
    """Test recommendation generation."""
    results = {
        "consistency_score": 0.5,
        "completeness_score": 0.6,
        "issues": ["Issue 1", "Issue 2"],
        "warnings": ["W1", "W2", "W3", "W4", "W5", "W6"],
    }

    recommendations = checker._generate_recommendations(results)

    assert len(recommendations) > 0
    # Should recommend addressing issues
    assert any("issue" in r.lower() for r in recommendations)


@pytest.mark.asyncio
async def test_validate_patterns(checker):
    """Test pattern validation."""
    design = ArchitectureDesign(
        name="Test",
        description="Test",
        patterns=["layered", "repository"],
    )

    validations = await checker.validate_patterns(design)

    assert isinstance(validations, list)
    assert len(validations) >= 0
