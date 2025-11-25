"""Tests for technology selector."""

import pytest
from src.architecture.technology_selector import TechnologySelector
from src.models.planning_models import PlanningContext


@pytest.fixture
def selector():
    """Create technology selector fixture."""
    return TechnologySelector()


def test_selector_initialization(selector):
    """Test selector initializes correctly."""
    assert selector is not None
    assert len(selector.technologies) > 0


def test_get_technology(selector):
    """Test getting technology by name."""
    python = selector.get_technology("Python")

    assert python is not None
    assert python.name == "Python"
    assert python.maturity == "adopt"


def test_add_technology(selector):
    """Test adding new technology."""
    from src.models.architecture_models import Technology

    new_tech = Technology(
        name="TestTech",
        category="test",
        license="MIT",
        maturity="trial",
    )

    initial_count = len(selector.technologies)

    selector.add_technology(new_tech)

    assert len(selector.technologies) == initial_count + 1
    assert selector.get_technology("TestTech") is not None


@pytest.mark.asyncio
async def test_filter_candidates(selector):
    """Test candidate filtering."""
    context = PlanningContext(
        team_capabilities=["Python", "FastAPI"],
        regulatory_requirements=["No GPL"],
    )

    requirements = {"web_development": True}

    candidates = selector._filter_candidates(requirements, context)

    # Should filter out high learning curve and GPL licensed
    assert len(candidates) > 0
    assert all(tech.maturity != "hold" for tech in candidates)


@pytest.mark.asyncio
async def test_evaluate_technology(selector):
    """Test technology evaluation."""
    context = PlanningContext(
        budget_limit=50000,
        team_capabilities=["Python"],
    )

    requirements = {"web_development": 0.9, "api_development": 0.9}

    python_tech = selector.get_technology("Python")

    evaluation = await selector._evaluate_technology(
        python_tech,
        requirements,
        context,
    )

    assert evaluation is not None
    assert 0 <= evaluation.score <= 1
    assert len(evaluation.pros) > 0
    assert evaluation.recommendation in ["adopt", "trial", "assess", "hold"]


@pytest.mark.asyncio
async def test_recommend_stack(selector):
    """Test technology stack recommendation."""
    context = PlanningContext(
        budget_limit=30000,
        timeline_weeks=12,
        team_capabilities=["Python", "JavaScript"],
    )

    requirements = {
        "web_development": True,
        "api_development": True,
        "database": True,
    }

    recommendations = await selector.recommend_stack(
        requirements,
        context,
        max_recommendations=5,
    )

    assert len(recommendations) <= 5
    assert len(recommendations) > 0

    # All recommendations should have scores
    assert all(0 <= r.score <= 1 for r in recommendations)

    # Should be sorted by score
    for i in range(len(recommendations) - 1):
        assert recommendations[i].score >= recommendations[i + 1].score


@pytest.mark.asyncio
async def test_ensure_compatibility(selector):
    """Test compatibility checking."""
    from src.models.architecture_models import Technology, TechnologyEvaluation

    # Create test evaluations with incompatible technologies
    tech1 = Technology(
        name="Tech1",
        category="test",
        license="MIT",
        incompatible_with=["Tech2"],
    )

    tech2 = Technology(
        name="Tech2",
        category="test",
        license="MIT",
        incompatible_with=["Tech1"],
    )

    tech3 = Technology(
        name="Tech3",
        category="test",
        license="MIT",
        compatible_with=["Tech1"],
    )

    evaluations = [
        TechnologyEvaluation(
            technology=tech1.model_dump(),
            score=0.9,
        ),
        TechnologyEvaluation(
            technology=tech2.model_dump(),
            score=0.8,
        ),
        TechnologyEvaluation(
            technology=tech3.model_dump(),
            score=0.7,
        ),
    ]

    compatible = selector._ensure_compatibility(evaluations)

    # Tech1 and Tech2 are incompatible, only one should be selected
    tech_names = [e.technology["name"] for e in compatible]
    assert not ("Tech1" in tech_names and "Tech2" in tech_names)


def test_estimate_migration_effort(selector):
    """Test migration effort estimation."""
    from src.models.architecture_models import Technology

    tech = Technology(
        name="NewTech",
        category="test",
        license="MIT",
    )

    # No existing architecture - low effort
    context1 = PlanningContext()
    effort1 = selector._estimate_migration_effort(tech, context1)
    assert effort1 == "low"

    # Existing architecture with same tech - low effort
    context2 = PlanningContext(
        existing_architecture={"technologies": ["NewTech"]}
    )
    effort2 = selector._estimate_migration_effort(tech, context2)
    assert effort2 == "low"

    # Existing architecture with different tech - high effort
    context3 = PlanningContext(
        existing_architecture={"technologies": ["OldTech"]}
    )
    effort3 = selector._estimate_migration_effort(tech, context3)
    assert effort3 == "high"


def test_get_recommendation(selector):
    """Test recommendation determination."""
    # High score + adopt maturity = adopt
    rec1 = selector._get_recommendation(0.8, "adopt")
    assert rec1 == "adopt"

    # Medium score + trial maturity = trial
    rec2 = selector._get_recommendation(0.6, "trial")
    assert rec2 == "trial"

    # Low score = assess or hold
    rec3 = selector._get_recommendation(0.3, "adopt")
    assert rec3 in ["assess", "hold"]
