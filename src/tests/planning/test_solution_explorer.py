"""Tests for solution explorer."""

import pytest
from src.planning.solution_explorer import SolutionExplorer
from src.models.planning_models import PlanningContext, SolutionApproach


@pytest.mark.asyncio
async def test_solution_explorer_initialization():
    """Test solution explorer initializes correctly."""
    explorer = SolutionExplorer(max_solutions=5)

    assert explorer.max_solutions == 5
    assert explorer.llm_service is None
    assert explorer.decomposition_service is None


@pytest.mark.asyncio
async def test_generate_base_solution():
    """Test base solution generation without LLM."""
    explorer = SolutionExplorer(max_solutions=3)

    context = PlanningContext(
        budget_limit=50000,
        timeline_weeks=12,
        team_capabilities=["Python", "FastAPI"],
    )

    solution = await explorer._generate_base_solution(
        problem="Build a notification system",
        context=context,
    )

    assert solution is not None
    assert solution.name is not None
    assert solution.approach == SolutionApproach.INCREMENTAL
    assert len(solution.components) > 0


@pytest.mark.asyncio
async def test_generate_variation():
    """Test solution variation generation."""
    explorer = SolutionExplorer(max_solutions=3)

    context = PlanningContext(
        budget_limit=30000,
        timeline_weeks=8,
    )

    base_solution = await explorer._generate_base_solution(
        problem="Test problem",
        context=context,
    )

    variation = await explorer._generate_variation(
        problem="Test problem",
        base_solution=base_solution,
        approach=SolutionApproach.BIG_BANG,
        context=context,
    )

    assert variation is not None
    assert variation.approach == SolutionApproach.BIG_BANG
    assert variation.name != base_solution.name


@pytest.mark.asyncio
async def test_is_sufficiently_different():
    """Test solution diversity checking."""
    explorer = SolutionExplorer()

    from src.models.planning_models import Solution

    solution1 = Solution(
        name="Solution 1",
        description="Test",
        approach=SolutionApproach.INCREMENTAL,
        components=["A", "B", "C"],
    )

    solution2 = Solution(
        name="Solution 2",
        description="Test",
        approach=SolutionApproach.BIG_BANG,
        components=["A", "B", "C"],  # Same components
    )

    solution3 = Solution(
        name="Solution 3",
        description="Test",
        approach=SolutionApproach.STRANGLER,
        components=["X", "Y", "Z"],  # Different components
    )

    # Solution 2 has same components, should not be different enough
    assert not explorer._is_sufficiently_different(solution2, [solution1])

    # Solution 3 has different components, should be different enough
    assert explorer._is_sufficiently_different(solution3, [solution1])


@pytest.mark.asyncio
async def test_estimate_trade_offs():
    """Test trade-off estimation."""
    explorer = SolutionExplorer()

    from src.models.planning_models import Solution

    solution = Solution(
        name="Test Solution",
        description="Test",
        approach=SolutionApproach.INCREMENTAL,
        components=["A", "B"],
        estimated_effort_hours=100,
        estimated_cost=15000,
    )

    context = PlanningContext(
        budget_limit=50000,
        timeline_weeks=10,
    )

    trade_offs = await explorer._estimate_trade_offs(solution, context)

    assert "cost" in trade_offs
    assert "time" in trade_offs
    assert "complexity" in trade_offs
    assert all(0 <= score <= 1 for score in trade_offs.values())


def test_calculate_confidence():
    """Test confidence calculation."""
    explorer = SolutionExplorer()

    from src.models.planning_models import Solution

    # Solution with many details should have higher confidence
    detailed_solution = Solution(
        name="Detailed Solution",
        description="Test",
        approach=SolutionApproach.INCREMENTAL,
        components=["A", "B", "C", "D"],
        advantages=["Adv1", "Adv2", "Adv3"],
        assumptions=["Assumption 1"],
        estimated_effort_hours=100,
        estimated_cost=10000,
    )

    # Solution with minimal details
    minimal_solution = Solution(
        name="Minimal",
        description="Test",
        approach=SolutionApproach.INCREMENTAL,
    )

    detailed_confidence = explorer._calculate_confidence(detailed_solution)
    minimal_confidence = explorer._calculate_confidence(minimal_solution)

    assert detailed_confidence > minimal_confidence
    assert 0 <= detailed_confidence <= 1
    assert 0 <= minimal_confidence <= 1
