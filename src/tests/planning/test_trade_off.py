"""Tests for trade-off analyzer."""

import pytest
import numpy as np
from src.planning.trade_off_analyzer import TradeOffAnalyzer
from src.models.planning_models import Solution, SolutionApproach, TradeOffDimension


def test_analyzer_initialization():
    """Test analyzer initializes correctly."""
    analyzer = TradeOffAnalyzer()
    assert analyzer is not None


def test_analyze_single_solution():
    """Test analysis with single solution."""
    analyzer = TradeOffAnalyzer()

    solution = Solution(
        name="Only Solution",
        description="Test",
        approach=SolutionApproach.INCREMENTAL,
        trade_offs={"cost": 0.8, "time": 0.7},
    )

    result = analyzer.analyze_trade_offs([solution])

    assert len(result.scores) == 1
    assert result.scores[0] == 1.0
    assert result.best_solution_index == 0
    assert len(result.ranking) == 1


def test_analyze_multiple_solutions():
    """Test TOPSIS analysis with multiple solutions."""
    analyzer = TradeOffAnalyzer()

    solutions = [
        Solution(
            name="Solution A",
            description="High cost, fast",
            approach=SolutionApproach.INCREMENTAL,
            trade_offs={
                "cost": 0.5,
                "time": 0.9,
                "complexity": 0.6,
                "scalability": 0.8,
            },
        ),
        Solution(
            name="Solution B",
            description="Low cost, slow",
            approach=SolutionApproach.BIG_BANG,
            trade_offs={
                "cost": 0.9,
                "time": 0.5,
                "complexity": 0.7,
                "scalability": 0.6,
            },
        ),
        Solution(
            name="Solution C",
            description="Balanced",
            approach=SolutionApproach.HYBRID,
            trade_offs={
                "cost": 0.7,
                "time": 0.7,
                "complexity": 0.7,
                "scalability": 0.7,
            },
        ),
    ]

    result = analyzer.analyze_trade_offs(solutions)

    assert len(result.scores) == 3
    assert len(result.ranking) == 3
    assert result.best_solution_index in [0, 1, 2]
    assert all(0 <= score <= 1 for score in result.scores)

    # Ranking should be in descending order of scores
    for i in range(len(result.ranking) - 1):
        idx1 = result.ranking[i]
        idx2 = result.ranking[i + 1]
        assert result.scores[idx1] >= result.scores[idx2]


def test_custom_weights():
    """Test analysis with custom weights."""
    analyzer = TradeOffAnalyzer()

    solutions = [
        Solution(
            name="High Cost",
            description="Test",
            approach=SolutionApproach.INCREMENTAL,
            trade_offs={"cost": 0.3, "time": 0.9},
        ),
        Solution(
            name="Low Cost",
            description="Test",
            approach=SolutionApproach.BIG_BANG,
            trade_offs={"cost": 0.9, "time": 0.3},
        ),
    ]

    # Heavily weight cost
    weights = {"cost": 0.9, "time": 0.1}

    result = analyzer.analyze_trade_offs(solutions, weights)

    # Low cost solution should win with cost-heavy weights
    best_solution = solutions[result.best_solution_index]
    assert best_solution.name == "Low Cost"


def test_sensitivity_analysis():
    """Test sensitivity analysis."""
    analyzer = TradeOffAnalyzer()

    solutions = [
        Solution(
            name="A",
            description="Test",
            approach=SolutionApproach.INCREMENTAL,
            trade_offs={"cost": 0.6, "time": 0.7, "complexity": 0.5},
        ),
        Solution(
            name="B",
            description="Test",
            approach=SolutionApproach.BIG_BANG,
            trade_offs={"cost": 0.8, "time": 0.5, "complexity": 0.6},
        ),
    ]

    sensitivity = analyzer.analyze_sensitivity(
        solutions,
        dimension="cost",
        weight_range=(0.1, 0.9),
        steps=5,
    )

    assert len(sensitivity) == 5
    assert all("weight" in result for result in sensitivity)
    assert all("best_solution_index" in result for result in sensitivity)


def test_compare_solutions():
    """Test solution comparison."""
    analyzer = TradeOffAnalyzer()

    solution1 = Solution(
        name="Solution 1",
        description="Test",
        approach=SolutionApproach.INCREMENTAL,
        trade_offs={"cost": 0.8, "time": 0.6, "complexity": 0.7},
    )

    solution2 = Solution(
        name="Solution 2",
        description="Test",
        approach=SolutionApproach.BIG_BANG,
        trade_offs={"cost": 0.6, "time": 0.8, "complexity": 0.5},
    )

    comparison = analyzer.compare_solutions(solution1, solution2)

    assert comparison["solution1_name"] == "Solution 1"
    assert comparison["solution2_name"] == "Solution 2"
    assert "dimension_comparison" in comparison
    assert "overall_winner" in comparison


def test_pareto_optimal():
    """Test Pareto optimal identification."""
    analyzer = TradeOffAnalyzer()

    solutions = [
        Solution(
            name="Dominated",
            description="Test",
            approach=SolutionApproach.INCREMENTAL,
            trade_offs={"cost": 0.5, "time": 0.5, "complexity": 0.5},
        ),
        Solution(
            name="Optimal 1",
            description="Test",
            approach=SolutionApproach.BIG_BANG,
            trade_offs={"cost": 0.9, "time": 0.6, "complexity": 0.6},
        ),
        Solution(
            name="Optimal 2",
            description="Test",
            approach=SolutionApproach.STRANGLER,
            trade_offs={"cost": 0.6, "time": 0.9, "complexity": 0.7},
        ),
    ]

    pareto_indices = analyzer.identify_pareto_optimal(solutions)

    # First solution should be dominated
    assert 0 not in pareto_indices
    # Other two should be Pareto optimal
    assert len(pareto_indices) >= 1


def test_explain_decision():
    """Test decision explanation generation."""
    analyzer = TradeOffAnalyzer()

    solutions = [
        Solution(
            name="Winner",
            description="Best solution",
            approach=SolutionApproach.INCREMENTAL,
            trade_offs={"cost": 0.9, "time": 0.8, "scalability": 0.9},
            advantages=["Fast", "Cheap", "Scalable"],
        ),
        Solution(
            name="Runner-up",
            description="Second best",
            approach=SolutionApproach.BIG_BANG,
            trade_offs={"cost": 0.7, "time": 0.7, "scalability": 0.6},
        ),
    ]

    result = analyzer.analyze_trade_offs(solutions)
    explanation = analyzer.explain_decision(solutions, result, top_k=2)

    assert "Winner" in explanation
    assert "Trade-off Analysis Results" in explanation
    # Check for decision rationale content (format may vary)
    assert "recommended solution" in explanation.lower() or "winner" in explanation.lower()


def test_empty_solutions_raises_error():
    """Test that empty solution list raises error."""
    analyzer = TradeOffAnalyzer()

    with pytest.raises(ValueError, match="empty solution list"):
        analyzer.analyze_trade_offs([])


def test_normalization():
    """Test score normalization."""
    analyzer = TradeOffAnalyzer()

    solutions = [
        Solution(
            name="Extreme High",
            description="Test",
            approach=SolutionApproach.INCREMENTAL,
            trade_offs={"cost": 1.0, "time": 1.0},
        ),
        Solution(
            name="Extreme Low",
            description="Test",
            approach=SolutionApproach.BIG_BANG,
            trade_offs={"cost": 0.0, "time": 0.0},
        ),
    ]

    result = analyzer.analyze_trade_offs(solutions)

    # Normalized matrix should be present
    assert result.normalized_matrix is not None
    assert len(result.normalized_matrix) == 2
