"""Tests for decision documenter."""

import pytest
from src.planning.decision_documenter import DecisionDocumenter
from src.models.planning_models import (
    Solution,
    PlanningContext,
    TradeOffAnalysisResult,
    SolutionApproach,
)


def test_documenter_initialization():
    """Test documenter initializes correctly."""
    documenter = DecisionDocumenter()

    assert documenter is not None
    assert len(documenter.decisions) == 0


def test_create_adr():
    """Test ADR creation."""
    documenter = DecisionDocumenter()

    solutions = [
        Solution(
            name="Selected Solution",
            description="The chosen approach",
            approach=SolutionApproach.INCREMENTAL,
            components=["Component A", "Component B"],
            technologies=["Python", "FastAPI"],
            estimated_effort_hours=160,
            estimated_cost=20000,
            advantages=["Fast", "Cheap"],
            disadvantages=["Complex"],
            assumptions=["Team has Python experience"],
        )
    ]

    analysis = TradeOffAnalysisResult(
        scores=[0.85],
        ranking=[0],
        best_solution_index=0,
        trade_off_matrix=[[0.8, 0.7, 0.9]],
        weights_used={"cost": 0.5, "time": 0.5},
    )

    context = PlanningContext(
        budget_limit=50000,
        timeline_weeks=12,
        team_capabilities=["Python"],
    )

    decision = documenter.create_adr(
        title="API Gateway Selection",
        solutions=solutions,
        selected_solution=solutions[0],
        trade_off_analysis=analysis,
        context=context,
    )

    assert decision is not None
    assert decision.title == "API Gateway Selection"
    assert decision.status == "accepted"
    assert "Selected Solution" in decision.decision
    assert len(decision.alternatives_considered) == 1
    assert decision.trade_off_analysis["scores"] == [0.85]


def test_update_decision_status():
    """Test updating decision status."""
    documenter = DecisionDocumenter()

    # Create a decision first
    solutions = [
        Solution(
            name="Test",
            description="Test",
            approach=SolutionApproach.INCREMENTAL,
        )
    ]

    analysis = TradeOffAnalysisResult(
        scores=[1.0],
        ranking=[0],
        best_solution_index=0,
        trade_off_matrix=[[0.5]],
        weights_used={},
    )

    context = PlanningContext()

    decision = documenter.create_adr(
        title="Test Decision",
        solutions=solutions,
        selected_solution=solutions[0],
        trade_off_analysis=analysis,
        context=context,
    )

    decision_id = decision.id

    # Update status
    updated = documenter.update_decision_status(
        decision_id,
        "deprecated",
        "Better solution found",
    )

    assert updated.status == "deprecated"
    assert "Better solution found" in updated.context


def test_supersede_decision():
    """Test superseding a decision."""
    documenter = DecisionDocumenter()

    # Create old decision
    solutions = [
        Solution(
            name="Old",
            description="Test",
            approach=SolutionApproach.INCREMENTAL,
        )
    ]

    analysis = TradeOffAnalysisResult(
        scores=[1.0],
        ranking=[0],
        best_solution_index=0,
        trade_off_matrix=[[0.5]],
        weights_used={},
    )

    context = PlanningContext()

    old_decision = documenter.create_adr(
        title="Old Decision",
        solutions=solutions,
        selected_solution=solutions[0],
        trade_off_analysis=analysis,
        context=context,
    )

    # Create new decision
    new_decision = documenter.create_adr(
        title="New Decision",
        solutions=solutions,
        selected_solution=solutions[0],
        trade_off_analysis=analysis,
        context=context,
    )

    # Supersede old with new
    superseded = documenter.supersede_decision(old_decision.id, new_decision)

    assert old_decision.status == "superseded"
    assert new_decision.supersedes == old_decision.id
    assert old_decision.id in new_decision.related_decisions


def test_get_decision_history():
    """Test retrieving decision history."""
    documenter = DecisionDocumenter()

    solutions = [
        Solution(
            name="Test",
            description="Test",
            approach=SolutionApproach.INCREMENTAL,
        )
    ]

    analysis = TradeOffAnalysisResult(
        scores=[1.0],
        ranking=[0],
        best_solution_index=0,
        trade_off_matrix=[[0.5]],
        weights_used={},
    )

    context = PlanningContext()

    # Create chain of decisions
    decision1 = documenter.create_adr(
        title="Decision 1",
        solutions=solutions,
        selected_solution=solutions[0],
        trade_off_analysis=analysis,
        context=context,
    )

    decision2 = documenter.create_adr(
        title="Decision 2",
        solutions=solutions,
        selected_solution=solutions[0],
        trade_off_analysis=analysis,
        context=context,
    )

    documenter.supersede_decision(decision1.id, decision2)

    # Get history
    history = documenter.get_decision_history(decision2.id)

    assert len(history) == 2
    assert history[0].id == decision1.id  # Oldest first
    assert history[1].id == decision2.id


def test_export_to_markdown():
    """Test markdown export."""
    documenter = DecisionDocumenter()

    solutions = [
        Solution(
            name="Test Solution",
            description="Test",
            approach=SolutionApproach.INCREMENTAL,
        )
    ]

    analysis = TradeOffAnalysisResult(
        scores=[1.0],
        ranking=[0],
        best_solution_index=0,
        trade_off_matrix=[[0.5]],
        weights_used={},
    )

    context = PlanningContext()

    decision = documenter.create_adr(
        title="Markdown Test",
        solutions=solutions,
        selected_solution=solutions[0],
        trade_off_analysis=analysis,
        context=context,
    )

    markdown = documenter.export_to_markdown(decision)

    assert "# Markdown Test" in markdown
    assert "**Status**: ACCEPTED" in markdown
    assert "## Context" in markdown
    assert "## Decision" in markdown
    assert "## Consequences" in markdown


def test_get_all_decisions():
    """Test retrieving all decisions."""
    documenter = DecisionDocumenter()

    solutions = [
        Solution(
            name="Test",
            description="Test",
            approach=SolutionApproach.INCREMENTAL,
        )
    ]

    analysis = TradeOffAnalysisResult(
        scores=[1.0],
        ranking=[0],
        best_solution_index=0,
        trade_off_matrix=[[0.5]],
        weights_used={},
    )

    context = PlanningContext()

    # Create multiple decisions
    for i in range(3):
        documenter.create_adr(
            title=f"Decision {i+1}",
            solutions=solutions,
            selected_solution=solutions[0],
            trade_off_analysis=analysis,
            context=context,
        )

    all_decisions = documenter.get_all_decisions()

    assert len(all_decisions) == 3


def test_get_active_decisions():
    """Test retrieving only active decisions."""
    documenter = DecisionDocumenter()

    solutions = [
        Solution(
            name="Test",
            description="Test",
            approach=SolutionApproach.INCREMENTAL,
        )
    ]

    analysis = TradeOffAnalysisResult(
        scores=[1.0],
        ranking=[0],
        best_solution_index=0,
        trade_off_matrix=[[0.5]],
        weights_used={},
    )

    context = PlanningContext()

    # Create decisions
    decision1 = documenter.create_adr(
        title="Active",
        solutions=solutions,
        selected_solution=solutions[0],
        trade_off_analysis=analysis,
        context=context,
    )

    decision2 = documenter.create_adr(
        title="To Deprecate",
        solutions=solutions,
        selected_solution=solutions[0],
        trade_off_analysis=analysis,
        context=context,
    )

    # Deprecate one
    documenter.update_decision_status(decision2.id, "deprecated")

    active_decisions = documenter.get_active_decisions()

    assert len(active_decisions) == 1
    assert active_decisions[0].id == decision1.id
