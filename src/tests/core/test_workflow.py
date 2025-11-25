"""Tests for workflow creation and execution."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.core.workflow import create_workflow
from src.core.state import create_initial_state, validate_state
from src.models.state_models import WorkflowStatus, AgentRole


def test_workflow_creation():
    """Test workflow can be created without checkpointer."""
    workflow = create_workflow(
        checkpointer=None,
        memory_service=None,
        enable_human_approval=False,
    )

    assert workflow is not None


def test_initial_state_creation():
    """Test initial state creation."""
    task = {
        "id": "test_1",
        "description": "Test task",
    }

    state = create_initial_state(
        task=task,
        project_id="test_project",
    )

    assert state["workflow_id"] is not None
    assert state["project_id"] == "test_project"
    assert state["status"] == WorkflowStatus.PENDING.value
    assert state["task"] == task
    assert state["messages"] == []
    assert state["retry_count"] == 0


def test_state_validation():
    """Test state validation."""
    task = {
        "id": "test_1",
        "description": "Test task",
    }

    state = create_initial_state(
        task=task,
        project_id="test_project",
    )

    assert validate_state(state) is True


@pytest.mark.asyncio
async def test_workflow_graph_structure():
    """Test workflow graph has all expected nodes."""
    workflow = create_workflow(
        checkpointer=None,
        memory_service=None,
        enable_human_approval=False,
    )

    # Get graph structure
    graph = workflow.get_graph()
    nodes = graph.nodes

    # Check all agents are present
    expected_nodes = [
        "coordinator",
        "planner",
        "architect",
        "implementer",
        "tester",
        "validator",
        "debugger",
    ]

    for node in expected_nodes:
        # Nodes might be strings or objects with id attribute
        node_ids = [n if isinstance(n, str) else n.id for n in nodes]
        assert node in node_ids
