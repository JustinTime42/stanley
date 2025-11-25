"""Tests for Coordinator agent."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.agents.coordinator import CoordinatorAgent
from src.core.state import create_initial_state
from src.models.state_models import WorkflowStatus, AgentRole


@pytest.mark.asyncio
async def test_coordinator_execution():
    """Test coordinator can execute."""
    agent = CoordinatorAgent(memory_service=None)

    task = {
        "id": "test_1",
        "description": "Test task",
    }

    state = create_initial_state(
        task=task,
        project_id="test_project",
    )

    response = await agent.execute(state)

    assert response.success is True
    assert response.next_agent is not None
    assert len(response.messages) > 0


@pytest.mark.asyncio
async def test_coordinator_routing():
    """Test coordinator routes to correct agent."""
    agent = CoordinatorAgent(memory_service=None)

    task = {
        "id": "test_1",
        "description": "Test task",
    }

    state = create_initial_state(
        task=task,
        project_id="test_project",
    )

    # Pending status should route to planner
    response = await agent.execute(state)

    assert response.next_agent == AgentRole.PLANNER
