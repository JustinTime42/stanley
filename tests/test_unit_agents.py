"""Unit tests for agent implementations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import asyncio

# Direct module imports
import agents.coordinator as coord_module
import agents.planner as planner_module
import agents.architect as architect_module
import agents.implementer as implementer_module
import agents.tester as tester_module
import agents.validator as validator_module
import agents.debugger as debugger_module
import core.state as state_module
import models.state_models as state_models

# Extract classes and functions
CoordinatorAgent = coord_module.CoordinatorAgent
PlannerAgent = planner_module.PlannerAgent
ArchitectAgent = architect_module.ArchitectAgent
ImplementerAgent = implementer_module.ImplementerAgent
TesterAgent = tester_module.TesterAgent
ValidatorAgent = validator_module.ValidatorAgent
DebuggerAgent = debugger_module.DebuggerAgent
create_initial_state = state_module.create_initial_state
AgentRole = state_models.AgentRole
WorkflowStatus = state_models.WorkflowStatus


class TestCoordinatorAgent:
    """Test Coordinator agent."""

    @pytest.mark.asyncio
    async def test_coordinator_execute(self):
        """Test coordinator execution."""
        agent = CoordinatorAgent(memory_service=None)

        task = {"id": "test_1", "description": "Test task"}
        state = create_initial_state(task=task, project_id="test_project")

        response = await agent.execute(state)

        assert response.success is True
        assert response.next_agent is not None
        assert len(response.messages) > 0

    @pytest.mark.asyncio
    async def test_coordinator_routing(self):
        """Test coordinator routes correctly."""
        agent = CoordinatorAgent(memory_service=None)

        task = {"id": "test_1", "description": "Test task"}
        state = create_initial_state(task=task, project_id="test_project")

        response = await agent.execute(state)

        # Pending status should route to planner
        assert response.next_agent == AgentRole.PLANNER


class TestPlannerAgent:
    """Test Planner agent."""

    @pytest.mark.asyncio
    async def test_planner_execute(self):
        """Test planner execution."""
        agent = PlannerAgent(memory_service=None)

        task = {"id": "test_1", "description": "Create a REST API"}
        state = create_initial_state(task=task, project_id="test_project")

        response = await agent.execute(state)

        assert response.success is True
        assert "plan" in response.state_updates
        assert response.requires_human_approval is True  # Planner requires approval

    @pytest.mark.asyncio
    async def test_planner_creates_subtasks(self):
        """Test planner creates subtasks."""
        agent = PlannerAgent(memory_service=None)

        task = {"id": "test_1", "description": "Build application"}
        state = create_initial_state(task=task, project_id="test_project")

        response = await agent.execute(state)

        plan = response.state_updates.get("plan", {})
        assert "subtasks" in plan
        assert len(plan["subtasks"]) > 0


class TestArchitectAgent:
    """Test Architect agent."""

    @pytest.mark.asyncio
    async def test_architect_execute(self):
        """Test architect execution."""
        agent = ArchitectAgent(memory_service=None)

        task = {"id": "test_1", "description": "Design system"}
        state = create_initial_state(task=task, project_id="test_project")
        state["plan"] = {"subtasks": []}

        response = await agent.execute(state)

        assert response.success is True
        assert "architecture" in response.state_updates
        assert response.requires_human_approval is True


class TestImplementerAgent:
    """Test Implementer agent."""

    @pytest.mark.asyncio
    async def test_implementer_execute(self):
        """Test implementer execution."""
        agent = ImplementerAgent(memory_service=None)

        task = {"id": "test_1", "description": "Implement feature"}
        state = create_initial_state(task=task, project_id="test_project")
        state["architecture"] = {"components": []}

        response = await agent.execute(state)

        assert response.success is True
        assert "implementation" in response.state_updates


class TestTesterAgent:
    """Test Tester agent."""

    @pytest.mark.asyncio
    async def test_tester_execute(self):
        """Test tester execution."""
        agent = TesterAgent(memory_service=None)

        task = {"id": "test_1", "description": "Test code"}
        state = create_initial_state(task=task, project_id="test_project")
        state["implementation"] = {"files": []}

        response = await agent.execute(state)

        assert response.success is True
        assert "test_results" in response.state_updates

    @pytest.mark.asyncio
    async def test_tester_routes_on_pass(self):
        """Test tester routes to validator on pass."""
        agent = TesterAgent(memory_service=None)

        task = {"id": "test_1", "description": "Test code"}
        state = create_initial_state(task=task, project_id="test_project")
        state["implementation"] = {"files": []}

        response = await agent.execute(state)

        # Default mock passes all tests
        assert response.next_agent == AgentRole.VALIDATOR


class TestValidatorAgent:
    """Test Validator agent."""

    @pytest.mark.asyncio
    async def test_validator_execute(self):
        """Test validator execution."""
        agent = ValidatorAgent(memory_service=None)

        task = {"id": "test_1", "description": "Validate work"}
        state = create_initial_state(task=task, project_id="test_project")
        state["plan"] = {}
        state["implementation"] = {}
        state["test_results"] = {"all_passed": True}

        response = await agent.execute(state)

        assert response.success is True
        assert "validation_results" in response.state_updates
        assert response.requires_human_approval is True


class TestDebuggerAgent:
    """Test Debugger agent."""

    @pytest.mark.asyncio
    async def test_debugger_execute(self):
        """Test debugger execution."""
        agent = DebuggerAgent(memory_service=None)

        task = {"id": "test_1", "description": "Debug issues"}
        state = create_initial_state(task=task, project_id="test_project")
        state["test_results"] = {"all_passed": False, "failed": 2}

        response = await agent.execute(state)

        assert response.success is True
        assert "debug_info" in response.state_updates


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
