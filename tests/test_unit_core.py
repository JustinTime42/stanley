"""Unit tests for core workflow components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import asyncio
from datetime import datetime

# Direct module imports (avoid __init__.py with relative imports)
import core.state as state_module
import core.edges as edges_module
import models.state_models as state_models

# Extract functions and classes
create_initial_state = state_module.create_initial_state
validate_state = state_module.validate_state
update_state = state_module.update_state
serialize_state = state_module.serialize_state
deserialize_state = state_module.deserialize_state
get_state_summary = state_module.get_state_summary

route_from_coordinator = edges_module.route_from_coordinator
route_from_tester = edges_module.route_from_tester
route_from_validator = edges_module.route_from_validator
route_from_debugger = edges_module.route_from_debugger
should_retry = edges_module.should_retry
needs_human_approval = edges_module.needs_human_approval

WorkflowStatus = state_models.WorkflowStatus
AgentRole = state_models.AgentRole


class TestStateManagement:
    """Test state creation and management."""

    def test_create_initial_state(self):
        """Test initial state creation."""
        task = {"id": "test_1", "description": "Test task"}
        state = create_initial_state(task=task, project_id="test_project")

        assert state["workflow_id"] is not None
        assert state["project_id"] == "test_project"
        assert state["status"] == WorkflowStatus.PENDING.value
        assert state["task"] == task
        assert state["messages"] == []
        assert state["retry_count"] == 0
        assert state["state_version"] == 1

    def test_validate_state_valid(self):
        """Test state validation with valid state."""
        task = {"id": "test_1", "description": "Test task"}
        state = create_initial_state(task=task, project_id="test_project")
        assert validate_state(state) is True

    def test_validate_state_invalid(self):
        """Test state validation with invalid state."""
        invalid_state = {"workflow_id": "test"}  # Missing required fields
        assert validate_state(invalid_state) is False

    def test_update_state(self):
        """Test immutable state updates."""
        task = {"id": "test_1", "description": "Test task"}
        state = create_initial_state(task=task, project_id="test_project")

        original_version = state["state_version"]

        # Update state
        updated = update_state(state, {"status": WorkflowStatus.PLANNING.value})

        # Verify immutability
        assert state["state_version"] == original_version  # Original unchanged
        assert updated["state_version"] == original_version + 1  # New version
        assert updated["status"] == WorkflowStatus.PLANNING.value
        assert state["status"] == WorkflowStatus.PENDING.value  # Original unchanged

    def test_serialize_deserialize(self):
        """Test state serialization roundtrip."""
        task = {"id": "test_1", "description": "Test task"}
        original = create_initial_state(task=task, project_id="test_project")

        # Serialize
        serialized = serialize_state(original)
        assert isinstance(serialized, dict)

        # Deserialize
        deserialized = deserialize_state(serialized)

        # Verify
        assert deserialized["workflow_id"] == original["workflow_id"]
        assert deserialized["project_id"] == original["project_id"]
        assert deserialized["status"] == original["status"]

    def test_get_state_summary(self):
        """Test state summary generation."""
        task = {"id": "test_1", "description": "Test task"}
        state = create_initial_state(task=task, project_id="test_project")

        summary = get_state_summary(state)

        assert "workflow_id" in summary
        assert "status" in summary
        assert "current_agent" in summary
        assert "state_version" in summary


class TestConditionalEdges:
    """Test workflow routing logic."""

    def test_route_from_coordinator_pending(self):
        """Test coordinator routes pending to planner."""
        task = {"id": "test", "description": "Test"}
        state = create_initial_state(task=task, project_id="test")

        next_node = route_from_coordinator(state)
        assert next_node == "planner"

    def test_route_from_coordinator_complete(self):
        """Test coordinator routes complete to end."""
        task = {"id": "test", "description": "Test"}
        state = create_initial_state(task=task, project_id="test")
        state = update_state(state, {"status": WorkflowStatus.COMPLETE.value})

        next_node = route_from_coordinator(state)
        assert next_node == "end"

    def test_route_from_tester_pass(self):
        """Test tester routes passing tests to validator."""
        task = {"id": "test", "description": "Test"}
        state = create_initial_state(task=task, project_id="test")
        state = update_state(state, {
            "status": WorkflowStatus.TESTING.value,
            "test_results": {"all_passed": True}
        })

        next_node = route_from_tester(state)
        assert next_node == "validator"

    def test_route_from_tester_fail(self):
        """Test tester routes failing tests to debugger."""
        task = {"id": "test", "description": "Test"}
        state = create_initial_state(task=task, project_id="test")
        state = update_state(state, {
            "test_results": {"all_passed": False}
        })

        next_node = route_from_tester(state)
        assert next_node == "debugger"

    def test_route_from_validator_approved(self):
        """Test validator routes approved to end."""
        task = {"id": "test", "description": "Test"}
        state = create_initial_state(task=task, project_id="test")
        state = update_state(state, {
            "validation_results": {"approved": True}
        })

        next_node = route_from_validator(state)
        assert next_node == "end"

    def test_route_from_validator_rejected(self):
        """Test validator routes rejected to debugger."""
        task = {"id": "test", "description": "Test"}
        state = create_initial_state(task=task, project_id="test")
        state = update_state(state, {
            "validation_results": {"approved": False}
        })

        next_node = route_from_validator(state)
        assert next_node == "debugger"

    def test_should_retry_within_limit(self):
        """Test retry logic within limits."""
        task = {"id": "test", "description": "Test"}
        state = create_initial_state(task=task, project_id="test")
        state = update_state(state, {"retry_count": 1, "max_retries": 3})

        assert should_retry(state) is True

    def test_should_retry_exceeded(self):
        """Test retry logic when limit exceeded."""
        task = {"id": "test", "description": "Test"}
        state = create_initial_state(task=task, project_id="test")
        state = update_state(state, {"retry_count": 3, "max_retries": 3})

        assert should_retry(state) is False

    def test_needs_human_approval(self):
        """Test human approval detection."""
        task = {"id": "test", "description": "Test"}
        state = create_initial_state(task=task, project_id="test")

        # Without approval requirement
        assert needs_human_approval(state) is False

        # With approval requirement
        state = update_state(state, {"requires_human_approval": True})
        assert needs_human_approval(state) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
