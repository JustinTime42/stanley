"""State management utilities for LangGraph workflows."""

import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from ..models.state_models import AgentState, WorkflowStatus, AgentRole

logger = logging.getLogger(__name__)


def create_initial_state(
    task: Dict[str, Any],
    project_id: str,
    workflow_id: Optional[str] = None,
    session_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> AgentState:
    """
    Create initial workflow state.

    Args:
        task: Task specification
        project_id: Project identifier
        workflow_id: Optional workflow ID (generated if not provided)
        session_id: Optional session ID
        config: Optional configuration overrides

    Returns:
        AgentState TypedDict initialized
    """
    now = datetime.now().isoformat()
    config = config or {}

    state: AgentState = {
        # Core workflow state
        "workflow_id": workflow_id or str(uuid.uuid4()),
        "project_id": project_id,
        "session_id": session_id or str(uuid.uuid4()),
        "status": WorkflowStatus.PENDING.value,
        # Agent communication
        "messages": [],
        "current_agent": AgentRole.COORDINATOR.value,
        "next_agent": None,
        # Task management
        "task": task,
        "subtasks": [],
        "completed_subtasks": [],
        # Results and artifacts
        "plan": None,
        "architecture": None,
        "implementation": None,
        "test_results": None,
        "validation_results": None,
        "debug_info": None,
        # Control flow
        "retry_count": 0,
        "max_retries": config.get("max_retries", 3),
        "should_continue": True,
        "requires_human_approval": False,
        "human_feedback": None,
        # Checkpoint and rollback
        "swarm_checkpoint_id": str(uuid.uuid4()),
        "parent_checkpoint_id": None,
        "state_version": 1,
        # Memory references
        "memory_ids": [],
        "context": {},
        # Metadata
        "created_at": now,
        "updated_at": now,
        "elapsed_time": 0.0,
        "token_count": 0,
        "cost": 0.0,
    }

    logger.info(f"Created initial state for workflow {state['workflow_id']}")
    return state


def validate_state(state: AgentState) -> bool:
    """
    Validate state schema and required fields.

    Args:
        state: State to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "workflow_id",
        "project_id",
        "session_id",
        "status",
        "messages",
        "task",
    ]

    for field in required_fields:
        if field not in state:
            logger.error(f"State validation failed: missing field '{field}'")
            return False

    # Validate status
    try:
        WorkflowStatus(state["status"])
    except ValueError:
        logger.error(f"Invalid workflow status: {state['status']}")
        return False

    return True


def update_state(
    state: AgentState,
    updates: Dict[str, Any],
) -> AgentState:
    """
    Update state immutably with new values.

    CRITICAL: Returns new state dict, does not mutate original.

    Args:
        state: Current state
        updates: Updates to apply

    Returns:
        New state dict with updates applied
    """
    # Create new state dict (immutable update)
    new_state: AgentState = {**state, **updates}

    # Always update timestamp and version
    new_state["updated_at"] = datetime.now().isoformat()
    new_state["state_version"] = state.get("state_version", 1) + 1

    logger.debug(f"State updated to version {new_state['state_version']}")
    return new_state


def serialize_state(state: AgentState) -> Dict[str, Any]:
    """
    Serialize state for storage or transmission.

    Args:
        state: State to serialize

    Returns:
        Serialized state dictionary
    """
    # State is already a dict, just ensure all values are serializable
    serialized = dict(state)

    # Convert any non-serializable types if needed
    # For now, state only contains basic types

    return serialized


def deserialize_state(data: Dict[str, Any]) -> AgentState:
    """
    Deserialize state from storage.

    Args:
        data: Serialized state data

    Returns:
        AgentState TypedDict
    """
    # Ensure all required fields exist with defaults
    state: AgentState = {
        "workflow_id": data.get("workflow_id", ""),
        "project_id": data.get("project_id", ""),
        "session_id": data.get("session_id", ""),
        "status": data.get("status", WorkflowStatus.PENDING.value),
        "messages": data.get("messages", []),
        "current_agent": data.get("current_agent", AgentRole.COORDINATOR.value),
        "next_agent": data.get("next_agent"),
        "task": data.get("task", {}),
        "subtasks": data.get("subtasks", []),
        "completed_subtasks": data.get("completed_subtasks", []),
        "plan": data.get("plan"),
        "architecture": data.get("architecture"),
        "implementation": data.get("implementation"),
        "test_results": data.get("test_results"),
        "validation_results": data.get("validation_results"),
        "debug_info": data.get("debug_info"),
        "retry_count": data.get("retry_count", 0),
        "max_retries": data.get("max_retries", 3),
        "should_continue": data.get("should_continue", True),
        "requires_human_approval": data.get("requires_human_approval", False),
        "human_feedback": data.get("human_feedback"),
        "swarm_checkpoint_id": data.get("swarm_checkpoint_id", str(uuid.uuid4())),
        "parent_checkpoint_id": data.get("parent_checkpoint_id"),
        "state_version": data.get("state_version", 1),
        "memory_ids": data.get("memory_ids", []),
        "context": data.get("context", {}),
        "created_at": data.get("created_at", datetime.now().isoformat()),
        "updated_at": data.get("updated_at", datetime.now().isoformat()),
        "elapsed_time": data.get("elapsed_time", 0.0),
        "token_count": data.get("token_count", 0),
        "cost": data.get("cost", 0.0),
    }

    return state


def get_state_summary(state: AgentState) -> Dict[str, Any]:
    """
    Get a summary of current state for logging/display.

    Args:
        state: Current state

    Returns:
        Summary dictionary
    """
    return {
        "workflow_id": state.get("workflow_id"),
        "status": state.get("status"),
        "current_agent": state.get("current_agent"),
        "next_agent": state.get("next_agent"),
        "state_version": state.get("state_version"),
        "retry_count": state.get("retry_count"),
        "messages_count": len(state.get("messages", [])),
        "requires_approval": state.get("requires_human_approval"),
    }
