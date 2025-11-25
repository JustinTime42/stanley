"""State manipulation and helper utilities."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..models.state_models import AgentState, WorkflowStatus

logger = logging.getLogger(__name__)


def merge_state_updates(
    current_state: AgentState,
    updates: Dict[str, Any],
) -> AgentState:
    """
    Merge state updates immutably.

    CRITICAL: Returns new state dict, does not mutate original.

    Args:
        current_state: Current state
        updates: Updates to apply

    Returns:
        New state dict with updates
    """
    # Create new state dict (immutable update)
    new_state: AgentState = {**current_state, **updates}

    # Always update timestamp and version
    new_state["updated_at"] = datetime.now().isoformat()
    new_state["state_version"] = current_state.get("state_version", 1) + 1

    return new_state


def append_message(
    state: AgentState,
    message: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Append message to state (for node updates).

    Args:
        state: Current state
        message: Message to append

    Returns:
        State update dict with new message
    """
    current_messages = state.get("messages", [])
    return {
        "messages": current_messages + [message],
        "updated_at": datetime.now().isoformat(),
    }


def increment_retry_count(state: AgentState) -> Dict[str, Any]:
    """
    Increment retry count.

    Args:
        state: Current state

    Returns:
        State update dict
    """
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "updated_at": datetime.now().isoformat(),
    }


def mark_subtask_complete(
    state: AgentState,
    subtask_id: str,
) -> Dict[str, Any]:
    """
    Mark subtask as complete.

    Args:
        state: Current state
        subtask_id: Subtask identifier

    Returns:
        State update dict
    """
    completed = state.get("completed_subtasks", [])
    if subtask_id not in completed:
        completed = completed + [subtask_id]

    return {
        "completed_subtasks": completed,
        "updated_at": datetime.now().isoformat(),
    }


def calculate_progress(state: AgentState) -> float:
    """
    Calculate workflow progress percentage.

    Args:
        state: Current state

    Returns:
        Progress percentage (0-100)
    """
    status = state.get("status", "")
    total_subtasks = len(state.get("subtasks", []))
    completed_subtasks = len(state.get("completed_subtasks", []))

    # Base progress on status
    status_progress = {
        WorkflowStatus.PENDING.value: 0,
        WorkflowStatus.PLANNING.value: 10,
        WorkflowStatus.DESIGNING.value: 25,
        WorkflowStatus.IMPLEMENTING.value: 50,
        WorkflowStatus.TESTING.value: 70,
        WorkflowStatus.VALIDATING.value: 85,
        WorkflowStatus.DEBUGGING.value: 60,
        WorkflowStatus.COMPLETE.value: 100,
        WorkflowStatus.FAILED.value: 0,
    }

    base_progress = status_progress.get(status, 0)

    # Adjust based on subtask completion
    if total_subtasks > 0:
        subtask_progress = (completed_subtasks / total_subtasks) * 100
        return (base_progress + subtask_progress) / 2

    return base_progress


def is_terminal_state(state: AgentState) -> bool:
    """
    Check if state is terminal (workflow done).

    Args:
        state: Current state

    Returns:
        True if terminal
    """
    status = state.get("status", "")
    return status in [
        WorkflowStatus.COMPLETE.value,
        WorkflowStatus.FAILED.value,
    ]


def get_active_agent(state: AgentState) -> Optional[str]:
    """
    Get currently active agent.

    Args:
        state: Current state

    Returns:
        Agent role or None
    """
    return state.get("current_agent")


def get_next_agent(state: AgentState) -> Optional[str]:
    """
    Get next scheduled agent.

    Args:
        state: Current state

    Returns:
        Agent role or None
    """
    return state.get("next_agent")


def filter_messages_by_type(
    state: AgentState,
    message_type: str,
) -> List[Dict[str, Any]]:
    """
    Filter messages by type.

    Args:
        state: Current state
        message_type: Message type (info, error, warning, success)

    Returns:
        Filtered messages
    """
    messages = state.get("messages", [])
    return [msg for msg in messages if msg.get("type") == message_type]


def get_error_messages(state: AgentState) -> List[Dict[str, Any]]:
    """
    Get all error messages.

    Args:
        state: Current state

    Returns:
        Error messages
    """
    return filter_messages_by_type(state, "error")


def has_errors(state: AgentState) -> bool:
    """
    Check if state has error messages.

    Args:
        state: Current state

    Returns:
        True if errors exist
    """
    return len(get_error_messages(state)) > 0


def estimate_remaining_time(state: AgentState) -> Optional[float]:
    """
    Estimate remaining time based on progress.

    Args:
        state: Current state

    Returns:
        Estimated seconds remaining or None
    """
    progress = calculate_progress(state)
    elapsed = state.get("elapsed_time", 0.0)

    if progress > 0 and elapsed > 0:
        total_estimated = (elapsed / progress) * 100
        remaining = total_estimated - elapsed
        return max(0, remaining)

    return None
