"""Conditional edge logic for workflow routing.

CRITICAL: Edge functions must return node name as string, not function references.
"""

import logging
from typing import Literal

from ..models.state_models import AgentState, WorkflowStatus

logger = logging.getLogger(__name__)

# Type for edge routing decisions
EdgeDecision = Literal[
    "planner",
    "architect",
    "implementer",
    "tester",
    "validator",
    "debugger",
    "coordinator",
    "end",
]


def route_from_coordinator(state: AgentState) -> EdgeDecision:
    """
    Route from coordinator to next agent.

    CRITICAL: Must return node name as string.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    status = state.get("status", WorkflowStatus.PENDING.value)
    next_agent = state.get("next_agent")

    logger.debug(f"Routing from coordinator: status={status}, next={next_agent}")

    # Use explicit next_agent if set
    if next_agent:
        return next_agent  # type: ignore

    # Otherwise route based on status
    routing_map = {
        WorkflowStatus.PENDING.value: "planner",
        WorkflowStatus.PLANNING.value: "planner",
        WorkflowStatus.DESIGNING.value: "architect",
        WorkflowStatus.IMPLEMENTING.value: "implementer",
        WorkflowStatus.TESTING.value: "tester",
        WorkflowStatus.VALIDATING.value: "validator",
        WorkflowStatus.DEBUGGING.value: "debugger",
        WorkflowStatus.COMPLETE.value: "end",
        WorkflowStatus.FAILED.value: "end",
    }

    next_node = routing_map.get(status, "end")
    logger.info(f"Coordinator routing to: {next_node}")
    return next_node  # type: ignore


def route_from_planner(state: AgentState) -> EdgeDecision:
    """
    Route from planner to next agent.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    # Planner always goes to architect after approval
    logger.info("Planner routing to: architect")
    return "architect"


def route_from_architect(state: AgentState) -> EdgeDecision:
    """
    Route from architect to next agent.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    # Architect always goes to implementer after approval
    logger.info("Architect routing to: implementer")
    return "implementer"


def route_from_implementer(state: AgentState) -> EdgeDecision:
    """
    Route from implementer to next agent.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    # Implementer always goes to tester
    logger.info("Implementer routing to: tester")
    return "tester"


def route_from_tester(state: AgentState) -> EdgeDecision:
    """
    Route from tester based on test results.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    test_results = state.get("test_results")
    if test_results is None:
        test_results = {}
    all_passed = test_results.get("all_passed", False)

    if all_passed:
        logger.info("Tests passed, routing to: validator")
        return "validator"
    else:
        logger.info("Tests failed, routing to: debugger")
        return "debugger"


def route_from_validator(state: AgentState) -> EdgeDecision:
    """
    Route from validator based on validation results.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    validation_results = state.get("validation_results") or {}
    approved = validation_results.get("approved", False)

    if approved:
        logger.info("Validation approved, workflow complete")
        return "end"
    else:
        logger.info("Validation failed, routing to: debugger")
        return "debugger"


def route_from_debugger(state: AgentState) -> EdgeDecision:
    """
    Route from debugger based on fixes.

    Args:
        state: Current workflow state

    Returns:
        Next node name
    """
    debug_info = state.get("debug_info") or {}
    fixed = debug_info.get("fixed", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if fixed and retry_count < max_retries:
        logger.info("Debugger fixed issues, routing to: tester")
        return "tester"
    else:
        logger.warning("Max retries exceeded or unfixed, ending workflow")
        return "end"


def should_retry(state: AgentState) -> bool:
    """
    Determine if workflow should retry after failure.

    Args:
        state: Current workflow state

    Returns:
        True if should retry
    """
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    return retry_count < max_retries


def needs_human_approval(state: AgentState) -> bool:
    """
    Determine if human approval is required.

    Args:
        state: Current workflow state

    Returns:
        True if human approval needed
    """
    return state.get("requires_human_approval", False)


def should_continue(state: AgentState) -> bool:
    """
    Determine if workflow should continue.

    Args:
        state: Current workflow state

    Returns:
        True if should continue
    """
    status = state.get("status", "")
    should_cont = state.get("should_continue", True)

    # Stop on terminal states
    if status in [WorkflowStatus.COMPLETE.value, WorkflowStatus.FAILED.value]:
        return False

    return should_cont
