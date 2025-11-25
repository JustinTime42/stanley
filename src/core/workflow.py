"""Main LangGraph workflow implementation.

CRITICAL patterns:
- Pass checkpointer to compile(), not invoke()
- Use TypedDict for state schema
- Conditional edges must return node names as strings
- Use interrupt_before for human approval points
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from ..models.state_models import AgentState
from ..services.memory_service import MemoryOrchestrator
from .nodes import (
    coordinator_node,
    planner_node,
    architect_node,
    implementer_node,
    tester_node,
    validator_node,
    debugger_node,
)
from .edges import (
    route_from_coordinator,
    route_from_planner,
    route_from_architect,
    route_from_implementer,
    route_from_tester,
    route_from_validator,
    route_from_debugger,
)

logger = logging.getLogger(__name__)


def create_workflow(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    memory_service: Optional[MemoryOrchestrator] = None,
    llm_service: Optional["LLMOrchestrator"] = None,
    tool_service: Optional["ToolOrchestrator"] = None,
    enable_human_approval: bool = True,
) -> StateGraph:
    """
    Create LangGraph workflow for agent swarm.

    CRITICAL: Pass checkpointer to compile(), not invoke()

    Args:
        checkpointer: Checkpoint saver for persistence
        memory_service: Memory orchestrator
        llm_service: LLM orchestrator for agent intelligence
        tool_service: Tool orchestrator for agent actions
        enable_human_approval: Enable human-in-the-loop approval

    Returns:
        Compiled StateGraph workflow
    """
    logger.info("Creating LangGraph workflow")

    # Create graph with AgentState schema
    # CRITICAL: Must use TypedDict, not Pydantic model
    graph = StateGraph(AgentState)

    # Wrap nodes with all services
    def wrap_node(node_func):
        """Wrap node function with all available services."""

        async def wrapped(state: AgentState):
            return await node_func(
                state,
                memory_service=memory_service,
                llm_service=llm_service,
                tool_service=tool_service,
            )

        return wrapped

    # Add nodes
    # CRITICAL: Use interrupt_before for human approval points
    graph.add_node("coordinator", wrap_node(coordinator_node))

    if enable_human_approval:
        # Planner, Architect, Validator require approval
        graph.add_node(
            "planner",
            wrap_node(planner_node),
            # Note: interrupt_before is set via graph configuration
        )
        graph.add_node(
            "architect",
            wrap_node(architect_node),
        )
        graph.add_node(
            "validator",
            wrap_node(validator_node),
        )
    else:
        graph.add_node("planner", wrap_node(planner_node))
        graph.add_node("architect", wrap_node(architect_node))
        graph.add_node("validator", wrap_node(validator_node))

    graph.add_node("implementer", wrap_node(implementer_node))
    graph.add_node("tester", wrap_node(tester_node))
    graph.add_node("debugger", wrap_node(debugger_node))

    # Add conditional edges
    # CRITICAL: Router functions must return node names as strings
    graph.add_conditional_edges(
        "coordinator",
        route_from_coordinator,
        {
            "planner": "planner",
            "architect": "architect",
            "implementer": "implementer",
            "tester": "tester",
            "validator": "validator",
            "debugger": "debugger",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "architect": "architect",
        },
    )

    graph.add_conditional_edges(
        "architect",
        route_from_architect,
        {
            "implementer": "implementer",
        },
    )

    graph.add_conditional_edges(
        "implementer",
        route_from_implementer,
        {
            "tester": "tester",
        },
    )

    graph.add_conditional_edges(
        "tester",
        route_from_tester,
        {
            "validator": "validator",
            "debugger": "debugger",
        },
    )

    graph.add_conditional_edges(
        "validator",
        route_from_validator,
        {
            "debugger": "debugger",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "debugger",
        route_from_debugger,
        {
            "tester": "tester",
            "end": END,
        },
    )

    # Set entry point
    graph.set_entry_point("coordinator")

    # Compile with checkpointer
    # CRITICAL: Pass checkpointer at compile time, not invoke time
    if checkpointer:
        logger.info("Compiling workflow with checkpointer")
        compiled = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["planner", "architect", "validator"]
            if enable_human_approval
            else None,
        )
    else:
        logger.info("Compiling workflow without checkpointer")
        compiled = graph.compile()

    logger.info("Workflow compiled successfully")
    return compiled


def create_workflow_with_config(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    memory_service: Optional[MemoryOrchestrator] = None,
    **config_kwargs,
):
    """
    Create workflow with additional configuration.

    Args:
        checkpointer: Checkpoint saver
        memory_service: Memory orchestrator
        **config_kwargs: Additional configuration

    Returns:
        Compiled workflow
    """
    enable_human_approval = config_kwargs.get("enable_human_approval", True)

    return create_workflow(
        checkpointer=checkpointer,
        memory_service=memory_service,
        enable_human_approval=enable_human_approval,
    )
