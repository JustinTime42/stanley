"""Node wrapper functions for LangGraph integration.

Each node wraps an agent's execute method and handles state updates.
CRITICAL: Nodes must return state updates dict, not mutate state directly.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from ..models.state_models import AgentState
from ..agents import (
    CoordinatorAgent,
    PlannerAgent,
    ArchitectAgent,
    ImplementerAgent,
    TesterAgent,
    ValidatorAgent,
    DebuggerAgent,
)
from ..services.memory_service import MemoryOrchestrator

logger = logging.getLogger(__name__)


async def coordinator_node(
    state: AgentState,
    memory_service: "MemoryOrchestrator" = None,
    llm_service: "LLMOrchestrator" = None,
    tool_service: "ToolOrchestrator" = None,
) -> Dict[str, Any]:
    """
    Coordinator node wrapper.

    CRITICAL: Must return state updates dict, not mutate state.

    Args:
        state: Current workflow state
        memory_service: Optional memory service
        llm_service: Optional LLM service
        tool_service: Optional tool service

    Returns:
        State updates dictionary
    """
    logger.info("Executing coordinator node")

    try:
        # Coordinator only uses memory_service
        agent = CoordinatorAgent(memory_service=memory_service)
        response = await agent.execute(state)

        # Return immutable state updates
        updates = {
            "messages": state.get("messages", []) + response.messages,
            "current_agent": agent.role.value,
            "updated_at": datetime.now().isoformat(),
        }

        # Merge in agent's state updates
        updates.update(response.state_updates)

        return updates

    except Exception as e:
        logger.error(f"Coordinator node failed: {e}")
        return {
            "messages": state.get("messages", [])
            + [
                {
                    "role": "coordinator",
                    "content": f"Error: {str(e)}",
                    "type": "error",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "status": "failed",
        }


async def planner_node(
    state: AgentState,
    memory_service: "MemoryOrchestrator" = None,
    llm_service: "LLMOrchestrator" = None,
    tool_service: "ToolOrchestrator" = None,
) -> Dict[str, Any]:
    """
    Planner node wrapper.

    Args:
        state: Current workflow state
        memory_service: Optional memory service
        llm_service: Optional LLM service
        tool_service: Optional tool service

    Returns:
        State updates dictionary
    """
    logger.info("Executing planner node")

    try:
        agent = PlannerAgent(
            memory_service=memory_service,
            llm_service=llm_service,
        )
        response = await agent.execute(state)

        updates = {
            "messages": state.get("messages", []) + response.messages,
            "current_agent": agent.role.value,
            "updated_at": datetime.now().isoformat(),
        }

        updates.update(response.state_updates)
        return updates

    except Exception as e:
        logger.error(f"Planner node failed: {e}")
        return {
            "messages": state.get("messages", [])
            + [
                {
                    "role": "planner",
                    "content": f"Error: {str(e)}",
                    "type": "error",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "status": "failed",
        }


async def architect_node(
    state: AgentState,
    memory_service: "MemoryOrchestrator" = None,
    llm_service: "LLMOrchestrator" = None,
    tool_service: "ToolOrchestrator" = None,
) -> Dict[str, Any]:
    """
    Architect node wrapper.

    Args:
        state: Current workflow state
        memory_service: Optional memory service
        llm_service: Optional LLM service
        tool_service: Optional tool service

    Returns:
        State updates dictionary
    """
    logger.info("Executing architect node")

    try:
        agent = ArchitectAgent(
            memory_service=memory_service,
            llm_service=llm_service,
        )
        response = await agent.execute(state)

        updates = {
            "messages": state.get("messages", []) + response.messages,
            "current_agent": agent.role.value,
            "updated_at": datetime.now().isoformat(),
        }

        updates.update(response.state_updates)
        return updates

    except Exception as e:
        logger.error(f"Architect node failed: {e}")
        return {
            "messages": state.get("messages", [])
            + [
                {
                    "role": "architect",
                    "content": f"Error: {str(e)}",
                    "type": "error",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "status": "failed",
        }


async def implementer_node(
    state: AgentState,
    memory_service: "MemoryOrchestrator" = None,
    llm_service: "LLMOrchestrator" = None,
    tool_service: "ToolOrchestrator" = None,
) -> Dict[str, Any]:
    """
    Implementer node wrapper.

    Args:
        state: Current workflow state
        memory_service: Optional memory service
        llm_service: Optional LLM service
        tool_service: Optional tool service

    Returns:
        State updates dictionary
    """
    logger.info("Executing implementer node")

    try:
        agent = ImplementerAgent(
            memory_service=memory_service,
            llm_service=llm_service,
            tool_service=tool_service,
        )
        response = await agent.execute(state)

        updates = {
            "messages": state.get("messages", []) + response.messages,
            "current_agent": agent.role.value,
            "updated_at": datetime.now().isoformat(),
        }

        updates.update(response.state_updates)
        return updates

    except Exception as e:
        logger.error(f"Implementer node failed: {e}")
        return {
            "messages": state.get("messages", [])
            + [
                {
                    "role": "implementer",
                    "content": f"Error: {str(e)}",
                    "type": "error",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "status": "failed",
        }


async def tester_node(
    state: AgentState,
    memory_service: "MemoryOrchestrator" = None,
    llm_service: "LLMOrchestrator" = None,
    tool_service: "ToolOrchestrator" = None,
) -> Dict[str, Any]:
    """
    Tester node wrapper.

    Args:
        state: Current workflow state
        memory_service: Optional memory service
        llm_service: Optional LLM service
        tool_service: Optional tool service

    Returns:
        State updates dictionary
    """
    logger.info("Executing tester node")

    try:
        agent = TesterAgent(
            memory_service=memory_service,
            # Note: TesterAgent uses its own TestingOrchestrator internally
        )
        response = await agent.execute(state)

        updates = {
            "messages": state.get("messages", []) + response.messages,
            "current_agent": agent.role.value,
            "updated_at": datetime.now().isoformat(),
        }

        updates.update(response.state_updates)
        return updates

    except Exception as e:
        logger.error(f"Tester node failed: {e}")
        return {
            "messages": state.get("messages", [])
            + [
                {
                    "role": "tester",
                    "content": f"Error: {str(e)}",
                    "type": "error",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "status": "failed",
        }


async def validator_node(
    state: AgentState,
    memory_service: "MemoryOrchestrator" = None,
    llm_service: "LLMOrchestrator" = None,
    tool_service: "ToolOrchestrator" = None,
) -> Dict[str, Any]:
    """
    Validator node wrapper.

    Args:
        state: Current workflow state
        memory_service: Optional memory service
        llm_service: Optional LLM service
        tool_service: Optional tool service

    Returns:
        State updates dictionary
    """
    logger.info("Executing validator node")

    try:
        agent = ValidatorAgent(
            memory_service=memory_service,
        )
        response = await agent.execute(state)

        updates = {
            "messages": state.get("messages", []) + response.messages,
            "current_agent": agent.role.value,
            "updated_at": datetime.now().isoformat(),
        }

        updates.update(response.state_updates)
        return updates

    except Exception as e:
        logger.error(f"Validator node failed: {e}")
        return {
            "messages": state.get("messages", [])
            + [
                {
                    "role": "validator",
                    "content": f"Error: {str(e)}",
                    "type": "error",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "status": "failed",
        }


async def debugger_node(
    state: AgentState,
    memory_service: "MemoryOrchestrator" = None,
    llm_service: "LLMOrchestrator" = None,
    tool_service: "ToolOrchestrator" = None,
) -> Dict[str, Any]:
    """
    Debugger node wrapper.

    Args:
        state: Current workflow state
        memory_service: Optional memory service
        llm_service: Optional LLM service
        tool_service: Optional tool service

    Returns:
        State updates dictionary
    """
    logger.info("Executing debugger node")

    try:
        agent = DebuggerAgent(
            memory_service=memory_service,
            llm_service=llm_service,
            tool_service=tool_service,
        )
        response = await agent.execute(state)

        updates = {
            "messages": state.get("messages", []) + response.messages,
            "current_agent": agent.role.value,
            "updated_at": datetime.now().isoformat(),
        }

        updates.update(response.state_updates)
        return updates

    except Exception as e:
        logger.error(f"Debugger node failed: {e}")
        return {
            "messages": state.get("messages", [])
            + [
                {
                    "role": "debugger",
                    "content": f"Error: {str(e)}",
                    "type": "error",
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            "status": "failed",
        }
