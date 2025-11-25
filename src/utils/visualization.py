"""Graph visualization utilities for workflow debugging."""

import logging
from typing import Dict, Any, List, Optional

from ..models.state_models import AgentState

logger = logging.getLogger(__name__)


def generate_mermaid_diagram(
    workflow_graph: Any,
    include_state: bool = False,
    state: Optional[AgentState] = None,
) -> str:
    """
    Generate Mermaid diagram for workflow visualization.

    Args:
        workflow_graph: Compiled LangGraph workflow
        include_state: Include current state information
        state: Optional current state

    Returns:
        Mermaid diagram as string
    """
    try:
        # Use LangGraph's built-in Mermaid generation
        mermaid = workflow_graph.get_graph().draw_mermaid()

        # Add state information if requested
        if include_state and state:
            state_info = _generate_state_info(state)
            mermaid += f"\n\n%% Current State\n{state_info}"

        return mermaid

    except Exception as e:
        logger.error(f"Failed to generate Mermaid diagram: {e}")
        return generate_fallback_diagram()


def _generate_state_info(state: AgentState) -> str:
    """
    Generate state information for diagram annotation.

    Args:
        state: Current state

    Returns:
        State info string
    """
    return (
        f"%% Workflow: {state.get('workflow_id')}\n"
        f"%% Status: {state.get('status')}\n"
        f"%% Current Agent: {state.get('current_agent')}\n"
        f"%% Version: {state.get('state_version')}"
    )


def generate_fallback_diagram() -> str:
    """
    Generate fallback diagram when auto-generation fails.

    Returns:
        Basic Mermaid diagram
    """
    return """
graph TD
    Start([Start]) --> Coordinator[Coordinator]
    Coordinator --> Planner[Planner]
    Planner --> Architect[Architect]
    Architect --> Implementer[Implementer]
    Implementer --> Tester[Tester]
    Tester -->|Pass| Validator[Validator]
    Tester -->|Fail| Debugger[Debugger]
    Debugger --> Tester
    Validator -->|Approved| End([Complete])
    Validator -->|Rejected| Debugger

    style Start fill:#90EE90
    style End fill:#FFB6C1
    style Planner fill:#FFE4B5
    style Architect fill:#FFE4B5
    style Validator fill:#FFE4B5
"""


def visualize_workflow(
    workflow_graph: Any,
    output_file: Optional[str] = None,
    state: Optional[AgentState] = None,
) -> str:
    """
    Visualize workflow and optionally save to file.

    Args:
        workflow_graph: Compiled workflow
        output_file: Optional file path to save
        state: Optional current state

    Returns:
        Mermaid diagram string
    """
    diagram = generate_mermaid_diagram(
        workflow_graph=workflow_graph,
        include_state=True,
        state=state,
    )

    if output_file:
        try:
            with open(output_file, "w") as f:
                f.write(diagram)
            logger.info(f"Saved workflow diagram to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save diagram: {e}")

    return diagram


def generate_execution_timeline(
    state: AgentState,
) -> List[Dict[str, Any]]:
    """
    Generate execution timeline from state messages.

    Args:
        state: Current workflow state

    Returns:
        List of timeline events
    """
    messages = state.get("messages", [])

    timeline = []
    for msg in messages:
        timeline.append(
            {
                "agent": msg.get("role", "unknown"),
                "type": msg.get("type", "info"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", ""),
            }
        )

    return timeline


def format_state_for_display(state: AgentState) -> Dict[str, Any]:
    """
    Format state for human-readable display.

    Args:
        state: Workflow state

    Returns:
        Formatted state dictionary
    """
    return {
        "Workflow ID": state.get("workflow_id"),
        "Project ID": state.get("project_id"),
        "Status": state.get("status"),
        "Current Agent": state.get("current_agent"),
        "State Version": state.get("state_version"),
        "Messages": len(state.get("messages", [])),
        "Retry Count": state.get("retry_count"),
        "Requires Approval": state.get("requires_human_approval"),
        "Created": state.get("created_at"),
        "Updated": state.get("updated_at"),
        "Elapsed Time": f"{state.get('elapsed_time', 0):.2f}s",
    }
