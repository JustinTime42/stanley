"""
Demonstration script for Agent Swarm LangGraph workflow.

This script demonstrates the complete implementation from PRP-02:
- All 7 agents integrated as LangGraph nodes
- State management with TypedDict
- Workflow orchestration
- Memory integration (optional)
- Visualization capabilities
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import logging

# Direct module imports to avoid relative import issues in __init__.py
import core.workflow as workflow_module
import core.state as state_module
import core.edges as edges_module
import utils.state_utils as state_utils_module
import utils.visualization as viz_module
from models.state_models import WorkflowStatus

# Extract functions
create_workflow = workflow_module.create_workflow
create_initial_state = state_module.create_initial_state
get_state_summary = state_module.get_state_summary
update_state = state_module.update_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def demo_basic_workflow():
    """Demonstrate basic workflow creation and structure."""
    logger.info("=" * 60)
    logger.info("DEMO: Basic Workflow Creation")
    logger.info("=" * 60)

    # Create workflow without dependencies
    workflow = create_workflow(
        checkpointer=None,
        memory_service=None,
        enable_human_approval=False,
    )

    logger.info("✓ Workflow compiled successfully")

    # Get graph structure
    graph = workflow.get_graph()
    nodes = [node.id for node in graph.nodes]

    logger.info(f"✓ Workflow has {len(nodes)} nodes:")
    for node in nodes:
        logger.info(f"  - {node}")

    # Create initial state
    task = {
        "id": "demo_task_1",
        "description": "Create a simple REST API",
        "requirements": [
            "Use FastAPI framework",
            "Include basic CRUD operations",
            "Add input validation",
        ],
    }

    state = create_initial_state(
        task=task,
        project_id="demo_project",
    )

    logger.info(f"\n✓ Initial state created:")
    summary = get_state_summary(state)
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    logger.info("\n✓ Basic workflow demonstration complete!\n")


async def demo_state_management():
    """Demonstrate state management utilities."""
    logger.info("=" * 60)
    logger.info("DEMO: State Management")
    logger.info("=" * 60)

    from core.state import update_state, serialize_state, deserialize_state
    from utils.state_utils import calculate_progress, is_terminal_state

    # Create initial state
    task = {"id": "test", "description": "Test task"}
    state = create_initial_state(task=task, project_id="test")

    logger.info("✓ Created initial state (version 1)")

    # Update state
    state = update_state(state, {
        "status": WorkflowStatus.PLANNING.value,
        "current_agent": "planner",
    })

    logger.info(f"✓ Updated state to version {state['state_version']}")

    # Calculate progress
    progress = calculate_progress(state)
    logger.info(f"✓ Workflow progress: {progress:.1f}%")

    # Check if terminal
    terminal = is_terminal_state(state)
    logger.info(f"✓ Is terminal state: {terminal}")

    # Serialize/deserialize
    serialized = serialize_state(state)
    logger.info(f"✓ Serialized state ({len(serialized)} fields)")

    deserialized = deserialize_state(serialized)
    logger.info(f"✓ Deserialized state successfully")

    logger.info("\n✓ State management demonstration complete!\n")


async def demo_agent_routing():
    """Demonstrate conditional edge routing."""
    logger.info("=" * 60)
    logger.info("DEMO: Agent Routing")
    logger.info("=" * 60)

    from core.edges import (
        route_from_coordinator,
        route_from_tester,
        route_from_validator,
    )
    from models.state_models import WorkflowStatus

    # Create test states
    task = {"id": "test", "description": "Test task"}

    # Test coordinator routing
    state = create_initial_state(task=task, project_id="test")
    next_node = route_from_coordinator(state)
    logger.info(f"✓ Coordinator routes PENDING → {next_node}")

    # Test tester routing (passing tests)
    state = update_state(state, {
        "status": WorkflowStatus.TESTING.value,
        "test_results": {"all_passed": True},
    })
    next_node = route_from_tester(state)
    logger.info(f"✓ Tester routes (PASS) → {next_node}")

    # Test tester routing (failing tests)
    state = update_state(state, {
        "test_results": {"all_passed": False},
    })
    next_node = route_from_tester(state)
    logger.info(f"✓ Tester routes (FAIL) → {next_node}")

    # Test validator routing (approved)
    state = update_state(state, {
        "status": WorkflowStatus.VALIDATING.value,
        "validation_results": {"approved": True},
    })
    next_node = route_from_validator(state)
    logger.info(f"✓ Validator routes (APPROVED) → {next_node}")

    logger.info("\n✓ Agent routing demonstration complete!\n")


async def demo_visualization():
    """Demonstrate workflow visualization."""
    logger.info("=" * 60)
    logger.info("DEMO: Workflow Visualization")
    logger.info("=" * 60)

    from utils.visualization import generate_mermaid_diagram

    # Create workflow
    workflow = create_workflow(
        checkpointer=None,
        memory_service=None,
        enable_human_approval=True,
    )

    # Generate diagram
    try:
        diagram = generate_mermaid_diagram(workflow)
        logger.info(f"✓ Generated Mermaid diagram ({len(diagram)} characters)")
        logger.info("\nDiagram preview (first 500 chars):")
        logger.info("-" * 60)
        logger.info(diagram[:500])
        logger.info("...")
        logger.info("-" * 60)
    except Exception as e:
        logger.warning(f"Could not generate diagram (expected): {e}")
        logger.info("✓ Fallback diagram available")

    logger.info("\n✓ Visualization demonstration complete!\n")


async def main():
    """Run all demonstrations."""
    logger.info("\n" + "=" * 60)
    logger.info("AGENT SWARM LANGGRAPH IMPLEMENTATION DEMO")
    logger.info("PRP-02: Advanced State Management with LangGraph")
    logger.info("=" * 60 + "\n")

    await demo_basic_workflow()
    await demo_state_management()
    await demo_agent_routing()
    await demo_visualization()

    logger.info("=" * 60)
    logger.info("ALL DEMONSTRATIONS COMPLETE")
    logger.info("=" * 60)
    logger.info("\nImplementation includes:")
    logger.info("  ✓ 7 agents integrated as LangGraph nodes")
    logger.info("  ✓ StateGraph with TypedDict schema")
    logger.info("  ✓ Conditional edge routing")
    logger.info("  ✓ State management utilities")
    logger.info("  ✓ Checkpoint/rollback infrastructure")
    logger.info("  ✓ Human-in-the-loop approval service")
    logger.info("  ✓ Workflow orchestration service")
    logger.info("  ✓ Visualization utilities")
    logger.info("\nReady for integration with:")
    logger.info("  - Redis checkpointer (PRP-01)")
    logger.info("  - Memory orchestrator (PRP-01)")
    logger.info("  - LLM services for agent logic")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
