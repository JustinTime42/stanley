"""
Agent Swarm System Demo

This script demonstrates the Agent Swarm system with:
- LangGraph workflow with 7 specialized agents
- State management and routing
- Workflow visualization
- Agent coordination and routing logic

Run this from the project root:
    python run.py
"""

import sys
from pathlib import Path

# Add project root to Python path so we can import from src as a package
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import asyncio
import logging

# Import from src as a package to avoid relative import issues
from src.core import workflow as workflow_module
from src.core import state as state_module
from src.core import edges as edges_module
from src.utils import state_utils as state_utils_module
from src.utils import visualization as viz_module
from src.models.state_models import WorkflowStatus

# Configure logging - suppress warnings for cleaner demo output
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to suppress agent warnings
    format="%(levelname)s - %(message)s",
)

# Create a custom logger for demo output
demo_logger = logging.getLogger("demo")
demo_logger.setLevel(logging.INFO)
demo_logger.propagate = False  # Prevent duplicate logging
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
demo_logger.addHandler(handler)


async def demo_workflow_structure():
    """Demonstrate workflow structure and agent graph."""
    demo_logger.info("=" * 70)
    demo_logger.info("DEMO 1: Workflow Structure & Agent Graph")
    demo_logger.info("=" * 70)

    # Create workflow without external dependencies
    demo_logger.info("\nCreating workflow with 7 specialized agents...")
    workflow = workflow_module.create_workflow(
        checkpointer=None,
        memory_service=None,
        enable_human_approval=False,
    )

    # Get graph structure
    graph = workflow.get_graph()
    nodes = list(graph.nodes.keys()) if hasattr(graph.nodes, 'keys') else list(graph.nodes)
    agent_nodes = [n for n in nodes if n not in ['__start__', '__end__']]

    demo_logger.info(f"✓ Workflow compiled successfully")
    demo_logger.info(f"✓ Total nodes: {len(nodes)} ({len(agent_nodes)} agents + start/end)")
    demo_logger.info(f"\nAgent nodes:")
    for agent in agent_nodes:
        demo_logger.info(f"  • {agent}")

    # Generate visualization
    demo_logger.info("\nGenerating workflow visualization...")
    try:
        diagram = viz_module.generate_mermaid_diagram(workflow)
        output_file = project_root / "workflow_demo.mermaid"
        with open(output_file, "w") as f:
            f.write(diagram)
        demo_logger.info(f"✓ Saved diagram to: {output_file.name}")
        demo_logger.info("  View at: https://mermaid.live/")
    except Exception as e:
        demo_logger.warning(f"Could not generate diagram: {e}")


async def demo_state_management():
    """Demonstrate state management capabilities."""
    demo_logger.info("\n" + "=" * 70)
    demo_logger.info("DEMO 2: State Management & Versioning")
    demo_logger.info("=" * 70)

    task = {
        "id": "demo_task",
        "description": "Build a REST API with authentication",
        "requirements": ["User registration", "JWT tokens", "Rate limiting"],
    }

    # Create initial state
    demo_logger.info("\nCreating initial workflow state...")
    state = state_module.create_initial_state(
        task=task,
        project_id="demo_project",
    )

    demo_logger.info(f"✓ Workflow ID: {state['workflow_id']}")
    demo_logger.info(f"✓ Project ID: {state['project_id']}")
    demo_logger.info(f"✓ Initial status: {state['status']}")
    demo_logger.info(f"✓ State version: {state['state_version']}")

    # Demonstrate state updates
    demo_logger.info("\nDemonstrating immutable state updates...")

    updated = state_module.update_state(state, {
        "status": WorkflowStatus.PLANNING.value,
        "current_agent": "planner",
    })
    demo_logger.info(f"✓ Updated to PLANNING (version {updated['state_version']})")

    updated2 = state_module.update_state(updated, {
        "status": WorkflowStatus.IMPLEMENTING.value,
        "current_agent": "implementer",
    })
    demo_logger.info(f"✓ Updated to IMPLEMENTING (version {updated2['state_version']})")

    # Show progress calculation
    progress = state_utils_module.calculate_progress(updated2)
    demo_logger.info(f"✓ Workflow progress: {progress:.0f}%")


async def demo_routing_logic():
    """Demonstrate conditional routing between agents."""
    demo_logger.info("\n" + "=" * 70)
    demo_logger.info("DEMO 3: Intelligent Agent Routing")
    demo_logger.info("=" * 70)

    task = {"id": "routing_demo", "description": "Test routing logic"}

    demo_logger.info("\nDemonstrating conditional routing between agents...")

    # Test different routing scenarios
    demo_logger.info("\n1. Coordinator → Planner (pending task)")
    state = state_module.create_initial_state(task=task, project_id="demo")
    next_agent = edges_module.route_from_coordinator(state)
    demo_logger.info(f"   Route: coordinator → {next_agent}")

    # Test tester routing (passing tests)
    demo_logger.info("\n2. Tester → Validator (all tests passing)")
    state = state_module.update_state(state, {
        "status": WorkflowStatus.TESTING.value,
        "test_results": {"all_passed": True, "total": 10, "passed": 10},
    })
    next_agent = edges_module.route_from_tester(state)
    demo_logger.info(f"   Route: tester → {next_agent}")

    # Test tester routing (failing tests)
    demo_logger.info("\n3. Tester → Debugger (tests failing)")
    state = state_module.update_state(state, {
        "test_results": {"all_passed": False, "total": 10, "passed": 7, "failed": 3},
    })
    next_agent = edges_module.route_from_tester(state)
    demo_logger.info(f"   Route: tester → {next_agent}")

    # Test validator routing (approved)
    demo_logger.info("\n4. Validator → End (validation approved)")
    state = state_module.update_state(state, {
        "status": WorkflowStatus.VALIDATING.value,
        "validation_results": {"approved": True, "score": 0.95},
    })
    next_agent = edges_module.route_from_validator(state)
    demo_logger.info(f"   Route: validator → {next_agent}")

    # Test validator routing (needs revision)
    demo_logger.info("\n5. Validator → Debugger (validation failed)")
    state = state_module.update_state(state, {
        "validation_results": {"approved": False, "issues": ["Style violations"]},
    })
    next_agent = edges_module.route_from_validator(state)
    demo_logger.info(f"   Route: validator → {next_agent}")


async def demo_summary():
    """Show system capabilities summary."""
    demo_logger.info("\n" + "=" * 70)
    demo_logger.info("AGENT SWARM SYSTEM - FEATURES")
    demo_logger.info("=" * 70)

    demo_logger.info("\nCore Features:")
    demo_logger.info("  ✓ 7 Specialized Agents")
    demo_logger.info("    • Coordinator - Workflow orchestration")
    demo_logger.info("    • Planner - Task decomposition")
    demo_logger.info("    • Architect - System design")
    demo_logger.info("    • Implementer - Code generation")
    demo_logger.info("    • Tester - Test generation & execution")
    demo_logger.info("    • Validator - Quality assurance")
    demo_logger.info("    • Debugger - Issue resolution")

    demo_logger.info("\n  ✓ LangGraph Workflow Engine")
    demo_logger.info("    • Compiled state graphs")
    demo_logger.info("    • Conditional routing logic")
    demo_logger.info("    • Async execution support")

    demo_logger.info("\n  ✓ State Management")
    demo_logger.info("    • Immutable state updates")
    demo_logger.info("    • Version tracking")
    demo_logger.info("    • Progress calculation")
    demo_logger.info("    • State serialization")

    demo_logger.info("\n  ✓ Memory System (Optional)")
    demo_logger.info("    • Working Memory (Redis)")
    demo_logger.info("    • Project Memory (Qdrant)")
    demo_logger.info("    • Global Memory (Qdrant)")
    demo_logger.info("    • Semantic caching")

    demo_logger.info("\n  ✓ Advanced Features")
    demo_logger.info("    • Checkpoint/rollback")
    demo_logger.info("    • Human-in-the-loop approval")
    demo_logger.info("    • Workflow visualization")
    demo_logger.info("    • Cost tracking")
    demo_logger.info("    • Model routing")

    demo_logger.info("\nFor More Information:")
    demo_logger.info("  • README.md - Full documentation")
    demo_logger.info("  • scripts/demo_workflow.py - Detailed feature demos")
    demo_logger.info("  • scripts/test_end_to_end.py - Integration tests")
    demo_logger.info("=" * 70 + "\n")


async def main():
    """Main entry point - run all demos."""
    try:
        demo_logger.info("\n" + "=" * 70)
        demo_logger.info(" " * 15 + "AGENT SWARM SYSTEM DEMO")
        demo_logger.info("=" * 70)

        # Run all demo sections
        await demo_workflow_structure()
        await demo_state_management()
        await demo_routing_logic()
        await demo_summary()

    except KeyboardInterrupt:
        demo_logger.info("\n\nDemo interrupted by user")
    except Exception as e:
        import traceback
        demo_logger.error(f"\n\nDemo failed with error: {e}")
        demo_logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
