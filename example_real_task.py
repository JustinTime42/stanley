"""
Example: Using Agent Swarm for a Real Task

This script demonstrates how to use the Agent Swarm system to complete
a real software development task with actual LLM integration.

Requirements:
1. Set up your .env file with API keys (OpenAI, Ollama, or OpenRouter)
2. Optional: Start Redis and Qdrant for memory features
   - docker-compose -f docker/docker-compose.yml up -d

Usage:
    python example_real_task.py
"""

import sys
from pathlib import Path
import asyncio
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.llm_config import LLMConfig
from src.services.llm_service import LLMOrchestrator
from src.services.memory_service import MemoryOrchestrator
from src.services.workflow_service import WorkflowOrchestrator
from src.services.checkpoint_service import CheckpointManager
from src.config.memory_config import MemoryConfig
from src.models.workflow_models import WorkflowConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def setup_services():
    """
    Set up services for the agent swarm.

    NOTE: The workflow creates LLM/tool services internally per-agent.
    We just need to configure memory and checkpointing.

    Returns:
        Tuple of (memory_service, checkpoint_manager)
    """
    logger.info("Setting up services...")

    # 1. Check LLM availability (for information only)
    try:
        llm_config = LLMConfig()
        llm_service = LLMOrchestrator(config=llm_config)
        available_models = llm_service.get_available_models()
        logger.info(f"[OK] LLM Service: {len(available_models)} models available")
        for model_name in available_models[:3]:  # Show first 3
            logger.info(f"  - {model_name}")
        await llm_service.cleanup()  # Close connections
    except Exception as e:
        logger.warning(f"LLM check failed: {e}")

    # 2. Tool availability check removed - was causing singleton registry bug
    # Tools are registered properly when WorkflowOrchestrator creates ToolOrchestrator with LLM service
    logger.info("[OK] Tool Service: Will be initialized with LLM service during workflow")

    # 3. Configure Memory Service (optional - works without Redis/Qdrant)
    memory_service = None
    try:
        memory_config = MemoryConfig()

        # Create a simple embedding function for memory
        async def embed_function(text: str):
            """Create embeddings for memory system."""
            # Use OpenAI for embeddings
            import openai
            try:
                response = await openai.AsyncOpenAI().embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return response.data[0].embedding
            except Exception:
                # Fallback to simple hash-based embedding
                import hashlib
                hash_val = hashlib.md5(text.encode()).hexdigest()
                # Convert to 1536-dim vector
                return [float(int(hash_val[i:i+2], 16)) / 255.0
                       for i in range(0, min(len(hash_val), 32), 2)] + [0.0] * 1520

        memory_service = MemoryOrchestrator(
            config=memory_config,
            embedding_function=embed_function,
        )
        logger.info("[OK] Memory Service: Configured")
    except Exception as e:
        logger.info("  → Continuing without memory (using agent fallbacks)")

    # 4. Configure Checkpoint Manager
    checkpoint_manager = None
    try:
        checkpoint_manager = CheckpointManager(memory_config if memory_service else MemoryConfig())
        logger.info("[OK] Checkpoint Manager: Ready")
    except Exception as e:
        logger.info("  → Continuing without checkpointing")

    return memory_service, checkpoint_manager


async def run_real_task():
    """
    Execute a real software development task using the agent swarm.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "AGENT SWARM - REAL TASK EXAMPLE")
    print("=" * 70 + "\n")

    # Step 1: Set up services
    memory_service, checkpoint_manager = await setup_services()

    # Step 2: Define your task
    task = {
        "id": "build_calculator",
        "description": "Create a simple Python calculator with unit tests",
        "requirements": [
            "Implement basic arithmetic operations (add, subtract, multiply, divide)",
            "Add error handling for division by zero",
            "Write comprehensive unit tests with pytest",
            "Include a simple CLI interface",
            "Follow PEP 8 style guidelines",
        ],
        "constraints": [
            "Python 3.9+",
            "Use only standard library (except pytest for testing)",
            "Keep it under 200 lines of code",
        ],
    }

    print(">> Task Definition:")
    print(f"   {task['description']}")
    print(f"\n   Requirements:")
    for req in task['requirements']:
        print(f"   - {req}")
    print()

    # Step 3: Configure workflow
    workflow_config = WorkflowConfig(
        project_id="calculator_project",
        enable_human_approval=False,  # Set to True if you want to review before execution
        max_retries=100,  # Very large max retries for development
        save_artifacts=True,  # Save generated code
    )

    # Step 4: Initialize workflow orchestrator
    # NOTE: Agents create their own LLM/tool services internally based on .env config
    try:
        orchestrator = WorkflowOrchestrator(
            checkpoint_manager=checkpoint_manager,
            memory_service=memory_service,
        )

        logger.info("[OK] Workflow Orchestrator: Ready")
        logger.info("  → Agents will create LLM/tool services internally")
        print("\n>> Starting workflow execution...\n")

        # Step 5: Execute the workflow
        execution = await orchestrator.start_workflow(
            task=task,
            config=workflow_config,
        )

        # Step 6: Monitor and display results
        print("\n" + "=" * 70)
        print("EXECUTION RESULTS")
        print("=" * 70)
        print(f"\n> Workflow ID: {execution.workflow_id}")
        print(f"> Status: {execution.status}")
        print(f"> Duration: {execution.elapsed_time_seconds:.2f}s")
        print(f"> Total Tokens: {execution.total_tokens}")
        print(f"> Total Cost: ${execution.total_cost_usd:.4f}")

        # Show execution metadata if available
        if execution.metadata:
            print(f"\n>> Metadata:")
            for key, value in execution.metadata.items():
                print(f"   {key}: {value}")

        # Show error if any
        if execution.error:
            print(f"\n>> Error: {execution.error}")

        print("\n" + "=" * 70)
        print(">> Workflow completed!")
        print("=" * 70 + "\n")

        # Show where to find the output
        output_dir = Path(f"output/{workflow_config.project_id}")
        if output_dir.exists():
            print(f">> Generated code may be saved to: {output_dir}")
            print(f"   View your files: ls {output_dir}")

    except Exception as e:
        import traceback
        logger.error(f"\n>> Workflow execution failed: {e}")
        logger.error(f"\nTraceback:\n{traceback.format_exc()}")

        print("\n" + "=" * 70)
        print("TROUBLESHOOTING")
        print("=" * 70)
        print("\nCommon issues:")
        print("1. Missing API keys - Check your .env file")
        print("2. No LLM available - Install Ollama or add OpenAI/OpenRouter key")
        print("3. Service timeouts - Check Redis/Qdrant are running (optional)")
        print("\nRun the basic demo to verify setup: python run.py")
        print("=" * 70 + "\n")
        raise


async def main():
    """Main entry point."""
    try:
        await run_real_task()
    except KeyboardInterrupt:
        print("\n\n>> Task interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
