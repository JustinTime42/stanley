"""Custom workflow execution script."""

import asyncio
import logging
from main import AgentSwarmSystem

logging.basicConfig(level=logging.INFO)


async def run_custom_workflow():
    """Run a custom workflow."""

    # Initialize system
    system = AgentSwarmSystem()

    # Define your task
    task = {
        "id": "my_task_1",
        "description": "Create a REST API with user authentication",
        "requirements": [
            "Use FastAPI framework",
            "Implement JWT authentication",
            "Include user registration and login endpoints",
            "Add password hashing with bcrypt",
            "Include input validation with Pydantic",
        ],
        "constraints": [
            "Must follow REST best practices",
            "Include comprehensive error handling",
            "Add API documentation with OpenAPI",
        ]
    }

    try:
        # Start workflow (with approval disabled for demo)
        print("Starting workflow...")
        execution = await system.start_workflow(
            task=task,
            project_id="my_project",
            enable_human_approval=False,  # Set True to enable approval gates
            max_retries=3,
        )

        print(f"\nWorkflow completed!")
        print(f"Status: {execution.status}")
        print(f"Workflow ID: {execution.workflow_id}")

        # Visualize the workflow
        await system.visualize_workflow(
            workflow_id=execution.workflow_id,
            output_file="my_workflow.mermaid"
        )
        print(f"Visualization saved to: my_workflow.mermaid")

    except Exception as e:
        print(f"Workflow failed: {e}")

    finally:
        await system.cleanup()


if __name__ == "__main__":
    asyncio.run(run_custom_workflow())
