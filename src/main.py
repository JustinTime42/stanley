"""Main application entry point for Agent Swarm system."""

import logging
import asyncio
from typing import Dict, Any, Optional

# Direct module imports (avoid __init__.py with relative imports)
from .config import memory_config as mem_config
from .services import checkpoint_service as checkpoint_svc
from .services import memory_service as memory_svc
from .services import workflow_service as workflow_svc
from .services import rollback_service as rollback_svc
from .services import human_approval_service as approval_svc
from .core import checkpoints as checkpoints_mod
from .models import workflow_models as workflow_models
from .graphs import main_graph as main_graph_mod

# Extract classes
MemoryConfig = mem_config.MemoryConfig
CheckpointManager = checkpoint_svc.CheckpointManager
MemoryOrchestrator = memory_svc.MemoryOrchestrator
WorkflowOrchestrator = workflow_svc.WorkflowOrchestrator
RollbackManager = rollback_svc.RollbackManager
HumanApprovalService = approval_svc.HumanApprovalService
EnhancedCheckpointManager = checkpoints_mod.EnhancedCheckpointManager
WorkflowConfig = workflow_models.WorkflowConfig
MainWorkflowGraph = main_graph_mod.MainWorkflowGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class AgentSwarmSystem:
    """
    Main Agent Swarm system integrating all components.

    Provides high-level interface to:
    - Start and manage workflows
    - Handle human approvals
    - Perform rollbacks
    - Visualize workflows
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embedding_function=None,
    ):
        """
        Initialize Agent Swarm system.

        Args:
            config: Memory configuration
            embedding_function: Embedding function for memory
        """
        self.config = config or MemoryConfig()
        self.embedding_function = embedding_function

        # Initialize core services
        self.checkpoint_manager = CheckpointManager(self.config)
        self.memory_service = MemoryOrchestrator(
            config=self.config,
            embedding_function=embedding_function,
        )

        # Initialize enhanced checkpoint
        self.enhanced_checkpoint = EnhancedCheckpointManager(self.checkpoint_manager)

        # Initialize workflow orchestrator
        self.workflow_orchestrator = WorkflowOrchestrator(
            checkpoint_manager=self.checkpoint_manager,
            memory_service=self.memory_service,
        )

        # Initialize rollback manager
        self.rollback_manager = RollbackManager(
            enhanced_checkpoint=self.enhanced_checkpoint
        )

        # Initialize human approval service
        self.approval_service = HumanApprovalService()

        logger.info("Agent Swarm system initialized")

    async def start_workflow(
        self,
        task: Dict[str, Any],
        project_id: str,
        enable_human_approval: bool = True,
        max_retries: int = 3,
    ):
        """
        Start a new workflow.

        Args:
            task: Task specification
            project_id: Project identifier
            enable_human_approval: Enable human approval gates
            max_retries: Maximum retry attempts

        Returns:
            WorkflowExecution
        """
        workflow_config = WorkflowConfig(
            project_id=project_id,
            enable_human_approval=enable_human_approval,
            max_retries=max_retries,
        )

        execution = await self.workflow_orchestrator.start_workflow(
            task=task,
            config=workflow_config,
        )

        logger.info(f"Started workflow {workflow_config.workflow_id}")
        return execution

    async def visualize_workflow(
        self,
        workflow_id: str,
        output_file: Optional[str] = None,
    ) -> str:
        """
        Visualize workflow graph.

        Args:
            workflow_id: Workflow identifier
            output_file: Optional output file

        Returns:
            Mermaid diagram
        """
        # Create graph for visualization
        graph = MainWorkflowGraph(
            checkpointer=self.checkpoint_manager.get_checkpointer(),
            memory_service=self.memory_service,
        )

        return graph.visualize(output_file=output_file)

    async def cleanup(self):
        """Cleanup resources."""
        await self.memory_service.cleanup()
        logger.info("Agent Swarm system cleaned up")


async def main():
    """Main entry point for CLI usage."""
    logger.info("Starting Agent Swarm system")

    # Initialize system
    system = AgentSwarmSystem()

    # Example: Start a sample workflow
    task = {
        "id": "sample_task_1",
        "description": "Create a simple calculator application",
        "requirements": [
            "Support basic arithmetic operations",
            "Include unit tests",
            "Follow Python best practices",
        ],
    }

    try:
        execution = await system.start_workflow(
            task=task,
            project_id="demo_project",
            enable_human_approval=False,  # Disable for demo
        )

        logger.info(f"Workflow completed with status: {execution.status}")

        # Visualize workflow
        await system.visualize_workflow(
            workflow_id=execution.workflow_id,
            output_file="workflow.mermaid",
        )
        logger.info("Workflow visualization saved to workflow.mermaid")

    except Exception as e:
        import traceback
        logger.error(f"Workflow failed: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

    finally:
        await system.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
