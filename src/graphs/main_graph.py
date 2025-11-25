"""Main workflow graph orchestrating all agents."""

import logging
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver

from ..core.workflow import create_workflow
from ..services.memory_service import MemoryOrchestrator
from ..models.workflow_models import WorkflowConfig

logger = logging.getLogger(__name__)


class MainWorkflowGraph:
    """
    Main workflow graph orchestrating all 7 agents.

    Provides high-level interface to create and configure
    the complete agent workflow graph.
    """

    def __init__(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        memory_service: Optional[MemoryOrchestrator] = None,
        config: Optional[WorkflowConfig] = None,
    ):
        """
        Initialize main workflow graph.

        Args:
            checkpointer: Checkpoint saver for persistence
            memory_service: Memory orchestrator
            config: Optional workflow configuration
        """
        self.checkpointer = checkpointer
        self.memory_service = memory_service
        self.config = config or WorkflowConfig(
            project_id="default",
            enable_human_approval=True,
        )
        self._compiled_graph = None

    def build(self):
        """
        Build and compile the workflow graph.

        Returns:
            Compiled LangGraph workflow
        """
        logger.info("Building main workflow graph")

        self._compiled_graph = create_workflow(
            checkpointer=self.checkpointer,
            memory_service=self.memory_service,
            enable_human_approval=self.config.enable_human_approval,
        )

        logger.info("Main workflow graph built successfully")
        return self._compiled_graph

    def get_graph(self):
        """
        Get compiled graph (builds if not yet built).

        Returns:
            Compiled workflow
        """
        if self._compiled_graph is None:
            return self.build()
        return self._compiled_graph

    async def execute(
        self,
        initial_state: dict,
        thread_id: str,
    ) -> dict:
        """
        Execute workflow with initial state.

        Args:
            initial_state: Initial state dictionary
            thread_id: Thread identifier for checkpointing

        Returns:
            Final state
        """
        graph = self.get_graph()

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        result = await graph.ainvoke(
            initial_state,
            config=config,
        )

        return result

    def visualize(
        self,
        output_file: Optional[str] = None,
    ) -> str:
        """
        Generate visualization of workflow graph.

        Args:
            output_file: Optional file to save diagram

        Returns:
            Mermaid diagram string
        """
        from ..utils.visualization import visualize_workflow

        graph = self.get_graph()
        return visualize_workflow(
            workflow_graph=graph,
            output_file=output_file,
        )
