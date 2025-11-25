"""Workflow orchestration service for managing LangGraph workflows."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..core.workflow import create_workflow
from ..core.state import create_initial_state
from ..core.checkpoints import EnhancedCheckpointManager
from ..models.workflow_models import WorkflowConfig, WorkflowExecution
from ..models.state_models import WorkflowStatus
from ..services.memory_service import MemoryOrchestrator
from ..services.checkpoint_service import CheckpointManager

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    High-level workflow orchestration service.

    Manages workflow execution, state persistence, and integration
    with memory and checkpoint systems.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        memory_service: Optional[MemoryOrchestrator] = None,
        llm_service: Optional["LLMOrchestrator"] = None,
        tool_service: Optional["ToolOrchestrator"] = None,
    ):
        """
        Initialize workflow orchestrator.

        Args:
            checkpoint_manager: Checkpoint manager
            memory_service: Optional memory orchestrator
            llm_service: Optional LLM orchestrator
            tool_service: Optional tool orchestrator
        """
        self.checkpoint_manager = checkpoint_manager
        self.memory_service = memory_service
        self.llm_service = llm_service
        self.tool_service = tool_service
        self.enhanced_checkpoint = EnhancedCheckpointManager(checkpoint_manager)
        self._active_workflows: Dict[str, WorkflowExecution] = {}

        # Create LLM and tool services if not provided
        if self.llm_service is None:
            try:
                from ..config.llm_config import LLMConfig
                from ..services.llm_service import LLMOrchestrator
                self.llm_service = LLMOrchestrator(config=LLMConfig())
                logger.info("Created LLM service with default configuration")
            except Exception as e:
                logger.warning(f"Could not create LLM service: {e}")

        if self.tool_service is None:
            try:
                from ..services.tool_service import ToolOrchestrator
                self.tool_service = ToolOrchestrator(llm_service=self.llm_service)
                logger.info("Created tool service with default configuration")
            except Exception as e:
                logger.warning(f"Could not create tool service: {e}")

    async def start_workflow(
        self,
        task: Dict[str, Any],
        config: WorkflowConfig,
    ) -> WorkflowExecution:
        """
        Start a new workflow execution.

        Args:
            task: Task specification
            config: Workflow configuration

        Returns:
            WorkflowExecution tracking object
        """
        logger.info(f"Starting workflow {config.workflow_id}")

        # Create initial state
        initial_state = create_initial_state(
            task=task,
            project_id=config.project_id,
            workflow_id=config.workflow_id,
            config={
                "max_retries": config.max_retries,
            },
        )

        # Create workflow execution tracking
        execution = WorkflowExecution(
            workflow_id=config.workflow_id,
            config=config,
            status=WorkflowStatus.PENDING,
            started_at=datetime.now(),
        )

        self._active_workflows[config.workflow_id] = execution

        # Create and compile workflow
        checkpointer = self.checkpoint_manager.get_checkpointer()
        workflow = create_workflow(
            checkpointer=checkpointer,
            memory_service=self.memory_service,
            llm_service=self.llm_service,
            tool_service=self.tool_service,
            enable_human_approval=config.enable_human_approval,
        )

        # Save initial checkpoint
        await self.enhanced_checkpoint.save_versioned_checkpoint(
            state=initial_state,
            checkpoint_type="initial",
        )

        try:
            # Execute workflow
            thread_config = {
                "configurable": {
                    "thread_id": config.workflow_id,
                },
                "recursion_limit": 500,  # Support up to 100 retries (each retry ~4-5 steps)
            }

            result = await workflow.ainvoke(
                initial_state,
                config=thread_config,
            )

            # Update execution
            execution.status = WorkflowStatus(result.get("status", "complete"))
            execution.completed_at = datetime.now()
            execution.elapsed_time_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds()

            logger.info(
                f"Workflow {config.workflow_id} completed in "
                f"{execution.elapsed_time_seconds:.2f}s"
            )

            return execution

        except Exception as e:
            import traceback
            logger.error(f"Workflow {config.workflow_id} failed: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()
            raise

    async def resume_workflow(
        self,
        workflow_id: str,
        new_input: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecution:
        """
        Resume a paused workflow.

        Args:
            workflow_id: Workflow identifier
            new_input: Optional new input (e.g., human feedback)

        Returns:
            Updated WorkflowExecution
        """
        logger.info(f"Resuming workflow {workflow_id}")

        execution = self._active_workflows.get(workflow_id)
        if not execution:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Create workflow with checkpointer
        checkpointer = self.checkpoint_manager.get_checkpointer()
        workflow = create_workflow(
            checkpointer=checkpointer,
            memory_service=self.memory_service,
            enable_human_approval=execution.config.enable_human_approval,
        )

        thread_config = {
            "configurable": {
                "thread_id": workflow_id,
            },
            "recursion_limit": 500,  # Support up to 100 retries (each retry ~4-5 steps)
        }

        # Resume with new input or None to continue
        input_data = new_input or None

        try:
            result = await workflow.ainvoke(
                input_data,
                config=thread_config,
            )

            execution.status = WorkflowStatus(result.get("status", "complete"))
            if execution.status in [WorkflowStatus.COMPLETE, WorkflowStatus.FAILED]:
                execution.completed_at = datetime.now()
                execution.elapsed_time_seconds = (
                    execution.completed_at - execution.started_at
                ).total_seconds()

            logger.info(f"Workflow {workflow_id} resumed successfully")
            return execution

        except Exception as e:
            logger.error(f"Failed to resume workflow {workflow_id}: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            raise

    async def get_workflow_status(
        self,
        workflow_id: str,
    ) -> Optional[WorkflowExecution]:
        """
        Get status of workflow execution.

        Args:
            workflow_id: Workflow identifier

        Returns:
            WorkflowExecution if found
        """
        return self._active_workflows.get(workflow_id)

    async def pause_workflow(
        self,
        workflow_id: str,
    ) -> bool:
        """
        Pause workflow execution.

        Args:
            workflow_id: Workflow identifier

        Returns:
            True if paused successfully
        """
        execution = self._active_workflows.get(workflow_id)
        if execution:
            execution.status = WorkflowStatus.HUMAN_REVIEW
            logger.info(f"Paused workflow {workflow_id}")
            return True
        return False

    async def cancel_workflow(
        self,
        workflow_id: str,
    ) -> bool:
        """
        Cancel workflow execution.

        Args:
            workflow_id: Workflow identifier

        Returns:
            True if cancelled successfully
        """
        execution = self._active_workflows.get(workflow_id)
        if execution:
            execution.status = WorkflowStatus.FAILED
            execution.error = "Cancelled by user"
            execution.completed_at = datetime.now()
            logger.info(f"Cancelled workflow {workflow_id}")
            return True
        return False

    def list_active_workflows(self) -> list[WorkflowExecution]:
        """
        List all active workflows.

        Returns:
            List of active WorkflowExecution objects
        """
        return list(self._active_workflows.values())
