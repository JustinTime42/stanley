"""Coordinator agent for workflow orchestration."""

import logging
from typing import Optional

from .base import BaseAgent
from ..models.state_models import AgentState, AgentRole, WorkflowStatus
from ..models.agent_models import AgentResponse
from ..services.memory_service import MemoryOrchestrator

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    """
    Coordinator agent responsible for orchestrating the overall workflow.

    Responsibilities:
    - Analyze incoming tasks and determine workflow path
    - Route to appropriate agents based on task type and current state
    - Monitor workflow progress and adjust as needed
    - Handle high-level decision making
    """

    def __init__(self, memory_service: Optional[MemoryOrchestrator] = None):
        """Initialize coordinator agent."""
        super().__init__(role=AgentRole.COORDINATOR, memory_service=memory_service)

    async def execute(self, state: AgentState) -> AgentResponse:
        """
        Execute coordinator logic to route workflow.

        Args:
            state: Current workflow state

        Returns:
            AgentResponse with routing decision
        """
        try:
            self.logger.info(
                f"Coordinator analyzing workflow {state.get('workflow_id')}"
            )

            # Get current status and task
            status = state.get("status", WorkflowStatus.PENDING.value)
            task = state.get("task", {})

            # Retrieve relevant context from memory
            context = await self.retrieve_context(
                query=task.get("description", ""),
                state=state,
                k=3,
            )

            # Determine next agent based on workflow status
            next_agent = self._route_to_agent(state, status)

            # Create coordination message
            coord_message = self.create_message(
                content=f"Routing to {next_agent.value} for {status}",
                message_type="info",
                metadata={"context_memories": len(context.get("memories", []))},
            )

            # Store coordination decision
            await self.store_result(
                content=f"Coordination decision: Route to {next_agent.value}",
                state=state,
                importance=0.6,
                tags=["coordination", "routing"],
            )

            # Prepare state updates
            state_updates = {
                "current_agent": self.role.value,
                "next_agent": next_agent.value,
                "updated_at": coord_message["timestamp"],
            }

            # Check if we're done
            if status == WorkflowStatus.COMPLETE.value:
                state_updates["should_continue"] = False
                next_agent = None  # Workflow ends

            return self._create_success_response(
                result={
                    "decision": "route",
                    "next_agent": next_agent.value if next_agent else "END",
                    "reasoning": f"Current status: {status}",
                    "context_retrieved": len(context.get("memories", [])),
                },
                next_agent=next_agent,
                state_updates=state_updates,
                messages=[coord_message],
            )

        except Exception as e:
            self.logger.error(f"Coordinator execution failed: {e}")
            return self._create_error_response(
                error=f"Coordinator failed: {str(e)}",
            )

    def _route_to_agent(
        self,
        state: AgentState,
        status: str,
    ) -> Optional[AgentRole]:
        """
        Determine next agent based on workflow status.

        Args:
            state: Current state
            status: Current workflow status

        Returns:
            Next agent role or None if workflow is complete
        """
        # Standard workflow routing
        routing_map = {
            WorkflowStatus.PENDING.value: AgentRole.PLANNER,
            WorkflowStatus.PLANNING.value: AgentRole.PLANNER,
            WorkflowStatus.DESIGNING.value: AgentRole.ARCHITECT,
            WorkflowStatus.IMPLEMENTING.value: AgentRole.IMPLEMENTER,
            WorkflowStatus.TESTING.value: AgentRole.TESTER,
            WorkflowStatus.VALIDATING.value: AgentRole.VALIDATOR,
            WorkflowStatus.DEBUGGING.value: AgentRole.DEBUGGER,
            WorkflowStatus.HUMAN_REVIEW.value: None,  # Wait for human approval
            WorkflowStatus.COMPLETE.value: None,  # End workflow
            WorkflowStatus.FAILED.value: AgentRole.DEBUGGER,
        }

        next_agent = routing_map.get(status)

        # Check if human approval is required
        if state.get("requires_human_approval"):
            self.logger.info("Human approval required, pausing workflow")
            return None

        # Check retry logic
        if state.get("retry_count", 0) >= state.get("max_retries", 3):
            self.logger.warning("Max retries exceeded, routing to debugger")
            return AgentRole.DEBUGGER

        return next_agent
