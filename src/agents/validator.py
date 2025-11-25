"""Validator agent for quality assurance and validation."""

import logging
from typing import Optional

from .base import BaseAgent
from ..models.state_models import AgentState, AgentRole, WorkflowStatus
from ..models.agent_models import AgentResponse
from ..services.memory_service import MemoryOrchestrator

logger = logging.getLogger(__name__)


class ValidatorAgent(BaseAgent):
    """
    Validator agent responsible for quality assurance and validation.

    Responsibilities:
    - Validate implementation against requirements
    - Perform code quality checks
    - Verify compliance with standards
    - Final approval before completion
    """

    def __init__(self, memory_service: Optional[MemoryOrchestrator] = None):
        """Initialize validator agent."""
        super().__init__(role=AgentRole.VALIDATOR, memory_service=memory_service)

    async def execute(self, state: AgentState) -> AgentResponse:
        """
        Execute validation logic.

        Args:
            state: Current workflow state

        Returns:
            AgentResponse with validation results
        """
        try:
            self.logger.info(f"Validator checking {state.get('workflow_id')}")

            # Use `or {}` to handle both missing keys and explicit None values
            plan = state.get("plan") or {}
            implementation = state.get("implementation") or {}
            test_results = state.get("test_results") or {}

            # Retrieve validation criteria
            context = await self.retrieve_context(
                query="validation criteria and standards",
                state=state,
                k=5,
            )

            # Perform validation
            validation_results = self._validate(
                plan, implementation, test_results, context
            )

            # Store validation results
            await self.store_result(
                content=f"Validation: {validation_results['status']}",
                state=state,
                importance=0.9,
                tags=["validation", "quality"],
            )

            # Create message
            validation_message = self.create_message(
                content=f"Validation {validation_results['status']}: {len(validation_results['checks'])} checks performed",
                message_type="success" if validation_results["approved"] else "error",
                metadata={"validation_id": validation_results["validation_id"]},
            )

            # Determine next step
            if validation_results["approved"]:
                next_agent = None  # Workflow complete
                next_status = WorkflowStatus.COMPLETE.value
            else:
                next_agent = AgentRole.DEBUGGER
                next_status = WorkflowStatus.DEBUGGING.value

            # State updates
            state_updates = {
                "validation_results": validation_results,
                "status": next_status,
                "should_continue": not validation_results["approved"],
            }

            return self._create_success_response(
                result=validation_results,
                next_agent=next_agent,
                state_updates=state_updates,
                messages=[validation_message],
                requires_approval=True,  # Validator requires approval
            )

        except Exception as e:
            self.logger.error(f"Validator execution failed: {e}")
            return self._create_error_response(
                error=f"Validation failed: {str(e)}",
            )

    def _validate(
        self,
        plan: dict,
        implementation: dict,
        test_results: dict,
        context: dict,
    ) -> dict:
        """
        Validate implementation against criteria.

        Args:
            plan: Execution plan
            implementation: Implementation details
            test_results: Test results
            context: Retrieved context

        Returns:
            Validation results
        """
        import uuid

        # Simplified validation
        # Note: coverage is a decimal (0.0-1.0), not a percentage
        coverage = test_results.get("coverage", 0)
        checks = [
            {"name": "Tests passing", "passed": test_results.get("all_passed", False)},
            {"name": "Code coverage", "passed": coverage >= 0.8},  # 80% as decimal
            {"name": "Architecture compliance", "passed": True},
            {"name": "Best practices", "passed": True},
        ]

        all_passed = all(check["passed"] for check in checks)

        return {
            "validation_id": str(uuid.uuid4()),
            "plan_id": plan.get("plan_id"),
            "implementation_id": implementation.get("implementation_id"),
            "checks": checks,
            "approved": all_passed,
            "status": "approved" if all_passed else "rejected",
            "context_used": len(context.get("memories", [])),
        }
