"""Human-in-the-loop approval service for workflow governance."""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import asyncio

from ..models.workflow_models import HumanApprovalRequest

logger = logging.getLogger(__name__)


class HumanApprovalService:
    """
    Human-in-the-loop approval service.

    Manages approval requests, feedback collection, and
    workflow resumption after human intervention.
    """

    def __init__(self):
        """Initialize human approval service."""
        self._pending_approvals: Dict[str, HumanApprovalRequest] = {}
        self._approval_history: Dict[str, List[HumanApprovalRequest]] = {}

    async def request_approval(
        self,
        workflow_id: str,
        agent_role: str,
        approval_point: str,
        context: Dict,
        result_preview: Dict,
    ) -> HumanApprovalRequest:
        """
        Request human approval at workflow checkpoint.

        Args:
            workflow_id: Workflow identifier
            agent_role: Agent requesting approval
            approval_point: Description of approval point
            context: Context for decision
            result_preview: Preview of results

        Returns:
            Approval request object
        """
        from ..models.state_models import AgentRole

        request = HumanApprovalRequest(
            workflow_id=workflow_id,
            agent_role=AgentRole(agent_role),
            approval_point=approval_point,
            context=context,
            result_preview=result_preview,
        )

        self._pending_approvals[request.request_id] = request

        logger.info(
            f"Approval requested for workflow {workflow_id} "
            f"at {approval_point} by {agent_role}"
        )

        return request

    async def submit_approval(
        self,
        request_id: str,
        approved: bool,
        feedback: Optional[str] = None,
        approved_by: str = "user",
    ) -> bool:
        """
        Submit approval decision.

        Args:
            request_id: Approval request ID
            approved: Whether approved
            feedback: Optional feedback
            approved_by: Who approved

        Returns:
            True if submission successful
        """
        request = self._pending_approvals.get(request_id)
        if not request:
            logger.warning(f"Approval request {request_id} not found")
            return False

        # Update request
        request.status = "approved" if approved else "rejected"
        request.feedback = feedback
        request.approved_by = approved_by
        request.approved_at = datetime.now()

        # Move to history
        workflow_id = request.workflow_id
        if workflow_id not in self._approval_history:
            self._approval_history[workflow_id] = []

        self._approval_history[workflow_id].append(request)

        # Remove from pending
        del self._pending_approvals[request_id]

        logger.info(f"Approval {request.status} for {request_id} by {approved_by}")

        return True

    async def wait_for_approval(
        self,
        request_id: str,
        timeout_seconds: Optional[int] = None,
    ) -> Optional[HumanApprovalRequest]:
        """
        Wait for approval decision.

        Args:
            request_id: Approval request ID
            timeout_seconds: Optional timeout

        Returns:
            Completed approval request or None if timeout
        """
        start_time = datetime.now()

        while True:
            # Check if moved to history (approved/rejected)
            for workflow_id, history in self._approval_history.items():
                for req in history:
                    if req.request_id == request_id:
                        return req

            # Check timeout
            if timeout_seconds:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= timeout_seconds:
                    # Timeout - mark as timeout
                    request = self._pending_approvals.get(request_id)
                    if request:
                        request.status = "timeout"
                        del self._pending_approvals[request_id]
                    logger.warning(f"Approval request {request_id} timed out")
                    return None

            # Wait before checking again
            await asyncio.sleep(0.5)

    def get_pending_approvals(
        self,
        workflow_id: Optional[str] = None,
    ) -> List[HumanApprovalRequest]:
        """
        Get pending approval requests.

        Args:
            workflow_id: Optional workflow filter

        Returns:
            List of pending requests
        """
        requests = list(self._pending_approvals.values())

        if workflow_id:
            requests = [r for r in requests if r.workflow_id == workflow_id]

        return requests

    def get_approval_history(
        self,
        workflow_id: str,
    ) -> List[HumanApprovalRequest]:
        """
        Get approval history for workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of approval requests
        """
        return self._approval_history.get(workflow_id, [])

    async def cancel_approval(
        self,
        request_id: str,
    ) -> bool:
        """
        Cancel pending approval request.

        Args:
            request_id: Approval request ID

        Returns:
            True if cancelled
        """
        if request_id in self._pending_approvals:
            request = self._pending_approvals[request_id]
            request.status = "cancelled"
            del self._pending_approvals[request_id]
            logger.info(f"Cancelled approval request {request_id}")
            return True

        return False
