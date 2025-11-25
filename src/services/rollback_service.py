"""Rollback service for state recovery and version management."""

import logging
from typing import Optional, List, Dict, Any

from ..core.checkpoints import EnhancedCheckpointManager
from ..models.state_models import AgentState, StateSnapshot
from ..models.workflow_models import RollbackRequest

logger = logging.getLogger(__name__)


class RollbackManager:
    """
    Rollback manager for state recovery and version control.

    Provides rollback to previous checkpoints, state versioning,
    and rollback history tracking.
    """

    def __init__(self, enhanced_checkpoint: EnhancedCheckpointManager):
        """
        Initialize rollback manager.

        Args:
            enhanced_checkpoint: Enhanced checkpoint manager
        """
        self.enhanced_checkpoint = enhanced_checkpoint
        self._rollback_history: Dict[str, List[Dict[str, Any]]] = {}

    async def rollback_to_checkpoint(
        self,
        request: RollbackRequest,
    ) -> Optional[AgentState]:
        """
        Rollback workflow to specific checkpoint.

        Args:
            request: Rollback request

        Returns:
            Restored state if successful
        """
        logger.info(
            f"Rolling back workflow {request.workflow_id} "
            f"to checkpoint {request.target_checkpoint_id}"
        )

        # Get current state for history
        checkpoints = await self.enhanced_checkpoint.list_checkpoints(
            request.workflow_id
        )

        # Perform rollback
        restored_state = await self.enhanced_checkpoint.rollback_to_checkpoint(
            workflow_id=request.workflow_id,
            checkpoint_id=request.target_checkpoint_id,
        )

        if restored_state:
            # Record rollback in history
            rollback_record = {
                "timestamp": AgentState.__annotations__,  # Would use datetime
                "target_checkpoint": request.target_checkpoint_id,
                "reason": request.reason,
                "requested_by": request.requested_by,
                "checkpoints_before": len(checkpoints),
            }

            if request.workflow_id not in self._rollback_history:
                self._rollback_history[request.workflow_id] = []

            self._rollback_history[request.workflow_id].append(rollback_record)

            logger.info(
                f"Successfully rolled back {request.workflow_id} "
                f"to checkpoint {request.target_checkpoint_id}"
            )

        return restored_state

    async def rollback_to_version(
        self,
        workflow_id: str,
        target_version: int,
        reason: str = "Manual rollback to version",
        requested_by: str = "system",
    ) -> Optional[AgentState]:
        """
        Rollback to specific state version.

        Args:
            workflow_id: Workflow identifier
            target_version: Target state version
            reason: Reason for rollback
            requested_by: Who requested rollback

        Returns:
            Restored state if successful
        """
        logger.info(f"Rolling back workflow {workflow_id} to version {target_version}")

        restored_state = await self.enhanced_checkpoint.rollback_to_version(
            workflow_id=workflow_id,
            target_version=target_version,
        )

        if restored_state:
            # Record rollback
            rollback_record = {
                "target_version": target_version,
                "reason": reason,
                "requested_by": requested_by,
            }

            if workflow_id not in self._rollback_history:
                self._rollback_history[workflow_id] = []

            self._rollback_history[workflow_id].append(rollback_record)

            logger.info(
                f"Successfully rolled back {workflow_id} to version {target_version}"
            )

        return restored_state

    async def list_checkpoints(
        self,
        workflow_id: str,
    ) -> List[Dict[str, Any]]:
        """
        List available checkpoints for rollback.

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of checkpoint information
        """
        return await self.enhanced_checkpoint.list_checkpoints(workflow_id)

    async def get_checkpoint_history(
        self,
        workflow_id: str,
        limit: int = 10,
    ) -> List[StateSnapshot]:
        """
        Get checkpoint history.

        Args:
            workflow_id: Workflow identifier
            limit: Maximum snapshots

        Returns:
            List of state snapshots
        """
        return await self.enhanced_checkpoint.get_checkpoint_history(
            workflow_id=workflow_id,
            limit=limit,
        )

    def get_rollback_history(
        self,
        workflow_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get rollback history for workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of rollback operations
        """
        return self._rollback_history.get(workflow_id, [])

    async def cleanup_old_checkpoints(
        self,
        workflow_id: str,
        keep_last: int = 50,
    ) -> int:
        """
        Clean up old checkpoints.

        Args:
            workflow_id: Workflow identifier
            keep_last: Number to keep

        Returns:
            Number deleted
        """
        return await self.enhanced_checkpoint.cleanup_old_snapshots(
            workflow_id=workflow_id,
            keep_last=keep_last,
        )
