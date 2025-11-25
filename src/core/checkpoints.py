"""Enhanced checkpoint management for rollback and state history."""

import logging
from typing import Dict, Any, List, Optional
import uuid

from ..models.state_models import AgentState, StateSnapshot
from ..services.checkpoint_service import CheckpointManager

logger = logging.getLogger(__name__)


class EnhancedCheckpointManager:
    """
    Enhanced checkpoint manager with versioning and rollback support.

    Extends the basic CheckpointManager with:
    - State version tracking
    - Checkpoint history management
    - Rollback to specific versions
    - State snapshot storage
    """

    def __init__(self, checkpoint_manager: CheckpointManager):
        """
        Initialize enhanced checkpoint manager.

        Args:
            checkpoint_manager: Base checkpoint manager
        """
        self.checkpoint_manager = checkpoint_manager
        self._snapshot_store: Dict[str, List[StateSnapshot]] = {}
        self._max_snapshots = 100

    async def save_versioned_checkpoint(
        self,
        state: AgentState,
        checkpoint_type: str = "auto",
    ) -> str:
        """
        Save checkpoint with version tracking.

        Args:
            state: Current state
            checkpoint_type: Type of checkpoint

        Returns:
            Checkpoint ID
        """
        workflow_id = state.get("workflow_id", "")
        state_version = state.get("state_version", 1)
        parent_checkpoint = state.get("parent_checkpoint_id")

        # Save checkpoint via base manager
        metadata = await self.checkpoint_manager.save_checkpoint(
            thread_id=workflow_id,
            agent_id=state.get("current_agent", "unknown"),
            state=dict(state),
            project_id=state.get("project_id"),
            checkpoint_type=checkpoint_type,
            parent_checkpoint=parent_checkpoint,
        )

        checkpoint_id = metadata.checkpoint_id

        # Create state snapshot
        snapshot = StateSnapshot(
            snapshot_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            state_version=state_version,
            swarm_checkpoint_id=checkpoint_id,
            state_data=dict(state),
            created_by=state.get("current_agent", "system"),
            metadata={
                "checkpoint_type": checkpoint_type,
                "status": state.get("status"),
            },
        )

        # Store snapshot
        if workflow_id not in self._snapshot_store:
            self._snapshot_store[workflow_id] = []

        self._snapshot_store[workflow_id].append(snapshot)

        # Limit snapshot history
        if len(self._snapshot_store[workflow_id]) > self._max_snapshots:
            self._snapshot_store[workflow_id] = self._snapshot_store[workflow_id][
                -self._max_snapshots :
            ]

        logger.info(
            f"Saved versioned checkpoint {checkpoint_id} "
            f"(version {state_version}) for {workflow_id}"
        )

        return checkpoint_id

    async def get_checkpoint_history(
        self,
        workflow_id: str,
        limit: int = 10,
    ) -> List[StateSnapshot]:
        """
        Get checkpoint history for workflow.

        Args:
            workflow_id: Workflow identifier
            limit: Maximum snapshots to return

        Returns:
            List of state snapshots
        """
        snapshots = self._snapshot_store.get(workflow_id, [])
        return snapshots[-limit:]

    async def rollback_to_checkpoint(
        self,
        workflow_id: str,
        checkpoint_id: str,
    ) -> Optional[AgentState]:
        """
        Rollback to specific checkpoint.

        Args:
            workflow_id: Workflow identifier
            checkpoint_id: Target checkpoint ID

        Returns:
            Restored state if found
        """
        snapshots = self._snapshot_store.get(workflow_id, [])

        # Find target snapshot
        target_snapshot = None
        for snapshot in snapshots:
            if snapshot.swarm_checkpoint_id == checkpoint_id:
                target_snapshot = snapshot
                break

        if not target_snapshot:
            logger.warning(f"Checkpoint {checkpoint_id} not found for {workflow_id}")
            return None

        # Restore state from snapshot
        restored_state: AgentState = target_snapshot.state_data  # type: ignore

        logger.info(
            f"Rolled back to checkpoint {checkpoint_id} "
            f"(version {target_snapshot.state_version})"
        )

        return restored_state

    async def rollback_to_version(
        self,
        workflow_id: str,
        target_version: int,
    ) -> Optional[AgentState]:
        """
        Rollback to specific state version.

        Args:
            workflow_id: Workflow identifier
            target_version: Target state version

        Returns:
            Restored state if found
        """
        snapshots = self._snapshot_store.get(workflow_id, [])

        # Find snapshot with target version
        target_snapshot = None
        for snapshot in snapshots:
            if snapshot.state_version == target_version:
                target_snapshot = snapshot
                break

        if not target_snapshot:
            logger.warning(f"Version {target_version} not found for {workflow_id}")
            return None

        restored_state: AgentState = target_snapshot.state_data  # type: ignore

        logger.info(f"Rolled back to version {target_version}")
        return restored_state

    async def list_checkpoints(
        self,
        workflow_id: str,
    ) -> List[Dict[str, Any]]:
        """
        List all checkpoints for workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of checkpoint information
        """
        snapshots = self._snapshot_store.get(workflow_id, [])

        return [
            {
                "checkpoint_id": s.checkpoint_id,
                "snapshot_id": s.snapshot_id,
                "state_version": s.state_version,
                "timestamp": s.timestamp.isoformat(),
                "created_by": s.created_by,
                "status": s.metadata.get("status"),
            }
            for s in snapshots
        ]

    def get_snapshot_count(self, workflow_id: str) -> int:
        """
        Get number of snapshots for workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Snapshot count
        """
        return len(self._snapshot_store.get(workflow_id, []))

    async def cleanup_old_snapshots(
        self,
        workflow_id: str,
        keep_last: int = 50,
    ) -> int:
        """
        Clean up old snapshots, keeping only recent ones.

        Args:
            workflow_id: Workflow identifier
            keep_last: Number of snapshots to keep

        Returns:
            Number of snapshots deleted
        """
        if workflow_id not in self._snapshot_store:
            return 0

        snapshots = self._snapshot_store[workflow_id]
        original_count = len(snapshots)

        if original_count > keep_last:
            self._snapshot_store[workflow_id] = snapshots[-keep_last:]
            deleted = original_count - keep_last
            logger.info(f"Cleaned up {deleted} old snapshots for {workflow_id}")
            return deleted

        return 0
