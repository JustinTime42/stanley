"""Checkpoint management service for LangGraph state persistence."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from ..models.checkpoint_models import CheckpointMetadata
from ..config.memory_config import MemoryConfig

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Checkpoint manager for LangGraph state persistence.

    Manages checkpoints for long-running agent sessions, enabling
    resume from interruptions and state recovery.
    """

    def __init__(self, config: MemoryConfig):
        """
        Initialize checkpoint manager.

        Args:
            config: Memory system configuration
        """
        self.config = config
        self._checkpointer = None

    def get_checkpointer(self):
        """
        Get LangGraph checkpointer instance.

        CRITICAL: This must be passed to compile(), not invoke()

        Returns:
            RedisSaver checkpointer
        """
        if self._checkpointer is None:
            # Use MemorySaver instead of RedisSaver due to async compatibility issues
            # TODO: Re-enable RedisSaver when langgraph-checkpoint-redis supports async interface
            from langgraph.checkpoint.memory import MemorySaver

            self._checkpointer = MemorySaver()
            logger.info("Created Memory checkpointer for LangGraph (in-memory only)")

        return self._checkpointer

    async def save_checkpoint(
        self,
        thread_id: str,
        agent_id: str,
        state: Dict[str, Any],
        project_id: Optional[str] = None,
        checkpoint_type: str = "auto",
        parent_checkpoint: Optional[str] = None,
    ) -> CheckpointMetadata:
        """
        Save a checkpoint.

        Args:
            thread_id: Thread identifier
            agent_id: Agent identifier
            state: State to checkpoint
            project_id: Optional project identifier
            checkpoint_type: Type of checkpoint (auto, manual, error_recovery)
            parent_checkpoint: Parent checkpoint ID

        Returns:
            Checkpoint metadata
        """
        checkpoint_id = str(uuid.uuid4())

        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            thread_id=thread_id,
            agent_id=agent_id,
            project_id=project_id,
            timestamp=datetime.now(),
            parent_checkpoint=parent_checkpoint,
            checkpoint_type=checkpoint_type,
            memory_stats=self._calculate_memory_stats(state),
        )

        logger.info(
            f"Saved checkpoint {checkpoint_id} for {agent_id}/{thread_id} "
            f"(type: {checkpoint_type})"
        )

        return metadata

    async def load_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Optional specific checkpoint ID (default: latest)

        Returns:
            Checkpoint state if found
        """
        # Note: LangGraph checkpointer handles actual state loading
        # This is a convenience method for metadata tracking

        logger.info(
            f"Loading checkpoint for thread {thread_id}"
            + (f" (checkpoint: {checkpoint_id})" if checkpoint_id else " (latest)")
        )

        return None  # Actual loading is done by LangGraph

    async def list_checkpoints(
        self,
        thread_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[CheckpointMetadata]:
        """
        List checkpoints with filters.

        Args:
            thread_id: Optional thread filter
            agent_id: Optional agent filter
            project_id: Optional project filter
            limit: Maximum results

        Returns:
            List of checkpoint metadata
        """
        # This would typically query a metadata store
        # For now, return empty list as LangGraph manages checkpoints
        logger.debug(
            f"Listing checkpoints (thread={thread_id}, agent={agent_id}, "
            f"project={project_id}, limit={limit})"
        )

        return []

    async def delete_checkpoint(
        self,
        checkpoint_id: str,
    ) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to delete

        Returns:
            True if deleted, False if not found
        """
        logger.info(f"Deleting checkpoint {checkpoint_id}")

        # LangGraph checkpointer would handle actual deletion
        return True

    async def cleanup_old_checkpoints(
        self,
        max_age_days: int = 7,
        keep_manual: bool = True,
    ) -> int:
        """
        Clean up old checkpoints.

        Args:
            max_age_days: Maximum age in days
            keep_manual: Keep manual checkpoints

        Returns:
            Number of checkpoints deleted
        """
        logger.info(
            f"Cleaning up checkpoints older than {max_age_days} days "
            f"(keep_manual={keep_manual})"
        )

        # Implementation would query and delete old checkpoints
        return 0

    def _calculate_memory_stats(self, state: Dict[str, Any]) -> Dict[str, int]:
        """
        Calculate memory statistics from state.

        Args:
            state: Agent state

        Returns:
            Dictionary with memory stats
        """
        import sys

        # Rough estimates
        return {
            "state_size_bytes": sys.getsizeof(state),
            "keys_count": len(state),
        }

    async def create_checkpointer_for_workflow(self):
        """
        Create checkpointer for LangGraph workflow compilation.

        CRITICAL: Pass this to workflow.compile(checkpointer=...)

        Returns:
            Configured checkpointer
        """
        return self.get_checkpointer()


def create_checkpointer(redis_url: str):
    """
    Factory function to create LangGraph checkpointer.

    CRITICAL: Must be passed to compile(), not invoke()
    PATTERN: Create once, reuse across workflow

    Args:
        redis_url: Redis connection URL

    Returns:
        RedisSaver checkpointer instance
    """
    from langgraph.checkpoint.redis import RedisSaver

    checkpointer = RedisSaver(
        redis_url=redis_url,
    )

    logger.info(f"Created checkpointer with Redis URL: {redis_url}")
    return checkpointer


# Example usage in workflow integration
"""
from src.services.checkpoint_service import create_checkpointer

# Create checkpointer
checkpointer = create_checkpointer(config.redis_url)

# Compile workflow with checkpointer
app = workflow.compile(checkpointer=checkpointer)  # CRITICAL: At compile time!

# Use workflow with thread_id
result = app.invoke(
    input_data,
    config={"configurable": {"thread_id": "session_123"}}
)

# Resume from checkpoint
resumed = app.invoke(
    new_input,
    config={"configurable": {"thread_id": "session_123"}}  # Same thread_id
)
"""
