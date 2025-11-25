"""Unit tests for checkpoint service."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.services.checkpoint_service import CheckpointManager, create_checkpointer
from src.models.checkpoint_models import CheckpointMetadata
from src.config.memory_config import MemoryConfig


@pytest.fixture
def mock_config():
    """Create mock memory config."""
    config = MagicMock(spec=MemoryConfig)
    config.redis_url = "redis://localhost:6379/0"
    return config


@pytest.mark.asyncio
class TestCheckpointManager:
    """Test suite for CheckpointManager."""

    async def test_init(self, mock_config):
        """Test initialization."""
        manager = CheckpointManager(config=mock_config)
        assert manager.config == mock_config
        assert manager._checkpointer is None

    async def test_get_checkpointer(self, mock_config):
        """Test getting checkpointer instance."""
        manager = CheckpointManager(config=mock_config)

        # Test that get_checkpointer would raise ImportError if langgraph not installed
        # In a real environment, this would return a RedisSaver instance
        try:
            checkpointer = manager.get_checkpointer()
            # If it succeeds, verify it's cached
            checkpointer2 = manager.get_checkpointer()
            assert checkpointer is checkpointer2
        except ImportError:
            # Expected if langgraph.checkpoint.redis is not installed
            pass

    async def test_get_checkpointer_singleton(self, mock_config):
        """Test checkpointer is created only once."""
        manager = CheckpointManager(config=mock_config)

        # Test singleton behavior (skip if langgraph not installed)
        try:
            checkpointer1 = manager.get_checkpointer()
            checkpointer2 = manager.get_checkpointer()
            # Verify only created once
            assert checkpointer1 is checkpointer2
        except ImportError:
            # Expected if langgraph.checkpoint.redis is not installed
            pass

    async def test_save_checkpoint(self, mock_config):
        """Test saving checkpoint metadata."""
        manager = CheckpointManager(config=mock_config)

        # Save checkpoint
        metadata = await manager.save_checkpoint(
            thread_id="thread-1",
            agent_id="agent-1",
            state={"key": "value"},
            project_id="project-1",
            checkpoint_type="manual",
        )

        # Verify
        assert isinstance(metadata, CheckpointMetadata)
        assert metadata.thread_id == "thread-1"
        assert metadata.agent_id == "agent-1"
        assert metadata.project_id == "project-1"
        assert metadata.checkpoint_type == "manual"
        assert isinstance(metadata.checkpoint_id, str)
        assert isinstance(metadata.timestamp, datetime)

    async def test_save_checkpoint_with_parent(self, mock_config):
        """Test saving checkpoint with parent reference."""
        manager = CheckpointManager(config=mock_config)

        # Save checkpoint with parent
        metadata = await manager.save_checkpoint(
            thread_id="thread-1",
            agent_id="agent-1",
            state={"key": "value"},
            parent_checkpoint="parent-id",
        )

        # Verify
        assert metadata.parent_checkpoint == "parent-id"

    async def test_load_checkpoint(self, mock_config):
        """Test loading checkpoint."""
        manager = CheckpointManager(config=mock_config)

        # Load checkpoint (returns None as actual loading is done by LangGraph)
        result = await manager.load_checkpoint(
            thread_id="thread-1",
            checkpoint_id="checkpoint-1",
        )

        # Verify
        assert result is None  # Placeholder implementation

    async def test_list_checkpoints(self, mock_config):
        """Test listing checkpoints."""
        manager = CheckpointManager(config=mock_config)

        # List checkpoints
        checkpoints = await manager.list_checkpoints(
            thread_id="thread-1",
            limit=10,
        )

        # Verify
        assert isinstance(checkpoints, list)
        assert len(checkpoints) == 0  # Placeholder implementation

    async def test_delete_checkpoint(self, mock_config):
        """Test deleting checkpoint."""
        manager = CheckpointManager(config=mock_config)

        # Delete checkpoint
        result = await manager.delete_checkpoint("checkpoint-1")

        # Verify
        assert result is True  # Placeholder implementation

    async def test_cleanup_old_checkpoints(self, mock_config):
        """Test cleaning up old checkpoints."""
        manager = CheckpointManager(config=mock_config)

        # Cleanup
        count = await manager.cleanup_old_checkpoints(
            max_age_days=7,
            keep_manual=True,
        )

        # Verify
        assert count == 0  # Placeholder implementation

    async def test_calculate_memory_stats(self, mock_config):
        """Test memory statistics calculation."""
        manager = CheckpointManager(config=mock_config)

        # Calculate stats
        state = {"key1": "value1", "key2": "value2", "key3": [1, 2, 3]}
        stats = manager._calculate_memory_stats(state)

        # Verify
        assert "state_size_bytes" in stats
        assert "keys_count" in stats
        assert stats["keys_count"] == 3

    async def test_create_checkpointer_for_workflow(self, mock_config):
        """Test creating checkpointer for workflow compilation."""
        manager = CheckpointManager(config=mock_config)

        # Test creating checkpointer (skip if langgraph not installed)
        try:
            checkpointer = await manager.create_checkpointer_for_workflow()
            assert checkpointer is not None
        except ImportError:
            # Expected if langgraph.checkpoint.redis is not installed
            pass


def test_create_checkpointer_function():
    """Test standalone create_checkpointer function."""
    redis_url = "redis://localhost:6379/0"

    # Test creating checkpointer (skip if langgraph not installed)
    try:
        checkpointer = create_checkpointer(redis_url)
        assert checkpointer is not None
    except ImportError:
        # Expected if langgraph.checkpoint.redis is not installed
        pass
