"""Tests for WorkflowOrchestrator service."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from src.services.workflow_service import WorkflowOrchestrator
from src.services.checkpoint_service import CheckpointManager
from src.config.memory_config import MemoryConfig
from src.models.workflow_models import WorkflowConfig


@pytest.fixture
def checkpoint_manager():
    """Create checkpoint manager for testing."""
    config = MemoryConfig()
    return CheckpointManager(config)


def test_workflow_orchestrator_init(checkpoint_manager):
    """Test workflow orchestrator initialization."""
    orchestrator = WorkflowOrchestrator(
        checkpoint_manager=checkpoint_manager,
        memory_service=None,
    )

    assert orchestrator.checkpoint_manager is not None
    assert orchestrator.enhanced_checkpoint is not None
