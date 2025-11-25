"""Checkpoint data models for state persistence."""

from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime


class CheckpointMetadata(BaseModel):
    """Checkpoint metadata model."""

    checkpoint_id: str
    thread_id: str
    agent_id: str
    project_id: Optional[str]
    timestamp: datetime
    parent_checkpoint: Optional[str] = None
    checkpoint_type: str = "auto"  # auto, manual, error_recovery
    memory_stats: Dict[str, int] = Field(default_factory=dict)
