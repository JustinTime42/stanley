"""Execution tracking models for the tool abstraction layer."""

import uuid
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

from .tool_models import ToolRequest


class ExecutionMetrics(BaseModel):
    """Metrics for tool execution."""

    tool_name: str
    total_executions: int = Field(default=0)
    successful_executions: int = Field(default=0)
    failed_executions: int = Field(default=0)
    total_retries: int = Field(default=0)
    average_execution_time_ms: float = Field(default=0.0)
    total_cpu_seconds: float = Field(default=0.0)
    total_memory_mb: float = Field(default=0.0)
    last_execution: Optional[datetime] = Field(default=None)
    error_types: Dict[str, int] = Field(default_factory=dict)


class ParallelExecutionPlan(BaseModel):
    """Plan for parallel tool execution."""

    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_groups: List[List[ToolRequest]] = Field(
        description="Groups of tools that can run in parallel"
    )
    dependencies: Dict[str, List[str]] = Field(
        default_factory=dict, description="Tool dependencies (tool -> [depends_on])"
    )
    estimated_time_ms: int = Field(description="Estimated total execution time")
    max_parallelism: int = Field(default=5, description="Max concurrent executions")


class ResourceUsage(BaseModel):
    """Resource usage tracking."""

    cpu_percent: float = Field(description="CPU usage percentage")
    memory_mb: float = Field(description="Memory usage in MB")
    io_read_bytes: int = Field(default=0)
    io_write_bytes: int = Field(default=0)
    network_sent_bytes: int = Field(default=0)
    network_recv_bytes: int = Field(default=0)
    open_files: int = Field(default=0)
    thread_count: int = Field(default=1)
