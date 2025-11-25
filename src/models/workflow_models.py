"""Workflow configuration and execution models."""

import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from .state_models import AgentRole, WorkflowStatus


class WorkflowConfig(BaseModel):
    """Workflow configuration model."""

    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str = Field(description="Project identifier")
    enable_human_approval: bool = Field(default=True)
    approval_points: List[AgentRole] = Field(
        default_factory=lambda: [
            AgentRole.PLANNER,
            AgentRole.ARCHITECT,
            AgentRole.VALIDATOR,
        ],
        description="Agents that require human approval",
    )
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout_seconds: int = Field(default=3600, description="Workflow timeout")
    checkpoint_interval: int = Field(default=5, description="Checkpoint every N nodes")
    enable_rollback: bool = Field(default=True, description="Enable rollback feature")
    max_rollback_states: int = Field(
        default=100, description="Maximum states to keep for rollback"
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class WorkflowExecution(BaseModel):
    """Workflow execution tracking model."""

    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = Field(description="Workflow identifier")
    config: WorkflowConfig = Field(description="Workflow configuration")
    status: WorkflowStatus = Field(
        default=WorkflowStatus.PENDING, description="Execution status"
    )
    current_agent: Optional[AgentRole] = Field(
        default=None, description="Currently executing agent"
    )
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    elapsed_time_seconds: float = Field(default=0.0)
    total_tokens: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)
    agent_executions: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of agent executions"
    )
    checkpoints: List[str] = Field(default_factory=list, description="Checkpoint IDs")
    error: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}


class HumanApprovalRequest(BaseModel):
    """Human approval request model."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = Field(description="Workflow identifier")
    agent_role: AgentRole = Field(description="Agent requesting approval")
    approval_point: str = Field(description="Description of what needs approval")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Context for approval decision"
    )
    result_preview: Dict[str, Any] = Field(description="Preview of agent's output")
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = Field(default="pending")  # pending, approved, rejected, timeout
    feedback: Optional[str] = Field(default=None)
    approved_at: Optional[datetime] = Field(default=None)
    approved_by: Optional[str] = Field(default=None)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class RollbackRequest(BaseModel):
    """Rollback request model."""

    workflow_id: str = Field(description="Workflow identifier")
    target_checkpoint_id: str = Field(description="Checkpoint to rollback to")
    target_state_version: Optional[int] = Field(
        default=None, description="Optional specific state version"
    )
    reason: str = Field(description="Reason for rollback")
    requested_by: str = Field(description="User or system requesting rollback")
    preserve_memory: bool = Field(
        default=True, description="Preserve memory after rollback"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
