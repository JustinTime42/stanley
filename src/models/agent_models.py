"""Agent request and response models."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from .state_models import AgentRole


class AgentRequest(BaseModel):
    """Request model for agent execution."""

    state: Dict[str, Any] = Field(description="Current workflow state")
    task: Dict[str, Any] = Field(description="Task to execute")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )
    memory_ids: List[str] = Field(
        default_factory=list, description="Relevant memory IDs"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class AgentResponse(BaseModel):
    """Response model from agent execution."""

    success: bool = Field(description="Execution success status")
    result: Dict[str, Any] = Field(description="Execution result")
    next_agent: Optional[AgentRole] = Field(
        default=None, description="Suggested next agent"
    )
    messages: List[Dict[str, Any]] = Field(
        default_factory=list, description="New messages"
    )
    state_updates: Dict[str, Any] = Field(
        default_factory=dict, description="State updates to apply"
    )
    requires_human_approval: bool = Field(
        default=False, description="Needs human review"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        use_enum_values = True


class AgentMetrics(BaseModel):
    """Metrics for agent execution."""

    agent_role: AgentRole = Field(description="Agent role")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    tokens_used: int = Field(default=0, description="Tokens consumed")
    cost_usd: float = Field(default=0.0, description="Estimated cost in USD")
    success: bool = Field(description="Whether execution succeeded")
    retry_count: int = Field(default=0, description="Number of retries")
    memory_operations: int = Field(default=0, description="Number of memory operations")
    error_type: Optional[str] = Field(default=None, description="Error type if failed")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
