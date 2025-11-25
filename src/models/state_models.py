"""State models for LangGraph workflow management."""

from typing import Dict, List, Any, Optional, TypedDict
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Agent roles in the workflow."""

    COORDINATOR = "coordinator"
    PLANNER = "planner"
    ARCHITECT = "architect"
    IMPLEMENTER = "implementer"
    TESTER = "tester"
    VALIDATOR = "validator"
    DEBUGGER = "debugger"


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    PENDING = "pending"
    PLANNING = "planning"
    DESIGNING = "designing"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    VALIDATING = "validating"
    DEBUGGING = "debugging"
    COMPLETE = "complete"
    FAILED = "failed"
    HUMAN_REVIEW = "human_review"


# TypedDict for LangGraph state (required format)
# CRITICAL: LangGraph requires TypedDict, not Pydantic models directly
class AgentState(TypedDict, total=False):
    """
    Main state schema for LangGraph workflow.
    Uses TypedDict as required by StateGraph.

    CRITICAL: TypedDict is required by LangGraph, not Pydantic BaseModel
    All fields are optional (total=False) to allow partial state updates
    """

    # Core workflow state
    workflow_id: str
    project_id: str
    session_id: str
    status: str  # WorkflowStatus value

    # Agent communication
    messages: List[Dict[str, Any]]
    current_agent: str  # AgentRole value
    next_agent: Optional[str]

    # Task management
    task: Dict[str, Any]
    subtasks: List[Dict[str, Any]]
    completed_subtasks: List[str]

    # Decomposition tree
    decomposition_tree_id: Optional[str]  # Reference to decomposition tree

    # Results and artifacts
    plan: Optional[Dict[str, Any]]
    architecture: Optional[Dict[str, Any]]
    implementation: Optional[Dict[str, Any]]
    test_results: Optional[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]]
    debug_info: Optional[Dict[str, Any]]

    # Control flow
    retry_count: int
    max_retries: int
    should_continue: bool
    requires_human_approval: bool
    human_feedback: Optional[str]

    # Checkpoint and rollback
    swarm_checkpoint_id: str
    parent_checkpoint_id: Optional[str]
    state_version: int

    # Memory references
    memory_ids: List[str]
    context: Dict[str, Any]

    # Metadata
    created_at: str  # ISO format datetime
    updated_at: str
    elapsed_time: float
    token_count: int
    cost: float


class StateSnapshot(BaseModel):
    """
    Snapshot of workflow state for rollback and history.
    Uses Pydantic for validation and serialization.
    """

    snapshot_id: str = Field(description="Unique snapshot identifier")
    workflow_id: str = Field(description="Workflow identifier")
    state_version: int = Field(description="State version number")
    swarm_checkpoint_id: str = Field(description="Associated checkpoint ID")
    state_data: Dict[str, Any] = Field(description="Serialized state")
    timestamp: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(description="Agent or system that created snapshot")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}
