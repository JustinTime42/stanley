"""Data models for fractal task decomposition system."""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from datetime import datetime
from uuid import uuid4


class TaskType(str, Enum):
    """Types of tasks for strategy selection."""

    CODE_GENERATION = "code_generation"
    CODE_MODIFICATION = "code_modification"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    ARCHITECTURE = "architecture"
    ANALYSIS = "analysis"


class TaskStatus(str, Enum):
    """Status of a task in the decomposition tree."""

    PENDING = "pending"
    DECOMPOSING = "decomposing"
    READY = "ready"  # Ready for execution (leaf node)
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Waiting on dependencies


class Task(BaseModel):
    """Represents a task in the decomposition tree."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique task ID")
    parent_id: Optional[str] = Field(default=None, description="Parent task ID")
    name: str = Field(description="Task name/title")
    description: str = Field(description="Detailed task description")
    type: TaskType = Field(description="Task type for strategy selection")
    status: TaskStatus = Field(default=TaskStatus.PENDING)

    # Decomposition metadata
    depth: int = Field(default=0, description="Depth in decomposition tree")
    is_leaf: bool = Field(default=False, description="Whether task is atomic")
    can_decompose: bool = Field(
        default=True, description="Whether further decomposition possible"
    )

    # Complexity and assignment
    estimated_complexity: float = Field(
        default=0.5, ge=0, le=1, description="Estimated complexity"
    )
    actual_complexity: Optional[float] = Field(
        default=None, description="Actual complexity after execution"
    )
    assigned_model: Optional[str] = Field(
        default=None, description="Assigned model/agent"
    )
    estimated_tokens: int = Field(default=0, description="Estimated token usage")
    estimated_cost: float = Field(default=0.0, description="Estimated cost in USD")

    # Dependencies
    dependencies: Set[str] = Field(
        default_factory=set, description="Task IDs this depends on"
    )
    dependents: Set[str] = Field(
        default_factory=set, description="Task IDs that depend on this"
    )

    # Progress tracking
    progress: float = Field(
        default=0.0, ge=0, le=100, description="Progress percentage"
    )
    subtask_count: int = Field(default=0, description="Number of subtasks")
    completed_subtask_count: int = Field(default=0, description="Completed subtasks")

    # Execution metadata
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)
    execution_time_ms: Optional[int] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    retry_count: int = Field(default=0)

    # Results and artifacts
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Task execution result"
    )
    artifacts: List[str] = Field(
        default_factory=list, description="Generated artifact IDs"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat() if v else None}


class DecompositionTree(BaseModel):
    """Represents the entire task decomposition tree."""

    root_task_id: str = Field(description="Root task ID")
    tasks: Dict[str, Task] = Field(default_factory=dict, description="All tasks by ID")
    execution_order: List[str] = Field(
        default_factory=list, description="Topological sort order"
    )

    # Tree metadata
    total_tasks: int = Field(default=1)
    max_depth: int = Field(default=0)
    leaf_tasks: List[str] = Field(
        default_factory=list, description="Executable leaf task IDs"
    )

    # Progress tracking
    overall_progress: float = Field(
        default=0.0, description="Overall progress percentage"
    )
    completed_tasks: int = Field(default=0)
    failed_tasks: int = Field(default=0)
    blocked_tasks: int = Field(default=0)

    # Cost tracking
    estimated_total_cost: float = Field(default=0.0)
    actual_total_cost: float = Field(default=0.0)

    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class DecompositionStrategy(BaseModel):
    """Strategy for decomposing specific task types."""

    name: str = Field(description="Strategy name")
    task_type: TaskType = Field(description="Task type this strategy handles")
    max_depth: int = Field(default=5, description="Maximum decomposition depth")
    min_complexity_for_decomposition: float = Field(
        default=0.3, description="Minimum complexity to decompose"
    )
    decomposition_patterns: List[str] = Field(
        default_factory=list, description="Patterns for decomposition"
    )


class DecompositionRequest(BaseModel):
    """Request to decompose a task."""

    task_description: str = Field(description="High-level task description")
    task_type: Optional[TaskType] = Field(
        default=None, description="Force specific task type"
    )
    max_depth: int = Field(default=5, ge=1, le=10)
    max_subtasks_per_level: int = Field(default=5, ge=2, le=10)
    complexity_threshold: float = Field(
        default=0.2, ge=0, le=1, description="Min complexity to decompose"
    )
    include_dependencies: bool = Field(default=True)
    estimate_costs: bool = Field(default=True)
    target_model_routing: bool = Field(
        default=True, description="Route subtasks to specific models"
    )


class DecompositionResult(BaseModel):
    """Result of task decomposition."""

    tree: DecompositionTree = Field(description="Complete decomposition tree")
    execution_plan: List[List[str]] = Field(
        default_factory=list,
        description="Batches of tasks that can run in parallel",
    )
    estimated_duration_ms: int = Field(description="Estimated total execution time")
    warnings: List[str] = Field(
        default_factory=list, description="Decomposition warnings"
    )


class ProgressUpdate(BaseModel):
    """Progress update for a task."""

    task_id: str = Field(description="Task being updated")
    progress: float = Field(ge=0, le=100, description="New progress percentage")
    status: TaskStatus = Field(description="New status")
    message: Optional[str] = Field(default=None, description="Progress message")
    partial_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Partial results"
    )


class DependencyValidation(BaseModel):
    """Result of dependency validation."""

    is_valid: bool = Field(description="Whether dependencies form valid DAG")
    has_cycles: bool = Field(
        default=False, description="Whether circular dependencies exist"
    )
    cycles: List[List[str]] = Field(
        default_factory=list, description="Circular dependency chains"
    )
    missing_dependencies: List[str] = Field(
        default_factory=list, description="Referenced but missing tasks"
    )
    execution_order: List[str] = Field(
        default_factory=list, description="Valid execution order if DAG"
    )
