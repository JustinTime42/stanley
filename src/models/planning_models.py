"""Planning-related data models for solution exploration and decision making."""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime
from uuid import uuid4


class SolutionApproach(str, Enum):
    """Types of solution approaches."""
    INCREMENTAL = "incremental"      # Build piece by piece
    BIG_BANG = "big_bang"           # Complete rewrite
    STRANGLER = "strangler"         # Gradual replacement
    PARALLEL = "parallel"           # Build alongside existing
    HYBRID = "hybrid"               # Mixed approach


class TradeOffDimension(str, Enum):
    """Dimensions for trade-off analysis."""
    COST = "cost"
    TIME = "time"
    COMPLEXITY = "complexity"
    SCALABILITY = "scalability"
    MAINTAINABILITY = "maintainability"
    PERFORMANCE = "performance"
    SECURITY = "security"
    RELIABILITY = "reliability"
    USABILITY = "usability"
    FLEXIBILITY = "flexibility"


class Solution(BaseModel):
    """Represents a solution alternative."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique solution ID")
    name: str = Field(description="Solution name")
    description: str = Field(description="Detailed solution description")
    approach: SolutionApproach = Field(description="Solution approach type")

    # Components and structure
    components: List[str] = Field(default_factory=list, description="Major components")
    dependencies: List[str] = Field(default_factory=list, description="External dependencies")
    technologies: List[str] = Field(default_factory=list, description="Required technologies")

    # Trade-off scores (0-1 normalized)
    trade_offs: Dict[str, float] = Field(
        default_factory=dict,
        description="Trade-off scores by dimension"
    )

    # Estimates
    estimated_effort_hours: float = Field(default=0.0, description="Estimated implementation effort")
    estimated_cost: float = Field(default=0.0, description="Estimated cost in USD")
    risk_level: float = Field(default=0.5, ge=0, le=1, description="Risk assessment")

    # Pros and cons
    advantages: List[str] = Field(default_factory=list, description="Solution advantages")
    disadvantages: List[str] = Field(default_factory=list, description="Solution disadvantages")
    assumptions: List[str] = Field(default_factory=list, description="Key assumptions")

    # Metadata
    confidence_score: float = Field(default=0.5, ge=0, le=1, description="Confidence in solution")
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"ser_json_timedelta": "iso8601", "json_encoders": {datetime: lambda v: v.isoformat()}}


class PlanningContext(BaseModel):
    """Context for planning decisions."""
    project_constraints: Dict[str, Any] = Field(default_factory=dict, description="Project constraints")
    team_capabilities: List[str] = Field(default_factory=list, description="Team skills/capabilities")
    existing_architecture: Optional[Dict[str, Any]] = Field(default=None, description="Current architecture")
    budget_limit: Optional[float] = Field(default=None, description="Budget constraint")
    timeline_weeks: Optional[int] = Field(default=None, description="Timeline constraint")
    quality_requirements: Dict[str, float] = Field(default_factory=dict, description="Quality thresholds")
    regulatory_requirements: List[str] = Field(default_factory=list, description="Compliance needs")


class DecisionRecord(BaseModel):
    """Architecture Decision Record (ADR)."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Decision ID")
    title: str = Field(description="Decision title")
    status: str = Field(default="proposed", description="proposed|accepted|deprecated|superseded")
    context: str = Field(description="Decision context")
    decision: str = Field(description="Decision made")
    consequences: str = Field(description="Decision consequences")

    # Detailed information
    alternatives_considered: List[Dict[str, Any]] = Field(default_factory=list, description="Alternatives evaluated")
    selection_rationale: str = Field(description="Why this solution was chosen")
    trade_off_analysis: Dict[str, Any] = Field(default_factory=dict, description="Trade-off analysis results")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    author: str = Field(default="AI Agent", description="Decision author")
    related_decisions: List[str] = Field(default_factory=list, description="Related decision IDs")
    supersedes: Optional[str] = Field(default=None, description="Previous decision ID if superseding")

    model_config = {"ser_json_timedelta": "iso8601", "json_encoders": {datetime: lambda v: v.isoformat()}}


class TradeOffAnalysisResult(BaseModel):
    """Result of trade-off analysis."""
    scores: List[float] = Field(description="TOPSIS scores for each solution")
    ranking: List[int] = Field(description="Solution indices in ranked order")
    best_solution_index: int = Field(description="Index of best solution")
    trade_off_matrix: List[List[float]] = Field(description="Raw trade-off matrix")
    weights_used: Dict[str, float] = Field(description="Weights applied to dimensions")
    normalized_matrix: Optional[List[List[float]]] = Field(default=None, description="Normalized decision matrix")

    model_config = {"arbitrary_types_allowed": True}
