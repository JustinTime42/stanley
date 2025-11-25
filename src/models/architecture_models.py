"""Architecture-related data models for pattern recognition and technology selection."""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime
from uuid import uuid4


class ArchitecturePattern(str, Enum):
    """Common architecture patterns."""
    LAYERED = "layered"
    MICROSERVICES = "microservices"
    EVENT_DRIVEN = "event_driven"
    SERVERLESS = "serverless"
    MONOLITHIC = "monolithic"
    SOA = "service_oriented"
    HEXAGONAL = "hexagonal"
    CQRS = "cqrs"
    EVENT_SOURCING = "event_sourcing"
    PIPE_AND_FILTER = "pipe_and_filter"
    MVC = "mvc"
    MVP = "mvp"
    MVVM = "mvvm"
    REPOSITORY = "repository"
    DOMAIN_DRIVEN = "domain_driven"
    CLEAN = "clean"
    ONION = "onion"


class Technology(BaseModel):
    """Represents a technology choice."""
    name: str = Field(description="Technology name")
    category: str = Field(description="Technology category (database, framework, etc.)")
    version: Optional[str] = Field(default=None, description="Specific version")
    license: str = Field(default="Unknown", description="License type")

    # Characteristics
    maturity: str = Field(default="trial", description="adopt|trial|assess|hold")
    learning_curve: float = Field(default=0.5, ge=0, le=1, description="Learning difficulty")
    community_size: str = Field(default="medium", description="large|medium|small")
    documentation_quality: float = Field(default=0.5, ge=0, le=1, description="Docs quality score")

    # Suitability scores for different aspects
    suitability_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Suitability for different use cases"
    )

    # Compatibility
    compatible_with: List[str] = Field(default_factory=list, description="Compatible technologies")
    incompatible_with: List[str] = Field(default_factory=list, description="Incompatible technologies")

    # Costs
    licensing_cost: float = Field(default=0.0, description="Licensing cost")
    operational_cost: float = Field(default=0.0, description="Operational cost estimate")


class ArchitectureDesign(BaseModel):
    """Represents an architecture design."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Design ID")
    name: str = Field(description="Architecture name")
    description: str = Field(description="Architecture description")

    # Patterns and structure
    patterns: List[str] = Field(default_factory=list, description="Patterns used")
    layers: List[str] = Field(default_factory=list, description="Architecture layers")
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Components specification")

    # Technology stack
    technologies: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Technology choices")

    # Quality attributes
    quality_attributes: Dict[str, float] = Field(
        default_factory=dict,
        description="Quality attribute scores"
    )

    # Validation
    consistency_score: float = Field(default=0.0, ge=0, le=1, description="Architecture consistency")
    completeness_score: float = Field(default=0.0, ge=0, le=1, description="Design completeness")

    # Documentation
    diagrams: List[str] = Field(default_factory=list, description="Diagram references")
    documentation_links: List[str] = Field(default_factory=list, description="Documentation URLs")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    validated_at: Optional[datetime] = Field(default=None)

    model_config = {"ser_json_timedelta": "iso8601", "json_encoders": {datetime: lambda v: v.isoformat()}}


class PatternMatch(BaseModel):
    """Result of pattern recognition."""
    pattern: str = Field(description="Detected pattern")
    confidence: float = Field(ge=0, le=1, description="Detection confidence")
    location: str = Field(description="Where pattern was found")
    evidence: List[str] = Field(default_factory=list, description="Evidence for pattern")
    matches_best_practice: bool = Field(default=True, description="Follows best practices")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")


class TechnologyEvaluation(BaseModel):
    """Technology evaluation result."""
    technology: Dict[str, Any] = Field(description="Evaluated technology")
    score: float = Field(ge=0, le=1, description="Overall suitability score")
    pros: List[str] = Field(default_factory=list, description="Advantages")
    cons: List[str] = Field(default_factory=list, description="Disadvantages")
    risks: List[str] = Field(default_factory=list, description="Identified risks")
    migration_effort: str = Field(default="medium", description="low|medium|high")
    recommendation: str = Field(default="assess", description="adopt|trial|assess|hold")


class PatternDefinition(BaseModel):
    """Definition of an architecture pattern."""
    name: str = Field(description="Pattern name")
    pattern_type: str = Field(description="Pattern type (e.g., layered, microservices)")
    description: str = Field(description="Pattern description")
    when_to_use: List[str] = Field(default_factory=list, description="When to apply this pattern")
    when_not_to_use: List[str] = Field(default_factory=list, description="When to avoid this pattern")
    structure: Dict[str, Any] = Field(default_factory=dict, description="Pattern structure definition")
    benefits: List[str] = Field(default_factory=list, description="Pattern benefits")
    drawbacks: List[str] = Field(default_factory=list, description="Pattern drawbacks")
    examples: List[str] = Field(default_factory=list, description="Example applications")
    related_patterns: List[str] = Field(default_factory=list, description="Related pattern names")
