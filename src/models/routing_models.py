"""Model routing decision models."""

from pydantic import BaseModel, Field
from typing import List, Optional
from .llm_models import (
    TaskComplexity,
    ModelCapability,
    ModelConfig,
)


class TaskAnalysis(BaseModel):
    """Analysis of a task for routing."""

    task_id: str
    complexity: TaskComplexity
    estimated_tokens: int = Field(description="Estimated token usage")
    required_capabilities: List[ModelCapability]
    requires_functions: bool = Field(default=False)
    requires_vision: bool = Field(default=False)
    confidence: float = Field(ge=0, le=1, description="Analysis confidence")
    reasoning: str = Field(description="Complexity reasoning")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class RoutingDecision(BaseModel):
    """Model routing decision."""

    task_analysis: TaskAnalysis
    selected_model: ModelConfig
    fallback_models: List[ModelConfig] = Field(default_factory=list)
    estimated_cost: float = Field(description="Estimated cost in USD")
    estimated_latency_ms: int = Field(description="Estimated latency")
    routing_reason: str = Field(description="Why this model was selected")
    cache_key: Optional[str] = Field(
        default=None,
        description="Cache lookup key"
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
