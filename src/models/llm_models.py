"""LLM-related data models."""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime


class ModelProvider(str, Enum):
    """Supported model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    MISTRAL = "mistral"


class ModelCapability(str, Enum):
    """Model capabilities for routing."""

    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    PLANNING = "planning"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    GENERAL = "general"


class TaskComplexity(str, Enum):
    """Task complexity levels."""

    SIMPLE = "simple"      # Local models, basic prompts
    MEDIUM = "medium"      # Better local or cheap cloud
    COMPLEX = "complex"    # Premium models required


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    provider: ModelProvider
    model_name: str = Field(description="Model identifier")
    api_endpoint: Optional[str] = Field(default=None, description="API endpoint")
    max_tokens: int = Field(default=4096, description="Max token limit")
    context_window: int = Field(description="Context window size")
    cost_per_1k_input: float = Field(description="Cost per 1k input tokens")
    cost_per_1k_output: float = Field(description="Cost per 1k output tokens")
    capabilities: List[ModelCapability] = Field(description="Model capabilities")
    performance_score: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Performance rating"
    )
    is_local: bool = Field(default=False, description="Is local model")
    supports_streaming: bool = Field(default=True)
    supports_functions: bool = Field(default=False)
    rate_limit: Optional[int] = Field(
        default=None,
        description="Requests per minute"
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class LLMRequest(BaseModel):
    """Request to LLM service."""

    messages: List[dict] = Field(description="Chat messages")
    agent_role: str = Field(description="Requesting agent role")
    task_description: str = Field(description="Task being performed")
    max_tokens: Optional[int] = Field(default=None)
    temperature: float = Field(default=0.7, ge=0, le=2)
    required_capability: Optional[ModelCapability] = Field(default=None)
    complexity_override: Optional[TaskComplexity] = Field(default=None)
    use_cache: bool = Field(default=True)
    stream: bool = Field(default=False)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class LLMResponse(BaseModel):
    """Response from LLM service."""

    content: str = Field(description="Response content")
    model_used: str = Field(description="Model that generated response")
    provider: ModelProvider
    input_tokens: int
    output_tokens: int
    total_cost: float = Field(description="Cost in USD")
    latency_ms: int = Field(description="Response time")
    cache_hit: bool = Field(default=False)
    fallback_used: bool = Field(default=False)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class PerformanceMetrics(BaseModel):
    """Performance metrics for a model."""

    model_name: str
    provider: ModelProvider
    total_requests: int = Field(default=0)
    successful_requests: int = Field(default=0)
    failed_requests: int = Field(default=0)
    total_tokens: int = Field(default=0)
    total_cost: float = Field(default=0.0)
    average_latency_ms: float = Field(default=0.0)
    success_rate: float = Field(default=1.0)
    last_updated: datetime = Field(default_factory=datetime.now)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
