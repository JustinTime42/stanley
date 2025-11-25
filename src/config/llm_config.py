"""LLM system configuration with environment variable loading."""

import os
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from ..models.llm_models import ModelProvider, ModelCapability, ModelConfig

# Load environment variables from .env file
load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for the LLM routing system."""

    # Ollama Configuration
    ollama_host: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        description="Ollama API endpoint",
    )
    ollama_models: List[str] = Field(
        default_factory=lambda: os.getenv(
            "OLLAMA_MODELS",
            "qwen2.5-coder:14b"
        ).split(","),
        description="Available Ollama models",
    )

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key",
    )
    openai_default_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
        description="Default OpenAI model",
    )

    # OpenRouter Configuration
    openrouter_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY"),
        description="OpenRouter API key",
    )
    openrouter_default_model: str = Field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_DEFAULT_MODEL",
            "anthropic/claude-3-sonnet"
        ),
        description="Default OpenRouter model",
    )
    openrouter_endpoint: str = Field(
        default="https://openrouter.ai/api/v1/chat/completions",
        description="OpenRouter API endpoint",
    )

    # Model Routing Configuration
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")),
        description="Maximum retry attempts",
    )
    fallback_timeout: int = Field(
        default_factory=lambda: int(os.getenv("FALLBACK_TIMEOUT", "30")),
        description="Timeout for fallback in seconds",
    )
    enable_cache: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_CACHE", "true").lower() == "true",
        description="Enable response caching",
    )
    cache_ttl: int = Field(
        default_factory=lambda: int(os.getenv("CACHE_TTL", "3600")),
        description="Cache time-to-live in seconds",
    )

    # Cost Limits
    max_cost_per_request: float = Field(
        default_factory=lambda: float(os.getenv("MAX_COST_PER_REQUEST", "0.10")),
        description="Maximum cost per request in USD",
    )
    max_cost_per_workflow: float = Field(
        default_factory=lambda: float(os.getenv("MAX_COST_PER_WORKFLOW", "1.00")),
        description="Maximum cost per workflow in USD",
    )
    daily_cost_limit: float = Field(
        default_factory=lambda: float(os.getenv("DAILY_COST_LIMIT", "50.00")),
        description="Daily cost limit in USD",
    )

    # Routing Preferences
    prefer_local: bool = Field(
        default_factory=lambda: os.getenv("PREFER_LOCAL", "true").lower() == "true",
        description="Prefer local models when possible",
    )
    routing_strategy: str = Field(
        default_factory=lambda: os.getenv("ROUTING_STRATEGY", "cost_optimized"),
        description="Routing strategy: cost_optimized, performance, balanced",
    )

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_model_configs(self) -> List[ModelConfig]:
        """
        Get all available model configurations.

        Returns:
            List of ModelConfig objects
        """
        configs = []

        # Ollama models (local)
        ollama_model_configs = {
            "qwen2.5-coder:14b": {
                "context_window": 32768,
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0,
                "capabilities": [
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.CODE_REVIEW,
                    ModelCapability.DEBUGGING,
                    ModelCapability.TESTING,
                    ModelCapability.DOCUMENTATION,
                    ModelCapability.PLANNING,
                ],
                "performance_score": 0.8,
            },
        }

        for model_name in self.ollama_models:
            if model_name in ollama_model_configs:
                config_data = ollama_model_configs[model_name]
                configs.append(
                    ModelConfig(
                        provider=ModelProvider.OLLAMA,
                        model_name=model_name,
                        api_endpoint=self.ollama_host,
                        is_local=True,
                        **config_data,
                    )
                )

        # OpenAI models
        if self.openai_api_key:
            configs.extend([
                ModelConfig(
                    provider=ModelProvider.OPENAI,
                    model_name="gpt-4o-mini",
                    context_window=128000,
                    cost_per_1k_input=0.00015,
                    cost_per_1k_output=0.0006,
                    capabilities=[
                        ModelCapability.CODE_GENERATION,
                        ModelCapability.CODE_REVIEW,
                        ModelCapability.PLANNING,
                        ModelCapability.DEBUGGING,
                        ModelCapability.TESTING,
                        ModelCapability.DOCUMENTATION,
                    ],
                    performance_score=0.85,
                    supports_functions=True,
                ),
                ModelConfig(
                    provider=ModelProvider.OPENAI,
                    model_name="gpt-4o",
                    context_window=128000,
                    cost_per_1k_input=0.0025,
                    cost_per_1k_output=0.01,
                    capabilities=[
                        ModelCapability.CODE_GENERATION,
                        ModelCapability.CODE_REVIEW,
                        ModelCapability.PLANNING,
                        ModelCapability.DEBUGGING,
                        ModelCapability.TESTING,
                        ModelCapability.DOCUMENTATION,
                    ],
                    performance_score=0.95,
                    supports_functions=True,
                ),
            ])

        # OpenRouter models
        if self.openrouter_api_key:
            configs.extend([
                ModelConfig(
                    provider=ModelProvider.OPENROUTER,
                    model_name="anthropic/claude-3-sonnet",
                    api_endpoint=self.openrouter_endpoint,
                    context_window=200000,
                    cost_per_1k_input=0.003,
                    cost_per_1k_output=0.015,
                    capabilities=[
                        ModelCapability.CODE_GENERATION,
                        ModelCapability.CODE_REVIEW,
                        ModelCapability.PLANNING,
                        ModelCapability.DEBUGGING,
                        ModelCapability.TESTING,
                        ModelCapability.DOCUMENTATION,
                    ],
                    performance_score=0.9,
                    supports_functions=True,
                ),
                ModelConfig(
                    provider=ModelProvider.OPENROUTER,
                    model_name="anthropic/claude-3-opus",
                    api_endpoint=self.openrouter_endpoint,
                    context_window=200000,
                    cost_per_1k_input=0.015,
                    cost_per_1k_output=0.075,
                    capabilities=[
                        ModelCapability.CODE_GENERATION,
                        ModelCapability.CODE_REVIEW,
                        ModelCapability.PLANNING,
                        ModelCapability.DEBUGGING,
                        ModelCapability.TESTING,
                        ModelCapability.DOCUMENTATION,
                    ],
                    performance_score=0.98,
                    supports_functions=True,
                ),
                ModelConfig(
                    provider=ModelProvider.OPENROUTER,
                    model_name="meta-llama/llama-3-70b",
                    api_endpoint=self.openrouter_endpoint,
                    context_window=8192,
                    cost_per_1k_input=0.0007,
                    cost_per_1k_output=0.0008,
                    capabilities=[
                        ModelCapability.GENERAL,
                        ModelCapability.CODE_GENERATION,
                        ModelCapability.DOCUMENTATION,
                    ],
                    performance_score=0.75,
                ),
            ])

        return configs

    def get_model_by_name(self, model_name: str) -> Optional[ModelConfig]:
        """
        Get model configuration by name.

        Args:
            model_name: Name of the model

        Returns:
            ModelConfig if found, None otherwise
        """
        configs = self.get_model_configs()
        for config in configs:
            if config.model_name == model_name:
                return config
        return None
