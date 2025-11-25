"""High-level LLM orchestration service integrating routing, fallback, and caching."""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..llm import (
    BaseLLM,
    TaskComplexityAnalyzer,
    ModelRouter,
    FallbackChainManager,
    LLMResponseCache,
)
from ..llm.providers import (
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from ..models.llm_models import (
    LLMRequest,
    LLMResponse,
    ModelConfig,
    ModelProvider,
)
from ..models.routing_models import RoutingDecision
from ..config.llm_config import LLMConfig

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """
    High-level LLM service orchestrating routing, fallback, and caching.

    PATTERN: Facade pattern like MemoryOrchestrator
    CRITICAL: Single entry point for all LLM operations
    GOTCHA: Initialize providers lazily to avoid unnecessary connections
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
    ):
        """
        Initialize LLM orchestrator.

        Args:
            config: LLM configuration (creates default if None)
        """
        self.config = config or LLMConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.analyzer = TaskComplexityAnalyzer()
        self.cache = LLMResponseCache(
            max_size=1000,
            default_ttl=self.config.cache_ttl,
        ) if self.config.enable_cache else None
        self.fallback_manager = FallbackChainManager(
            max_retries=self.config.max_retries,
            max_delay=self.config.fallback_timeout,
        )

        # Initialize providers (lazy)
        self.providers: Dict[str, BaseLLM] = {}
        self._initialize_providers()

        # Initialize router
        available_models = self.config.get_model_configs()
        self.router = ModelRouter(
            available_models=available_models,
            analyzer=self.analyzer,
            prefer_local=self.config.prefer_local,
            routing_strategy=self.config.routing_strategy,
        )

        self.logger.info(
            f"LLM Orchestrator initialized with {len(available_models)} models"
        )

    def _initialize_providers(self) -> None:
        """
        Initialize LLM providers.

        PATTERN: Create providers for each configured model
        CRITICAL: Handle missing API keys gracefully
        """
        model_configs = self.config.get_model_configs()

        for model_config in model_configs:
            try:
                if model_config.provider == ModelProvider.OLLAMA:
                    provider = OllamaProvider(model_config)

                elif model_config.provider == ModelProvider.OPENAI:
                    if not self.config.openai_api_key:
                        self.logger.warning("OpenAI API key not configured")
                        continue
                    provider = OpenAIProvider(
                        model_config,
                        api_key=self.config.openai_api_key,
                    )

                elif model_config.provider == ModelProvider.OPENROUTER:
                    if not self.config.openrouter_api_key:
                        self.logger.warning("OpenRouter API key not configured")
                        continue
                    provider = OpenRouterProvider(
                        model_config,
                        api_key=self.config.openrouter_api_key,
                    )

                else:
                    self.logger.warning(
                        f"Unknown provider: {model_config.provider}"
                    )
                    continue

                self.providers[model_config.model_name] = provider
                self.logger.debug(f"Initialized provider: {model_config.model_name}")

            except Exception as e:
                self.logger.error(
                    f"Failed to initialize {model_config.model_name}: {e}"
                )

    async def generate_response(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate LLM response with routing, caching, and fallback.

        PATTERN: Check cache -> Route -> Execute with fallback -> Cache result
        CRITICAL: This is the main entry point for all agents

        Args:
            request: LLM request

        Returns:
            LLM response

        Raises:
            Exception: If all providers fail
        """
        start_time = datetime.now()

        # Step 1: Route request to determine model
        routing_decision = self.router.route_request(request)

        self.logger.info(
            f"Routing decision: {routing_decision.selected_model.model_name} "
            f"(est. cost: ${routing_decision.estimated_cost:.4f})"
        )

        # Step 2: Check cache if enabled
        if self.cache and request.use_cache and routing_decision.cache_key:
            cached_response = await self.cache.get(routing_decision.cache_key)
            if cached_response:
                self.logger.info(
                    f"Cache hit! Saved ${cached_response.total_cost:.4f}"
                )
                return cached_response

        # Step 3: Build fallback chain
        fallback_chain = self._build_fallback_chain(routing_decision)

        if not fallback_chain:
            raise Exception("No available providers for this request")

        # Step 4: Execute with fallback
        try:
            response = await self.fallback_manager.execute_with_fallback(
                request=request,
                providers=fallback_chain,
            )

            # Step 5: Cache response if enabled
            if self.cache and request.use_cache and routing_decision.cache_key:
                await self.cache.set(
                    cache_key=routing_decision.cache_key,
                    response=response,
                )

            latency_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            response.latency_ms = latency_ms

            self.logger.info(
                f"Generated response: {response.model_used} "
                f"(cost: ${response.total_cost:.4f}, "
                f"latency: {latency_ms}ms, "
                f"tokens: {response.input_tokens + response.output_tokens})"
            )

            return response

        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            raise

    async def stream_response(
        self,
        request: LLMRequest,
    ):
        """
        Stream LLM response.

        PATTERN: Similar to generate_response but streaming
        GOTCHA: Caching is more complex with streaming

        Args:
            request: LLM request

        Yields:
            Response chunks
        """
        # Route request
        routing_decision = self.router.route_request(request)

        self.logger.info(
            f"Streaming with: {routing_decision.selected_model.model_name}"
        )

        # Build fallback chain
        fallback_chain = self._build_fallback_chain(routing_decision)

        if not fallback_chain:
            raise Exception("No available providers for streaming")

        # Stream with fallback
        async for chunk in self.fallback_manager.execute_streaming_with_fallback(
            request=request,
            providers=fallback_chain,
        ):
            yield chunk

    def _build_fallback_chain(
        self,
        routing_decision: RoutingDecision,
    ) -> List[BaseLLM]:
        """
        Build fallback chain from routing decision.

        PATTERN: Convert model configs to provider instances
        CRITICAL: Filter out unavailable providers

        Args:
            routing_decision: Routing decision with selected and fallback models

        Returns:
            List of provider instances
        """
        chain = []

        # Add primary model
        primary_provider = self.providers.get(
            routing_decision.selected_model.model_name
        )
        if primary_provider:
            chain.append(primary_provider)

        # Add fallback models
        for fallback_model in routing_decision.fallback_models:
            fallback_provider = self.providers.get(fallback_model.model_name)
            if fallback_provider and fallback_provider not in chain:
                chain.append(fallback_provider)

        return chain

    def get_available_models(self) -> List[str]:
        """
        Get list of available model names.

        Returns:
            List of model names
        """
        return list(self.providers.keys())

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Cache stats dictionary
        """
        if not self.cache:
            return {"enabled": False}

        stats = self.cache.get_stats()
        stats["enabled"] = True
        stats["cost_savings"] = self.cache.estimate_cost_savings()

        return stats

    async def cleanup(self) -> None:
        """Clean up resources."""
        # Close all providers
        for provider in self.providers.values():
            try:
                if hasattr(provider, "close"):
                    await provider.close()
            except Exception as e:
                self.logger.error(f"Error closing provider: {e}")

        # Clean cache
        if self.cache:
            self.cache.cleanup_expired()

        self.logger.info("LLM Orchestrator cleaned up")

    async def close(self) -> None:
        """Alias for cleanup() to match common close() pattern."""
        await self.cleanup()
