"""Fallback chain manager for robust LLM request handling."""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime

from .base import BaseLLM, RateLimitError, APIError
from ..models.llm_models import LLMRequest, LLMResponse, ModelProvider

logger = logging.getLogger(__name__)


class FallbackChainManager:
    """
    Manages fallback chains for resilient LLM request execution.

    PATTERN: Cascading fallback with exponential backoff
    CRITICAL: Track which model succeeded for metrics
    GOTCHA: Different providers may have different error types
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        """
        Initialize fallback chain manager.

        Args:
            max_retries: Maximum retry attempts per provider
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.logger = logging.getLogger(__name__)

    async def execute_with_fallback(
        self,
        request: LLMRequest,
        providers: List[BaseLLM],
    ) -> LLMResponse:
        """
        Execute request with fallback chain.

        PATTERN: Try each provider with retries before moving to next
        CRITICAL: Return which model was used for cost tracking

        Args:
            request: LLM request
            providers: Ordered list of providers (primary first)

        Returns:
            LLM response with metadata

        Raises:
            Exception: If all providers fail
        """
        if not providers:
            raise ValueError("No providers available")

        last_error = None
        start_time = datetime.now()

        for provider_index, provider in enumerate(providers):
            self.logger.info(
                f"Attempting provider {provider_index + 1}/{len(providers)}: "
                f"{provider.config.model_name}"
            )

            # Try this provider with retries
            for retry in range(self.max_retries):
                try:
                    # Generate response
                    content = await provider.agenerate(
                        messages=request.messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                    )

                    # Calculate tokens and cost
                    input_tokens = sum(
                        provider.get_num_tokens(msg.get("content", ""))
                        for msg in request.messages
                    )
                    output_tokens = provider.get_num_tokens(content)
                    total_cost = provider.calculate_cost(
                        input_tokens,
                        output_tokens,
                    )

                    # Calculate latency
                    latency_ms = int(
                        (datetime.now() - start_time).total_seconds() * 1000
                    )

                    self.logger.info(
                        f"Success with {provider.config.model_name} "
                        f"(attempt {retry + 1}, provider {provider_index + 1})"
                    )

                    return LLMResponse(
                        content=content,
                        model_used=provider.config.model_name,
                        provider=provider.config.provider,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_cost=total_cost,
                        latency_ms=latency_ms,
                        cache_hit=False,
                        fallback_used=(provider_index > 0),
                    )

                except RateLimitError as e:
                    # Rate limited - apply exponential backoff
                    wait_time = min(
                        self.base_delay * (2 ** retry),
                        self.max_delay,
                    )
                    self.logger.warning(
                        f"Rate limited on {provider.config.model_name} "
                        f"(retry {retry + 1}/{self.max_retries}), "
                        f"waiting {wait_time}s: {e}"
                    )

                    if retry < self.max_retries - 1:
                        await asyncio.sleep(wait_time)

                    last_error = e

                except APIError as e:
                    # API error - log and try next provider
                    self.logger.error(
                        f"API error on {provider.config.model_name}: {e}"
                    )
                    last_error = e
                    break  # Don't retry on API errors, try next provider

                except Exception as e:
                    # Unexpected error
                    self.logger.error(
                        f"Unexpected error on {provider.config.model_name}: {e}",
                        exc_info=True,
                    )
                    last_error = e
                    break  # Try next provider

        # All providers failed
        error_msg = (
            f"All {len(providers)} providers failed. "
            f"Last error: {last_error}"
        )
        self.logger.error(error_msg)
        raise Exception(error_msg)

    async def execute_streaming_with_fallback(
        self,
        request: LLMRequest,
        providers: List[BaseLLM],
    ):
        """
        Execute streaming request with fallback.

        PATTERN: Similar to execute_with_fallback but for streaming
        GOTCHA: Harder to recover from mid-stream failures

        Args:
            request: LLM request
            providers: Ordered list of providers

        Yields:
            Response chunks

        Raises:
            Exception: If all providers fail
        """
        if not providers:
            raise ValueError("No providers available")

        last_error = None

        for provider_index, provider in enumerate(providers):
            self.logger.info(
                f"Attempting streaming with provider {provider_index + 1}/{len(providers)}: "
                f"{provider.config.model_name}"
            )

            # Try this provider (no retries for streaming - too complex)
            try:
                async for chunk in provider.astream(
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                ):
                    yield chunk

                # Success
                self.logger.info(
                    f"Streaming succeeded with {provider.config.model_name}"
                )
                return

            except RateLimitError as e:
                self.logger.warning(
                    f"Rate limited on {provider.config.model_name}: {e}"
                )
                last_error = e
                # Try next provider

            except Exception as e:
                self.logger.error(
                    f"Streaming error on {provider.config.model_name}: {e}"
                )
                last_error = e
                # Try next provider

        # All providers failed
        error_msg = (
            f"All {len(providers)} providers failed for streaming. "
            f"Last error: {last_error}"
        )
        self.logger.error(error_msg)
        raise Exception(error_msg)

    def create_fallback_chain(
        self,
        primary: BaseLLM,
        fallbacks: List[BaseLLM],
    ) -> List[BaseLLM]:
        """
        Create ordered fallback chain.

        PATTERN: Primary first, then fallbacks in order
        GOTCHA: Ensure providers are initialized

        Args:
            primary: Primary provider
            fallbacks: Fallback providers

        Returns:
            Ordered provider list
        """
        chain = [primary] + fallbacks
        self.logger.info(
            f"Created fallback chain: {' -> '.join(p.config.model_name for p in chain)}"
        )
        return chain

    def estimate_success_rate(
        self,
        providers: List[BaseLLM],
        provider_success_rates: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Estimate overall success rate with fallback chain.

        PATTERN: Compound probability of at least one success
        CRITICAL: P(success) = 1 - P(all fail)

        Args:
            providers: Provider chain
            provider_success_rates: Optional success rates per provider

        Returns:
            Estimated overall success rate
        """
        if not provider_success_rates:
            # Use default success rates
            provider_success_rates = {}

        # Probability all providers fail
        prob_all_fail = 1.0

        for provider in providers:
            # Get provider success rate (default 0.9 for cloud, 0.95 for local)
            success_rate = provider_success_rates.get(
                provider.config.model_name,
                0.95 if provider.config.is_local else 0.9,
            )

            # With retries, effective success rate is higher
            effective_success = 1 - (1 - success_rate) ** self.max_retries

            # Multiply probability of failure
            prob_all_fail *= (1 - effective_success)

        # Overall success rate
        overall_success = 1 - prob_all_fail

        self.logger.debug(
            f"Estimated success rate: {overall_success:.2%} "
            f"with {len(providers)} providers"
        )

        return overall_success
