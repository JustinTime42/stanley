"""Base LLM provider abstraction."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from ..models.llm_models import ModelConfig, LLMResponse

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.

    All providers must implement async generation methods and
    follow consistent error handling patterns.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize LLM provider.

        Args:
            config: Model configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.model_name}")

    @abstractmethod
    async def agenerate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Generate a response asynchronously.

        CRITICAL: Must be async for use in agent workflow
        CRITICAL: Handle rate limits with exponential backoff

        Args:
            messages: Chat messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response

        Raises:
            RateLimitError: When rate limited
            APIError: On API failures
            ValidationError: On invalid inputs
        """
        pass

    @abstractmethod
    async def astream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream response chunks asynchronously.

        CRITICAL: Use aiter() for async iteration
        CRITICAL: Handle connection errors gracefully

        Args:
            messages: Chat messages
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Yields:
            Response chunks

        Raises:
            StreamingError: On streaming failures
        """
        pass

    @abstractmethod
    def get_num_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        CRITICAL: Token counting varies by model
        CRITICAL: Check against context window before sending

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        pass

    async def validate_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
    ) -> None:
        """
        Validate request before sending to API.

        Args:
            messages: Chat messages
            max_tokens: Maximum tokens

        Raises:
            ValidationError: On validation failures
        """
        # Calculate total tokens
        total_tokens = sum(
            self.get_num_tokens(msg.get("content", ""))
            for msg in messages
        )

        if max_tokens:
            total_tokens += max_tokens

        # Check context window
        if total_tokens > self.config.context_window:
            raise ValueError(
                f"Request exceeds context window: {total_tokens} > "
                f"{self.config.context_window}"
            )

        self.logger.debug(f"Request validated: {total_tokens} tokens")

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for request.

        CRITICAL: Account for both input and output tokens
        CRITICAL: Different pricing for prompt vs completion

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD
        """
        input_cost = (input_tokens / 1000) * self.config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.config.cost_per_1k_output
        return input_cost + output_cost

    def supports_capability(self, capability: str) -> bool:
        """
        Check if model supports a capability.

        Args:
            capability: Capability to check

        Returns:
            True if supported
        """
        return capability in [c.value for c in self.config.capabilities]


class RateLimitError(Exception):
    """Raised when rate limited by provider."""

    pass


class APIError(Exception):
    """Raised on API failures."""

    pass


class ValidationError(Exception):
    """Raised on validation failures."""

    pass


class StreamingError(Exception):
    """Raised on streaming failures."""

    pass
