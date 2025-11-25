"""OpenAI provider for GPT model access."""

import logging
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
import tiktoken
from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError, APIError as OpenAIAPIError

from ..base import BaseLLM, RateLimitError, APIError
from ...models.llm_models import ModelConfig

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLM):
    """
    OpenAI provider for GPT model access.

    PATTERN: Official OpenAI SDK with async client
    GOTCHA: Rate limits require exponential backoff
    GOTCHA: Different pricing for input vs output tokens
    """

    def __init__(self, config: ModelConfig, api_key: str):
        """
        Initialize OpenAI provider.

        Args:
            config: Model configuration
            api_key: OpenAI API key
        """
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=api_key)

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(config.model_name)
        except KeyError:
            # Fallback to cl100k_base for newer models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def agenerate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Generate response using OpenAI API.

        CRITICAL: Use AsyncOpenAI for async support
        CRITICAL: Handle rate limits with exponential backoff

        Args:
            messages: Chat messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            **kwargs: Additional parameters (functions, etc.)

        Returns:
            Generated response text

        Raises:
            RateLimitError: When rate limited
            APIError: On API failures
        """
        await self.validate_request(messages, max_tokens)

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return response.choices[0].message.content or ""

        except OpenAIRateLimitError as e:
            self.logger.warning(f"Rate limited: {e}")
            raise RateLimitError(f"OpenAI rate limit: {str(e)}")
        except OpenAIAPIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise APIError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise APIError(f"Unexpected error: {str(e)}")

    async def astream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream response from OpenAI.

        CRITICAL: Use async iteration over stream
        CRITICAL: Handle connection errors gracefully

        Args:
            messages: Chat messages
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Yields:
            Response chunks

        Raises:
            RateLimitError: When rate limited
            APIError: On failures
        """
        await self.validate_request(messages, max_tokens)

        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except OpenAIRateLimitError as e:
            self.logger.warning(f"Rate limited during streaming: {e}")
            raise RateLimitError(f"OpenAI rate limit: {str(e)}")
        except OpenAIAPIError as e:
            self.logger.error(f"OpenAI streaming error: {e}")
            raise APIError(f"OpenAI streaming error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected streaming error: {e}")
            raise APIError(f"Unexpected error: {str(e)}")

    def get_num_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken.

        CRITICAL: Use proper tokenizer for accurate counting
        CRITICAL: Different models have different tokenizers

        Args:
            text: Text to count tokens for

        Returns:
            Exact token count
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            self.logger.warning(f"Token counting failed, using estimate: {e}")
            # Fallback to word-based estimation
            return int(len(text.split()) * 1.3)

    def get_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens for a list of messages.

        PATTERN: OpenAI's token counting includes message formatting
        GOTCHA: Each message has ~4 tokens of formatting overhead

        Args:
            messages: List of chat messages

        Returns:
            Total token count including formatting
        """
        num_tokens = 0
        for message in messages:
            # Every message has formatting overhead
            num_tokens += 4  # <|im_start|>{role}\n{content}<|im_end|>\n

            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += self.get_num_tokens(value)
                    if key == "name":
                        num_tokens += 1  # name has special formatting

        num_tokens += 2  # Every reply is primed with <|im_start|>assistant
        return num_tokens

    async def close(self):
        """Close OpenAI client."""
        await self.client.close()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            loop = asyncio.get_running_loop()
            if loop and not loop.is_closed():
                loop.create_task(self.close())
        except RuntimeError:
            # No running event loop, cannot cleanup async resources
            pass
        except Exception:
            pass
