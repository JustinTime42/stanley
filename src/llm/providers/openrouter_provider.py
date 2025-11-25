"""OpenRouter provider for multi-model access via unified API."""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, AsyncIterator
import httpx
import tiktoken

from ..base import BaseLLM, RateLimitError, APIError, StreamingError
from ...models.llm_models import ModelConfig

logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseLLM):
    """
    OpenRouter provider for accessing multiple models through unified API.

    PATTERN: OpenAI-compatible API with model name translation
    GOTCHA: Model names require provider prefix (e.g., "anthropic/claude-3-opus")
    GOTCHA: Rate limits vary by model provider
    """

    def __init__(self, config: ModelConfig, api_key: str):
        """
        Initialize OpenRouter provider.

        Args:
            config: Model configuration
            api_key: OpenRouter API key
        """
        super().__init__(config)
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/agent-swarm",
                "X-Title": "Agent Swarm Workflow System",
            },
        )
        self.endpoint = config.api_endpoint or "https://openrouter.ai/api/v1/chat/completions"

        # Initialize tokenizer for token counting
        # Use cl100k_base as default for most modern models
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    async def agenerate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Generate response using OpenRouter API.

        CRITICAL: Model name must include provider prefix
        CRITICAL: Handle rate limits from underlying providers

        Args:
            messages: Chat messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            **kwargs: Additional parameters

        Returns:
            Generated response text

        Raises:
            RateLimitError: When rate limited
            APIError: On API failures
        """
        await self.validate_request(messages, max_tokens)

        # Prepare request payload (OpenAI-compatible)
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Add any additional parameters
        payload.update(kwargs)

        try:
            response = await self.client.post(
                self.endpoint,
                json=payload,
            )

            # Handle rate limiting
            if response.status_code == 429:
                raise RateLimitError(
                    f"OpenRouter rate limit exceeded: {response.text}"
                )

            response.raise_for_status()
            data = response.json()

            return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(f"OpenRouter rate limit: {str(e)}")
            self.logger.error(f"OpenRouter API error: {e}")
            raise APIError(f"OpenRouter API error: {str(e)}")
        except httpx.HTTPError as e:
            self.logger.error(f"HTTP error: {e}")
            raise APIError(f"HTTP error: {str(e)}")
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
        Stream response from OpenRouter.

        CRITICAL: Uses Server-Sent Events (SSE) format
        CRITICAL: Parse data: prefix from SSE chunks

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
        await self.validate_request(messages, max_tokens)

        # Prepare request payload
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        payload.update(kwargs)

        try:
            async with self.client.stream(
                "POST",
                self.endpoint,
                json=payload,
            ) as response:
                if response.status_code == 429:
                    raise RateLimitError("OpenRouter rate limit exceeded")

                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            self.logger.debug(f"Failed to parse SSE data: {data_str}")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(f"OpenRouter rate limit: {str(e)}")
            self.logger.error(f"OpenRouter streaming error: {e}")
            raise StreamingError(f"OpenRouter streaming error: {str(e)}")
        except httpx.HTTPError as e:
            self.logger.error(f"HTTP streaming error: {e}")
            raise StreamingError(f"HTTP error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected streaming error: {e}")
            raise StreamingError(f"Unexpected error: {str(e)}")

    def get_num_tokens(self, text: str) -> int:
        """
        Estimate token count for OpenRouter models.

        PATTERN: Use tiktoken for estimation
        GOTCHA: Different models have different tokenizers

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass

        # Fallback to word-based estimation
        return int(len(text.split()) * 1.3)

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

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
