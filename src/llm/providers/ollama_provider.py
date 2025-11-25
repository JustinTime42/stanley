"""Ollama provider for local model integration."""

import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, AsyncIterator
import httpx

from ..base import BaseLLM, APIError, StreamingError
from ...models.llm_models import ModelConfig

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLM):
    """
    Ollama provider for local model access.

    PATTERN: HTTP-based API communication with async httpx
    GOTCHA: Models must be pulled before use with ollama pull
    GOTCHA: Streaming uses newline-delimited JSON
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize Ollama provider.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minutes for large models
        self.base_url = config.api_endpoint or "http://localhost:11434"

    async def check_model_available(self) -> bool:
        """
        Check if model is available locally.

        Returns:
            True if model is pulled and available
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            return self.config.model_name in models
        except Exception as e:
            self.logger.error(f"Error checking model availability: {e}")
            return False

    async def agenerate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Generate response using Ollama API.

        CRITICAL: Ollama uses /api/chat endpoint for chat format
        CRITICAL: Check model availability first

        Args:
            messages: Chat messages
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Generated response text

        Raises:
            APIError: On API failures
        """
        await self.validate_request(messages, max_tokens)

        # Check model availability
        if not await self.check_model_available():
            self.logger.warning(
                f"Model {self.config.model_name} not available, "
                "attempting to pull..."
            )
            # Note: In production, we'd queue a model pull
            # For now, raise an error
            raise APIError(
                f"Model {self.config.model_name} not available. "
                f"Run: ollama pull {self.config.model_name}"
            )

        # Prepare request
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]

        except httpx.TimeoutException as e:
            self.logger.error(f"Ollama request timeout (model may be loading): {e}")
            raise APIError(f"Ollama request timeout after {self.client.timeout}s. Try again or increase timeout.")
        except httpx.HTTPError as e:
            self.logger.error(f"Ollama API error: {e}")
            raise APIError(f"Ollama API error: {str(e)}")
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
        Stream response from Ollama.

        CRITICAL: Ollama streams newline-delimited JSON
        CRITICAL: Use aiter() for async iteration

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

        # Prepare request
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "message" in data:
                                content = data["message"].get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            self.logger.warning(f"Failed to parse line: {line}")

        except httpx.HTTPError as e:
            self.logger.error(f"Ollama streaming error: {e}")
            raise StreamingError(f"Ollama streaming error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected streaming error: {e}")
            raise StreamingError(f"Unexpected error: {str(e)}")

    def get_num_tokens(self, text: str) -> int:
        """
        Estimate token count for Ollama models.

        PATTERN: Simple word-based estimation (Ollama doesn't provide tokenizer)
        GOTCHA: This is an approximation, actual count may vary

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Simple estimation: ~1.3 tokens per word
        words = text.split()
        return int(len(words) * 1.3)

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
