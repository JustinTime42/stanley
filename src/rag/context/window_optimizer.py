"""Token window optimizer for managing context limits."""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class WindowOptimizer:
    """
    Optimize token usage within context windows.

    PATTERN: Dynamic truncation and compression
    CRITICAL: Different models count tokens differently
    """

    def __init__(self):
        """Initialize window optimizer."""
        self.logger = logger

    def optimize_window(
        self,
        texts: List[str],
        max_tokens: int,
        reserve_tokens: int = 500,
    ) -> List[str]:
        """
        Optimize text list to fit within token window.

        Args:
            texts: List of text pieces
            max_tokens: Maximum total tokens
            reserve_tokens: Tokens to reserve for response

        Returns:
            Optimized list of texts that fit within limit
        """
        available_tokens = max_tokens - reserve_tokens

        # Estimate current token usage
        text_tokens = [(text, self.estimate_tokens(text)) for text in texts]
        total_tokens = sum(tokens for _, tokens in text_tokens)

        # If under limit, return as-is
        if total_tokens <= available_tokens:
            return texts

        # Otherwise, truncate from end until we fit
        selected_texts = []
        current_tokens = 0

        for text, tokens in text_tokens:
            if current_tokens + tokens > available_tokens:
                # Try to fit a truncated version
                remaining_tokens = available_tokens - current_tokens
                if remaining_tokens > 50:  # Minimum viable chunk
                    truncated = self.truncate_to_tokens(text, remaining_tokens)
                    if truncated:
                        selected_texts.append(truncated)
                break

            selected_texts.append(text)
            current_tokens += tokens

        self.logger.info(
            f"Optimized context: {total_tokens} -> {current_tokens} tokens"
        )

        return selected_texts

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        PATTERN: Rough estimation (1 token ≈ 4 characters)
        TODO: Use tiktoken for accurate counting per model

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token
        # This is approximate but works for most models
        return max(1, len(text) // 4)

    def truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
    ) -> Optional[str]:
        """
        Truncate text to fit within token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens

        Returns:
            Truncated text or None if too small
        """
        # Estimate max characters
        max_chars = max_tokens * 4

        if len(text) <= max_chars:
            return text

        # Truncate at word boundary
        truncated = text[:max_chars]

        # Find last space
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]

        # Add truncation indicator
        truncated += "..."

        return truncated if len(truncated) > 20 else None

    def compress_text(
        self,
        text: str,
        target_ratio: float = 0.5,
    ) -> str:
        """
        Compress text by removing less important content.

        PATTERN: Keep first and last sentences, summarize middle
        GOTCHA: This is simplistic, ideally use LLM summarization

        Args:
            text: Text to compress
            target_ratio: Target compression ratio (0-1)

        Returns:
            Compressed text
        """
        if target_ratio >= 1.0:
            return text

        # Split into sentences
        sentences = text.split(". ")

        if len(sentences) <= 3:
            return text

        # Calculate how many sentences to keep
        target_count = max(3, int(len(sentences) * target_ratio))

        # Keep first and last, sample middle
        if target_count >= len(sentences):
            return text

        first_count = target_count // 2
        last_count = target_count - first_count

        compressed_sentences = (
            sentences[:first_count] + ["..."] + sentences[-last_count:]
        )

        return ". ".join(compressed_sentences)

    def calculate_optimal_distribution(
        self,
        items: List[tuple[str, float]],  # (text, score) pairs
        max_tokens: int,
    ) -> List[str]:
        """
        Distribute token budget across items by score.

        Args:
            items: List of (text, relevance_score) tuples
            max_tokens: Total token budget

        Returns:
            Selected texts that maximize relevance within budget
        """
        # Sort by score descending
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)

        selected = []
        current_tokens = 0

        for text, score in sorted_items:
            text_tokens = self.estimate_tokens(text)

            if current_tokens + text_tokens <= max_tokens:
                selected.append(text)
                current_tokens += text_tokens
            elif current_tokens < max_tokens * 0.9:  # If we have room
                # Try to fit truncated version
                remaining = max_tokens - current_tokens
                if remaining > 50:
                    truncated = self.truncate_to_tokens(text, remaining)
                    if truncated:
                        selected.append(truncated)
                        break

        return selected
