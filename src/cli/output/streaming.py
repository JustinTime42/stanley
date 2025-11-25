"""Streaming display for CLI."""

import asyncio
import logging
from typing import Optional, AsyncIterator, Any

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from .renderer import OutputRenderer
from ..config.cli_config import CLIConfig

logger = logging.getLogger(__name__)


class StreamingDisplay:
    """
    Streaming display handler.

    CRITICAL: Rich Live display conflicts with prompt_toolkit input.
    Must stop Live context before prompting for input.
    """

    def __init__(
        self,
        renderer: Optional[OutputRenderer] = None,
        config: Optional[CLIConfig] = None,
    ):
        """
        Initialize streaming display.

        Args:
            renderer: Output renderer
            config: CLI configuration
        """
        self.renderer = renderer or OutputRenderer(config)
        self.console = self.renderer.console
        self.config = config or CLIConfig()
        self._accumulated = ""

    async def stream_response(
        self,
        chunks: AsyncIterator[Any],
        prefix: str = "",
    ) -> str:
        """
        Stream response chunks to display.

        PATTERN: Display tokens as they arrive, accumulate for history.
        GOTCHA: Must handle partial markdown gracefully.

        Args:
            chunks: Async iterator of response chunks
            prefix: Optional prefix before response

        Returns:
            Complete accumulated response
        """
        self._accumulated = ""

        if prefix:
            self.console.print(prefix, end="")

        try:
            async for chunk in chunks:
                # Extract content from chunk
                content = self._extract_content(chunk)
                if content:
                    self._accumulated += content
                    # Print without newline for streaming effect
                    self.console.print(content, end="")

        except KeyboardInterrupt:
            self.console.print("\n[dim](interrupted)[/dim]")

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            self.console.print(f"\n[red]Streaming error: {e}[/red]")

        # Ensure newline after streaming completes
        self.console.print()

        return self._accumulated

    def _extract_content(self, chunk: Any) -> str:
        """
        Extract content string from a chunk.

        Args:
            chunk: Response chunk (various formats)

        Returns:
            Content string
        """
        if isinstance(chunk, str):
            return chunk

        # LLMResponse-like object
        if hasattr(chunk, "content"):
            return str(chunk.content)

        # Dictionary
        if isinstance(chunk, dict):
            return chunk.get("content", "") or chunk.get("text", "")

        return str(chunk)

    async def show_spinner(
        self,
        message: str = "Thinking...",
        style: str = "dots",
    ) -> "SpinnerContext":
        """
        Show a spinner while waiting.

        Args:
            message: Spinner message
            style: Spinner style

        Returns:
            SpinnerContext for use with async with
        """
        return SpinnerContext(self.console, message, style)

    def show_typing_indicator(self) -> None:
        """Show typing indicator."""
        self.console.print("[dim]...[/dim]", end="")

    def clear_typing_indicator(self) -> None:
        """Clear typing indicator."""
        # Move cursor back and clear
        self.console.print("\r   \r", end="")


class SpinnerContext:
    """Context manager for spinner display."""

    def __init__(
        self,
        console: Console,
        message: str,
        style: str = "dots",
    ):
        """
        Initialize spinner context.

        Args:
            console: Rich console
            message: Spinner message
            style: Spinner style
        """
        self.console = console
        self.message = message
        self.style = style
        self._live: Optional[Live] = None
        self._spinner: Optional[Spinner] = None

    async def __aenter__(self) -> "SpinnerContext":
        """Enter spinner context."""
        self._spinner = Spinner(self.style, text=self.message)
        self._live = Live(self._spinner, console=self.console, refresh_per_second=10)
        self._live.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit spinner context."""
        if self._live:
            self._live.stop()

    def update(self, message: str) -> None:
        """
        Update spinner message.

        Args:
            message: New message
        """
        if self._spinner:
            self._spinner.text = message


async def stream_with_accumulator(
    chunks: AsyncIterator[Any],
    console: Console,
    prefix: str = "",
) -> str:
    """
    Stream chunks while accumulating the full response.

    CRITICAL: This is the main streaming function for chat mode.
    Must handle KeyboardInterrupt gracefully.

    Args:
        chunks: Async iterator of chunks
        console: Rich console
        prefix: Optional prefix

    Returns:
        Accumulated response
    """
    accumulated = ""

    if prefix:
        console.print(prefix, end="")

    try:
        async for chunk in chunks:
            if isinstance(chunk, str):
                content = chunk
            elif hasattr(chunk, "content"):
                content = chunk.content
            elif isinstance(chunk, dict):
                content = chunk.get("content", "")
            else:
                content = str(chunk)

            if content:
                accumulated += content
                console.print(content, end="", highlight=False)

    except KeyboardInterrupt:
        console.print("\n[dim](interrupted)[/dim]")

    except asyncio.CancelledError:
        console.print("\n[dim](cancelled)[/dim]")

    # Final newline
    console.print()

    return accumulated
