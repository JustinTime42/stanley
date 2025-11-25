"""Multi-line input handling."""

import time
from typing import Optional, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys


class MultilineInputHandler:
    """
    Handler for multi-line input.

    Supports:
    - Backslash continuation (\\)
    - Paste detection (rapid input)
    - Code block detection (```)
    """

    def __init__(
        self,
        paste_threshold_ms: int = 50,
    ):
        """
        Initialize multiline handler.

        Args:
            paste_threshold_ms: Time threshold for paste detection (ms)
        """
        self.paste_threshold_ms = paste_threshold_ms
        self._last_input_time: float = 0
        self._buffer: list[str] = []
        self._in_code_block: bool = False

    def create_key_bindings(self) -> KeyBindings:
        """
        Create key bindings for multiline input.

        Returns:
            KeyBindings instance
        """
        kb = KeyBindings()

        @kb.add(Keys.Escape, Keys.Enter)
        def _(event):
            """Alt+Enter or Escape+Enter for newline."""
            event.current_buffer.insert_text("\n")

        return kb

    def should_continue(self, text: str) -> bool:
        """
        Check if input should continue to next line.

        Args:
            text: Current input text

        Returns:
            True if should continue
        """
        if not text:
            return False

        # Trailing backslash
        if text.rstrip().endswith("\\"):
            return True

        # Unclosed code block
        if text.count("```") % 2 != 0:
            return True

        # Unclosed parentheses/brackets (simple check)
        open_count = text.count("(") + text.count("[") + text.count("{")
        close_count = text.count(")") + text.count("]") + text.count("}")
        if open_count > close_count:
            return True

        return False

    def is_paste(self) -> bool:
        """
        Check if current input is likely a paste.

        Returns:
            True if input timing suggests paste
        """
        current_time = time.time() * 1000  # Convert to ms
        time_diff = current_time - self._last_input_time
        self._last_input_time = current_time

        return time_diff < self.paste_threshold_ms

    def process_line(self, line: str) -> tuple[str, bool]:
        """
        Process a line of input.

        Args:
            line: Input line

        Returns:
            Tuple of (processed_line, needs_more)
        """
        # Track code blocks
        code_block_count = line.count("```")
        for _ in range(code_block_count):
            self._in_code_block = not self._in_code_block

        # Check if we need more input
        needs_more = self.should_continue(line) or self._in_code_block

        # Process backslash continuation
        if line.rstrip().endswith("\\") and not self._in_code_block:
            processed = line.rstrip()[:-1]  # Remove backslash
        else:
            processed = line

        return processed, needs_more

    def accumulate(self, line: str) -> Optional[str]:
        """
        Accumulate lines until input is complete.

        Args:
            line: New line to add

        Returns:
            Complete input if done, None if more needed
        """
        processed, needs_more = self.process_line(line)
        self._buffer.append(processed)

        if not needs_more:
            complete = "\n".join(self._buffer)
            self._buffer = []
            self._in_code_block = False
            return complete

        return None

    def reset(self) -> None:
        """Reset the accumulator state."""
        self._buffer = []
        self._in_code_block = False
        self._last_input_time = 0

    def get_continuation_prompt(self) -> str:
        """
        Get prompt for continuation lines.

        Returns:
            Continuation prompt string
        """
        if self._in_code_block:
            return "... "
        return "... "


async def get_multiline_input(
    session: PromptSession,
    initial_prompt: str,
    continuation_prompt: str = "... ",
    validator: Optional[Callable[[str], bool]] = None,
) -> str:
    """
    Get potentially multiline input from user.

    Args:
        session: PromptSession instance
        initial_prompt: Prompt for first line
        continuation_prompt: Prompt for continuation lines
        validator: Optional validator function

    Returns:
        Complete input string
    """
    handler = MultilineInputHandler()
    lines = []
    prompt = initial_prompt

    while True:
        line = await session.prompt_async(prompt)

        processed, needs_more = handler.process_line(line)
        lines.append(processed)

        if not needs_more:
            break

        prompt = continuation_prompt

    complete = "\n".join(lines)

    if validator and not validator(complete):
        # Could raise or return empty, depending on use case
        return ""

    return complete
