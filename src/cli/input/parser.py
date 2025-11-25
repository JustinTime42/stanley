"""Input parsing for CLI."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class InputType(Enum):
    """Type of parsed input."""

    MESSAGE = "message"  # Regular chat message
    COMMAND = "command"  # Slash command
    MEMORY = "memory"  # Memory shortcut (#)
    FILE_REF = "file_ref"  # File reference (@)
    EMPTY = "empty"  # Empty input


@dataclass
class ParsedInput:
    """Result of parsing user input."""

    type: InputType
    content: str
    command_name: Optional[str] = None
    command_args: Optional[str] = None
    file_refs: Optional[List[str]] = None
    is_multiline: bool = False


class InputParser:
    """
    Parser for user input.

    Handles:
    - Slash commands (/help, /model, etc.)
    - Memory shortcuts (#)
    - File references (@filename)
    - Regular messages
    """

    # Patterns
    COMMAND_PATTERN = re.compile(r"^/([a-zA-Z0-9_:-]+)(?:\s+(.*))?$", re.DOTALL)
    MEMORY_PATTERN = re.compile(r"^#\s*(.*)$", re.DOTALL)
    FILE_REF_PATTERN = re.compile(r"@([^\s]+)")
    MULTILINE_CONTINUE = re.compile(r"\\$")

    def parse(self, text: str) -> ParsedInput:
        """
        Parse user input.

        Args:
            text: Raw user input

        Returns:
            ParsedInput with type and parsed content
        """
        if not text or not text.strip():
            return ParsedInput(type=InputType.EMPTY, content="")

        text = text.strip()

        # Check for slash command
        cmd_match = self.COMMAND_PATTERN.match(text)
        if cmd_match:
            return ParsedInput(
                type=InputType.COMMAND,
                content=text,
                command_name=cmd_match.group(1),
                command_args=cmd_match.group(2) or "",
            )

        # Check for memory shortcut
        memory_match = self.MEMORY_PATTERN.match(text)
        if memory_match:
            return ParsedInput(
                type=InputType.MEMORY,
                content=memory_match.group(1),
            )

        # Extract file references
        file_refs = self.FILE_REF_PATTERN.findall(text)

        # Regular message
        return ParsedInput(
            type=InputType.MESSAGE,
            content=text,
            file_refs=file_refs if file_refs else None,
        )

    def needs_continuation(self, text: str) -> bool:
        """
        Check if input needs continuation (multiline).

        Args:
            text: Current input text

        Returns:
            True if input ends with continuation marker
        """
        if not text:
            return False

        # Check for backslash at end of line
        if self.MULTILINE_CONTINUE.search(text):
            return True

        # Check for unclosed code blocks
        code_block_count = text.count("```")
        if code_block_count % 2 != 0:
            return True

        return False

    def extract_file_refs(self, text: str) -> List[str]:
        """
        Extract all file references from text.

        Args:
            text: Input text

        Returns:
            List of file paths referenced
        """
        return self.FILE_REF_PATTERN.findall(text)

    def strip_continuation(self, text: str) -> str:
        """
        Remove continuation markers from text.

        Args:
            text: Input text with potential continuation markers

        Returns:
            Cleaned text
        """
        # Remove trailing backslash continuations
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            if line.endswith("\\"):
                cleaned.append(line[:-1])
            else:
                cleaned.append(line)
        return "\n".join(cleaned)

    def is_command(self, text: str) -> bool:
        """
        Quick check if input is a command.

        Args:
            text: Input text

        Returns:
            True if input starts with /
        """
        return bool(text and text.strip().startswith("/"))

    def get_command_name(self, text: str) -> Optional[str]:
        """
        Extract command name from input.

        Args:
            text: Input text

        Returns:
            Command name or None
        """
        match = self.COMMAND_PATTERN.match(text.strip())
        return match.group(1) if match else None
