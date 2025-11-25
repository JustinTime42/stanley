"""Tab completion for CLI."""

from typing import Iterable, Optional, TYPE_CHECKING

from prompt_toolkit.completion import (
    Completer,
    Completion,
    PathCompleter,
    WordCompleter,
)
from prompt_toolkit.document import Document

if TYPE_CHECKING:
    from ..commands.base import CommandRegistry


class CLICompleter(Completer):
    """
    Combined completer for CLI.

    Handles:
    - Slash command completion
    - Command argument completion
    - File path completion (for @ references)
    """

    def __init__(
        self,
        command_registry: Optional["CommandRegistry"] = None,
    ):
        """
        Initialize CLI completer.

        Args:
            command_registry: Command registry for slash commands
        """
        self.command_registry = command_registry
        self.path_completer = PathCompleter(
            expanduser=True,
            only_directories=False,
        )

    def get_completions(
        self,
        document: Document,
        complete_event,
    ) -> Iterable[Completion]:
        """
        Get completions for current input.

        Args:
            document: Current document
            complete_event: Completion event

        Yields:
            Completion objects
        """
        text = document.text_before_cursor
        text_stripped = text.lstrip()

        # Slash command completion
        if text_stripped.startswith("/"):
            yield from self._complete_command(text_stripped, document)

        # File reference completion
        elif "@" in text:
            yield from self._complete_file_ref(text, document)

        # No other completions for now

    def _complete_command(
        self,
        text: str,
        document: Document,
    ) -> Iterable[Completion]:
        """
        Complete slash commands.

        Args:
            text: Text starting with /
            document: Current document

        Yields:
            Command completions
        """
        if not self.command_registry:
            return

        # Get command prefix
        parts = text[1:].split(maxsplit=1)
        cmd_prefix = parts[0] if parts else ""

        # Check if we're completing command name or arguments
        if len(parts) <= 1 and " " not in text:
            # Completing command name
            for name in self.command_registry.get_all_names():
                if name.startswith(cmd_prefix):
                    # Calculate start position
                    start_pos = -len(cmd_prefix) - 1  # Include the /
                    yield Completion(
                        f"/{name}",
                        start_position=start_pos,
                        display=f"/{name}",
                        display_meta=self._get_command_meta(name),
                    )
        else:
            # Could add argument completion here
            pass

    def _complete_file_ref(
        self,
        text: str,
        document: Document,
    ) -> Iterable[Completion]:
        """
        Complete file references (@path).

        Args:
            text: Text containing @
            document: Current document

        Yields:
            File path completions
        """
        # Find the @ position
        at_pos = text.rfind("@")
        if at_pos < 0:
            return

        # Get path prefix after @
        path_prefix = text[at_pos + 1 :]

        # Don't complete if there's a space after the path
        if " " in path_prefix:
            return

        # Get completions from path completer
        path_doc = Document(path_prefix)
        for completion in self.path_completer.get_completions(
            path_doc, None  # type: ignore
        ):
            yield Completion(
                completion.text,
                start_position=completion.start_position,
                display=completion.display,
                display_meta="file",
            )

    def _get_command_meta(self, name: str) -> str:
        """
        Get metadata for command.

        Args:
            name: Command name

        Returns:
            Short description
        """
        if not self.command_registry:
            return ""

        command = self.command_registry.get(name)
        if command:
            # Return first 30 chars of description
            desc = command.description[:30]
            if len(command.description) > 30:
                desc += "..."
            return desc

        return ""


class CommandCompleter(WordCompleter):
    """Simple word completer for commands."""

    def __init__(self, command_registry: Optional["CommandRegistry"] = None):
        """
        Initialize command completer.

        Args:
            command_registry: Command registry
        """
        words = []
        meta_dict = {}

        if command_registry:
            for name in command_registry.get_all_names():
                words.append(f"/{name}")
                command = command_registry.get(name)
                if command:
                    meta_dict[f"/{name}"] = command.description

        super().__init__(
            words=words,
            meta_dict=meta_dict,
            ignore_case=True,
        )


def create_completer(
    command_registry: Optional["CommandRegistry"] = None,
) -> CLICompleter:
    """
    Create a CLI completer.

    Args:
        command_registry: Command registry

    Returns:
        CLICompleter instance
    """
    return CLICompleter(command_registry=command_registry)
