"""Base command infrastructure."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from ..session.state import SessionState
    from ..config.cli_config import CLIConfig
    from ...services.llm_service import LLMOrchestrator
    from ...services.tool_service import ToolOrchestrator
    from ...services.memory_service import MemoryOrchestrator
    from ...services.workflow_service import WorkflowOrchestrator


class CommandContext:
    """Context passed to command execution."""

    def __init__(
        self,
        session: "SessionState",
        llm_service: Optional["LLMOrchestrator"] = None,
        tool_service: Optional["ToolOrchestrator"] = None,
        memory_service: Optional["MemoryOrchestrator"] = None,
        workflow_service: Optional["WorkflowOrchestrator"] = None,
        console: Optional[Console] = None,
        config: Optional["CLIConfig"] = None,
    ):
        """
        Initialize command context.

        Args:
            session: Current session state
            llm_service: LLM orchestrator
            tool_service: Tool orchestrator
            memory_service: Memory orchestrator
            workflow_service: Workflow orchestrator
            console: Rich console for output
            config: CLI configuration
        """
        self.session = session
        self.llm = llm_service
        self.tools = tool_service
        self.memory = memory_service
        self.workflow = workflow_service
        self.console = console or Console()
        self.config = config


class BaseCommand(ABC):
    """Base class for slash commands."""

    name: str = ""  # Command name (without /)
    description: str = ""  # Help text
    aliases: List[str] = []  # Alternative names
    arguments: str = ""  # Argument hint for help

    @abstractmethod
    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """
        Execute the command.

        Args:
            args: Arguments passed to command
            context: Execution context with services

        Returns:
            Optional response to display
        """
        pass

    def get_help(self) -> str:
        """
        Get help text for the command.

        Returns:
            Help string
        """
        help_text = f"/{self.name}"
        if self.arguments:
            help_text += f" {self.arguments}"
        if self.description:
            help_text += f"\n  {self.description}"
        if self.aliases:
            help_text += f"\n  Aliases: {', '.join('/' + a for a in self.aliases)}"
        return help_text


class CommandRegistry:
    """Registry for slash commands."""

    def __init__(self):
        """Initialize command registry."""
        self._commands: Dict[str, BaseCommand] = {}
        self._aliases: Dict[str, str] = {}  # alias -> command name

    def register(self, command: BaseCommand) -> None:
        """
        Register a command.

        Args:
            command: Command instance to register
        """
        self._commands[command.name.lower()] = command

        # Register aliases
        for alias in command.aliases:
            self._aliases[alias.lower()] = command.name.lower()

    def get(self, name: str) -> Optional[BaseCommand]:
        """
        Get a command by name or alias.

        Args:
            name: Command name or alias

        Returns:
            Command instance if found
        """
        name_lower = name.lower()

        # Check direct command
        if name_lower in self._commands:
            return self._commands[name_lower]

        # Check aliases
        if name_lower in self._aliases:
            return self._commands[self._aliases[name_lower]]

        return None

    def list_commands(self) -> List[BaseCommand]:
        """
        List all registered commands.

        Returns:
            List of command instances
        """
        return list(self._commands.values())

    def get_all_names(self) -> List[str]:
        """
        Get all command names and aliases.

        Returns:
            List of all names/aliases
        """
        names = list(self._commands.keys())
        names.extend(self._aliases.keys())
        return sorted(set(names))


class CustomCommand(BaseCommand):
    """
    Custom command loaded from markdown file.

    Supports frontmatter for configuration:
    - description: Command description
    - allowed-tools: List of allowed tools
    - model: Model override
    - argument-hint: Hint for arguments
    """

    def __init__(
        self,
        name: str,
        content: str,
        description: str = "",
        allowed_tools: Optional[List[str]] = None,
        model: Optional[str] = None,
        argument_hint: str = "",
        source_path: str = "",
        scope: str = "project",  # project or user
    ):
        """
        Initialize custom command.

        Args:
            name: Command name
            content: Prompt content
            description: Command description
            allowed_tools: List of allowed tools
            model: Model override
            argument_hint: Hint for arguments
            source_path: Path to source file
            scope: Command scope (project or user)
        """
        self.name = name
        self.description = description or content[:50] + "..."
        self.content = content
        self.allowed_tools = allowed_tools or []
        self.model_override = model
        self.arguments = argument_hint
        self.source_path = source_path
        self.scope = scope
        self.aliases = []

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """
        Execute the custom command.

        Args:
            args: Arguments to substitute
            context: Execution context

        Returns:
            Expanded prompt (to be processed by chat)
        """
        # Replace $ARGUMENTS placeholder
        prompt = self.content.replace("$ARGUMENTS", args)

        # TODO: Handle !`bash commands` execution
        # TODO: Handle @file references

        return prompt
