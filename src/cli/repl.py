"""Main REPL loop implementation."""

import logging
from typing import Optional, TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings

from .config.cli_config import CLIConfig, ensure_directories
from .session.state import SessionState, CLIMode
from .session.manager import SessionManager
from .session.history import HistoryManager
from .commands.base import CommandContext, CommandRegistry
from .commands.builtin import load_builtin_commands
from .commands.config_cmd import ConfigCommand
from .commands.memory_cmd import MemoryCommand
from .commands.loader import load_custom_commands
from .input.parser import InputParser, InputType
from .input.completer import create_completer
from .input.multiline import MultilineInputHandler
from .output.renderer import OutputRenderer
from .modes.chat import ChatMode
from .modes.task import TaskMode

if TYPE_CHECKING:
    from ..services.llm_service import LLMOrchestrator
    from ..services.tool_service import ToolOrchestrator
    from ..services.memory_service import MemoryOrchestrator
    from ..services.workflow_service import WorkflowOrchestrator

logger = logging.getLogger(__name__)


class REPL:
    """
    Main REPL (Read-Eval-Print Loop) implementation.

    CRITICAL: Uses prompt_toolkit with prompt_async() for non-blocking IO.
    CRITICAL: Rich Live display must stop before prompting for input.
    """

    def __init__(
        self,
        session: SessionState,
        session_manager: SessionManager,
        config: CLIConfig,
        commands: CommandRegistry,
        renderer: OutputRenderer,
        llm_service: Optional["LLMOrchestrator"] = None,
        tool_service: Optional["ToolOrchestrator"] = None,
        memory_service: Optional["MemoryOrchestrator"] = None,
        workflow_service: Optional["WorkflowOrchestrator"] = None,
    ):
        """
        Initialize REPL.

        Args:
            session: Current session state
            session_manager: Session persistence manager
            config: CLI configuration
            commands: Command registry
            renderer: Output renderer
            llm_service: LLM orchestrator
            tool_service: Tool orchestrator
            memory_service: Memory orchestrator
            workflow_service: Workflow orchestrator
        """
        self.session = session
        self.session_manager = session_manager
        self.config = config
        self.commands = commands
        self.renderer = renderer
        self.console = renderer.console
        self.llm = llm_service
        self.tools = tool_service
        self.memory = memory_service
        self.workflow = workflow_service

        # Initialize parser and multiline handler
        self.parser = InputParser()
        self.multiline = MultilineInputHandler()

        # Initialize history
        self.history_manager = HistoryManager(config, session.working_directory)

        # Prompt session is lazily initialized (only for interactive mode)
        self._prompt_session: Optional[PromptSession] = None

        # Initialize modes
        self.chat_mode = ChatMode(
            session=session,
            config=config,
            renderer=renderer,
            console=self.console,
            llm_service=llm_service,
            tool_service=tool_service,
            memory_service=memory_service,
            workflow_service=workflow_service,
        )
        self.task_mode = TaskMode(
            session=session,
            config=config,
            renderer=renderer,
            console=self.console,
            llm_service=llm_service,
            tool_service=tool_service,
            memory_service=memory_service,
            workflow_service=workflow_service,
        )

    def _create_key_bindings(self) -> KeyBindings:
        """
        Create custom key bindings.

        Returns:
            KeyBindings instance
        """
        kb = KeyBindings()

        @kb.add("c-c")
        def handle_ctrl_c(event):
            """Handle Ctrl+C - cancel current input."""
            event.current_buffer.reset()

        @kb.add("c-d")
        def handle_ctrl_d(event):
            """Handle Ctrl+D - exit."""
            event.app.exit()

        @kb.add("c-l")
        def handle_ctrl_l(event):
            """Handle Ctrl+L - clear screen."""
            event.app.renderer.clear()

        return kb

    @property
    def prompt_session(self) -> PromptSession:
        """
        Lazily initialize prompt session.

        Only creates the PromptSession when actually needed (interactive mode).
        This avoids errors in non-interactive contexts (e.g., -p flag, piped input).
        """
        if self._prompt_session is None:
            self._prompt_session = PromptSession(
                history=self.history_manager.get_history(),
                auto_suggest=AutoSuggestFromHistory(),
                key_bindings=self._create_key_bindings(),
                vi_mode=self.config.vim_mode,
                completer=create_completer(self.commands),
                complete_while_typing=True,
            )
        return self._prompt_session

    async def run(self, initial_prompt: Optional[str] = None) -> None:
        """
        Main REPL loop.

        Args:
            initial_prompt: Optional initial prompt to process
        """
        self._display_welcome()

        # Process initial prompt if provided
        if initial_prompt:
            await self._process_input(initial_prompt)

        while True:
            try:
                # Get prompt based on mode
                prompt_text = self._get_prompt()

                # Get user input (async for non-blocking)
                user_input = await self.prompt_session.prompt_async(prompt_text)

                if not user_input or not user_input.strip():
                    continue

                # Check for multiline continuation
                while self.multiline.should_continue(user_input):
                    continuation = await self.prompt_session.prompt_async("... ")
                    processed, _ = self.multiline.process_line(continuation)
                    user_input += "\n" + processed

                # Process input
                await self._process_input(user_input)

            except KeyboardInterrupt:
                # Ctrl+C pressed - reset and continue
                self.console.print()
                continue

            except EOFError:
                # Ctrl+D pressed - exit
                break

            except SystemExit:
                # /quit command
                break

            except Exception as e:
                logger.error(f"REPL error: {e}")
                self.renderer.render_error(e)

        self._display_goodbye()

    async def process_once(self, prompt: str) -> str:
        """
        Process a single prompt and return the response.

        Used for one-shot mode (-p flag).

        Args:
            prompt: User prompt

        Returns:
            Assistant response
        """
        # Parse input to check for commands
        parsed = self.parser.parse(prompt)

        if parsed.type == InputType.COMMAND:
            # Handle command
            await self._handle_command(
                parsed.command_name or "",
                parsed.command_args or "",
            )
            return ""

        # Add user message
        self.session.add_message("user", prompt)

        # Process based on mode
        if self.session.mode == CLIMode.CHAT:
            await self.chat_mode.process(prompt)
        else:
            await self.task_mode.process(prompt)

        # Return last assistant message
        for msg in reversed(self.session.messages):
            if msg.role == "assistant":
                return msg.content

        return ""

    async def _process_input(self, user_input: str) -> None:
        """
        Process user input.

        Args:
            user_input: Raw user input
        """
        # Parse input
        parsed = self.parser.parse(user_input)

        if parsed.type == InputType.EMPTY:
            return

        elif parsed.type == InputType.COMMAND:
            await self._handle_command(
                parsed.command_name or "",
                parsed.command_args or "",
            )

        elif parsed.type == InputType.MEMORY:
            await self._handle_memory_shortcut(parsed.content)

        else:
            # Regular message - process with current mode
            await self._handle_message(user_input)

    async def _handle_command(self, cmd_name: str, args: str) -> None:
        """
        Handle a slash command.

        Args:
            cmd_name: Command name (without /)
            args: Command arguments
        """
        command = self.commands.get(cmd_name)

        if command:
            # Create command context
            context = CommandContext(
                session=self.session,
                llm_service=self.llm,
                tool_service=self.tools,
                memory_service=self.memory,
                workflow_service=self.workflow,
                console=self.console,
                config=self.config,
            )

            try:
                result = await command.execute(args, context)

                # If command returns a prompt (custom commands), process it
                if result and command.name not in [
                    "help", "clear", "cost", "model", "mode",
                    "quit", "exit", "sessions", "save", "status", "vim",
                    "config", "memory", "compact",
                ]:
                    await self._handle_message(result)

            except SystemExit:
                raise
            except Exception as e:
                logger.error(f"Command error: {e}")
                self.renderer.render_error(e)
        else:
            self.renderer.render_error(f"Unknown command: /{cmd_name}")
            self.renderer.render_dim("Type /help for available commands")

    async def _handle_memory_shortcut(self, content: str) -> None:
        """
        Handle memory shortcut (#content).

        Args:
            content: Content after #
        """
        # TODO: Implement memory file selection like Claude Code
        # For now, just show a message
        self.renderer.render_warning("Memory shortcuts not yet implemented")
        self.renderer.render_dim("Use /memory add <content> instead")

    async def _handle_message(self, message: str) -> None:
        """
        Handle a regular chat/task message.

        Args:
            message: User message
        """
        # Process based on current mode
        if self.session.mode == CLIMode.CHAT:
            await self.chat_mode.process(message)
        else:
            await self.task_mode.process(message)

        # Auto-save session
        if self.config.auto_save_sessions:
            await self.session_manager.save(self.session)

    def _get_prompt(self) -> str:
        """
        Get prompt string based on current mode.

        Returns:
            Prompt string
        """
        if self.session.mode == CLIMode.CHAT:
            return self.chat_mode.get_prompt()
        else:
            return self.task_mode.get_prompt()

    def _display_welcome(self) -> None:
        """Display welcome message."""
        self.console.print()
        self.console.print("[bold green]Agent Swarm CLI[/bold green]")
        self.console.print(
            f"Mode: {self.session.mode} | "
            f"Model: {self.session.model or 'auto'}"
        )
        self.console.print(
            "Type [bold]/help[/bold] for commands, "
            "[bold]/quit[/bold] to exit"
        )
        self.console.print()

    def _display_goodbye(self) -> None:
        """Display goodbye message."""
        self.console.print()
        self.console.print("[dim]Session saved. Goodbye![/dim]")


def create_repl(
    session: SessionState,
    session_manager: SessionManager,
    config: CLIConfig,
    llm_service: Optional["LLMOrchestrator"] = None,
    tool_service: Optional["ToolOrchestrator"] = None,
    memory_service: Optional["MemoryOrchestrator"] = None,
    workflow_service: Optional["WorkflowOrchestrator"] = None,
) -> REPL:
    """
    Create a REPL instance with all dependencies.

    Args:
        session: Session state
        session_manager: Session manager
        config: CLI configuration
        llm_service: LLM orchestrator
        tool_service: Tool orchestrator
        memory_service: Memory orchestrator
        workflow_service: Workflow orchestrator

    Returns:
        Configured REPL instance
    """
    # Ensure directories exist
    ensure_directories(config)

    # Create command registry
    commands = CommandRegistry()
    load_builtin_commands(commands)
    commands.register(ConfigCommand())
    commands.register(MemoryCommand())
    load_custom_commands(commands, config)

    # Create renderer
    renderer = OutputRenderer(config)

    return REPL(
        session=session,
        session_manager=session_manager,
        config=config,
        commands=commands,
        renderer=renderer,
        llm_service=llm_service,
        tool_service=tool_service,
        memory_service=memory_service,
        workflow_service=workflow_service,
    )
