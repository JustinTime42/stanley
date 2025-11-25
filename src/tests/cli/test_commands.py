"""Tests for CLI commands."""

import pytest
from unittest.mock import Mock, AsyncMock
from rich.console import Console

from src.cli.commands.base import (
    BaseCommand,
    CommandContext,
    CommandRegistry,
    CustomCommand,
)
from src.cli.commands.builtin import (
    HelpCommand,
    ClearCommand,
    CostCommand,
    ModelCommand,
    ModeCommand,
    StatusCommand,
    load_builtin_commands,
)
from src.cli.session.state import SessionState, CLIMode
from src.cli.config.cli_config import CLIConfig


@pytest.fixture
def session():
    """Create test session."""
    return SessionState(
        session_id="test_session",
        working_directory="/tmp/test",
    )


@pytest.fixture
def config():
    """Create test config."""
    return CLIConfig()


@pytest.fixture
def console():
    """Create test console."""
    return Console(force_terminal=True, width=80)


@pytest.fixture
def context(session, config, console):
    """Create test command context."""
    return CommandContext(
        session=session,
        config=config,
        console=console,
    )


class TestCommandRegistry:
    """Tests for CommandRegistry."""

    def test_register_command(self):
        """Test registering a command."""
        registry = CommandRegistry()

        class TestCommand(BaseCommand):
            name = "test"
            description = "Test command"
            aliases = ["t"]

            async def execute(self, args, context):
                return "executed"

        registry.register(TestCommand())

        assert registry.get("test") is not None
        assert registry.get("t") is not None  # Alias

    def test_get_nonexistent(self):
        """Test getting nonexistent command."""
        registry = CommandRegistry()
        assert registry.get("nonexistent") is None

    def test_list_commands(self):
        """Test listing commands."""
        registry = CommandRegistry()

        class TestCommand1(BaseCommand):
            name = "test1"

            async def execute(self, args, context):
                pass

        class TestCommand2(BaseCommand):
            name = "test2"

            async def execute(self, args, context):
                pass

        registry.register(TestCommand1())
        registry.register(TestCommand2())

        commands = registry.list_commands()
        assert len(commands) == 2

    def test_get_all_names(self):
        """Test getting all names and aliases."""
        registry = CommandRegistry()

        class TestCommand(BaseCommand):
            name = "test"
            aliases = ["t", "tst"]

            async def execute(self, args, context):
                pass

        registry.register(TestCommand())

        names = registry.get_all_names()
        assert "test" in names
        assert "t" in names
        assert "tst" in names


class TestBuiltinCommands:
    """Tests for built-in commands."""

    def test_load_builtin_commands(self):
        """Test loading all builtin commands."""
        registry = CommandRegistry()
        load_builtin_commands(registry)

        # Check core commands exist
        assert registry.get("help") is not None
        assert registry.get("clear") is not None
        assert registry.get("cost") is not None
        assert registry.get("model") is not None
        assert registry.get("mode") is not None
        assert registry.get("quit") is not None
        assert registry.get("status") is not None

    @pytest.mark.asyncio
    async def test_clear_command(self, context):
        """Test /clear command."""
        context.session.add_message("user", "Hello")
        context.session.add_message("assistant", "Hi!")

        cmd = ClearCommand()
        await cmd.execute("", context)

        assert len(context.session.messages) == 0

    @pytest.mark.asyncio
    async def test_cost_command(self, context):
        """Test /cost command."""
        context.session.total_tokens = 1000
        context.session.total_cost = 0.01

        cmd = CostCommand()
        result = await cmd.execute("", context)

        # Command prints to console, no return value
        assert result is None

    @pytest.mark.asyncio
    async def test_model_command_show(self, context):
        """Test /model command (show current)."""
        context.session.model = "gpt-4"

        cmd = ModelCommand()
        await cmd.execute("", context)

        # Verify model wasn't changed
        assert context.session.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_model_command_set(self, context):
        """Test /model command (set model)."""
        cmd = ModelCommand()
        await cmd.execute("claude-3-opus", context)

        assert context.session.model == "claude-3-opus"

    @pytest.mark.asyncio
    async def test_mode_command_show(self, context):
        """Test /mode command (show current)."""
        context.session.mode = CLIMode.CHAT

        cmd = ModeCommand()
        await cmd.execute("", context)

        assert context.session.mode == CLIMode.CHAT

    @pytest.mark.asyncio
    async def test_mode_command_switch(self, context):
        """Test /mode command (switch mode)."""
        context.session.mode = CLIMode.CHAT

        cmd = ModeCommand()
        await cmd.execute("task", context)

        assert context.session.mode == CLIMode.TASK

    @pytest.mark.asyncio
    async def test_status_command(self, context):
        """Test /status command."""
        context.session.model = "test-model"
        context.session.turn_count = 5

        cmd = StatusCommand()
        result = await cmd.execute("", context)

        assert result is None  # Prints to console


class TestCustomCommand:
    """Tests for CustomCommand."""

    def test_create_custom_command(self):
        """Test creating a custom command."""
        cmd = CustomCommand(
            name="test",
            content="Test content with $ARGUMENTS",
            description="Test description (project)",  # Note: loader adds scope to description
            scope="project",
        )

        assert cmd.name == "test"
        assert cmd.content == "Test content with $ARGUMENTS"
        assert cmd.scope == "project"
        assert "Test description" in cmd.description

    @pytest.mark.asyncio
    async def test_execute_with_arguments(self, context):
        """Test executing custom command with arguments."""
        cmd = CustomCommand(
            name="explain",
            content="Explain the following: $ARGUMENTS",
        )

        result = await cmd.execute("Python classes", context)

        assert result == "Explain the following: Python classes"

    @pytest.mark.asyncio
    async def test_execute_without_arguments(self, context):
        """Test executing custom command without arguments."""
        cmd = CustomCommand(
            name="help",
            content="Provide helpful information",
        )

        result = await cmd.execute("", context)

        assert result == "Provide helpful information"

    def test_get_help(self):
        """Test getting command help."""
        cmd = CustomCommand(
            name="test",
            content="Test content",
            description="A test command",
            argument_hint="<arg>",
        )

        help_text = cmd.get_help()

        assert "/test" in help_text
        assert "<arg>" in help_text
        assert "A test command" in help_text
