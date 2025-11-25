"""Tests for CLI REPL."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from rich.console import Console

from src.cli.repl import REPL, create_repl
from src.cli.session.state import SessionState, CLIMode
from src.cli.session.manager import SessionManager
from src.cli.config.cli_config import CLIConfig
from src.cli.commands.base import CommandRegistry
from src.cli.commands.builtin import load_builtin_commands
from src.cli.output.renderer import OutputRenderer
from src.cli.input.parser import InputParser, InputType


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
def manager(config, tmp_path):
    """Create session manager."""
    config.sessions_dir = str(tmp_path / "sessions")
    return SessionManager(config)


@pytest.fixture
def commands():
    """Create command registry."""
    registry = CommandRegistry()
    load_builtin_commands(registry)
    return registry


@pytest.fixture
def renderer(config):
    """Create renderer."""
    console = Console(force_terminal=True, width=80, record=True)
    return OutputRenderer(config, console)


@pytest.fixture
def repl(session, manager, config, commands, renderer):
    """Create REPL instance with mocked prompt_toolkit."""
    mock_prompt_session = Mock()
    mock_prompt_session.prompt_async = AsyncMock(return_value="")

    repl_instance = REPL(
        session=session,
        session_manager=manager,
        config=config,
        commands=commands,
        renderer=renderer,
    )
    # Set the private attribute to avoid triggering PromptSession creation
    repl_instance._prompt_session = mock_prompt_session
    return repl_instance


class TestInputParser:
    """Tests for InputParser."""

    def test_parse_empty(self):
        """Test parsing empty input."""
        parser = InputParser()
        result = parser.parse("")
        assert result.type == InputType.EMPTY

    def test_parse_command(self):
        """Test parsing slash command."""
        parser = InputParser()
        result = parser.parse("/help")

        assert result.type == InputType.COMMAND
        assert result.command_name == "help"
        assert result.command_args == ""

    def test_parse_command_with_args(self):
        """Test parsing command with arguments."""
        parser = InputParser()
        result = parser.parse("/model gpt-4")

        assert result.type == InputType.COMMAND
        assert result.command_name == "model"
        assert result.command_args == "gpt-4"

    def test_parse_namespaced_command(self):
        """Test parsing namespaced command."""
        parser = InputParser()
        result = parser.parse("/frontend:component Button")

        assert result.type == InputType.COMMAND
        assert result.command_name == "frontend:component"
        assert result.command_args == "Button"

    def test_parse_memory_shortcut(self):
        """Test parsing memory shortcut."""
        parser = InputParser()
        result = parser.parse("# Remember this")

        assert result.type == InputType.MEMORY
        assert result.content == "Remember this"

    def test_parse_message(self):
        """Test parsing regular message."""
        parser = InputParser()
        result = parser.parse("Hello, how are you?")

        assert result.type == InputType.MESSAGE
        assert result.content == "Hello, how are you?"

    def test_parse_message_with_file_refs(self):
        """Test parsing message with file references."""
        parser = InputParser()
        result = parser.parse("Check @src/main.py and @README.md")

        assert result.type == InputType.MESSAGE
        assert result.file_refs is not None
        assert "src/main.py" in result.file_refs
        assert "README.md" in result.file_refs

    def test_needs_continuation_backslash(self):
        """Test multiline continuation with backslash."""
        parser = InputParser()
        assert parser.needs_continuation("Hello \\") is True
        assert parser.needs_continuation("Hello") is False

    def test_needs_continuation_code_block(self):
        """Test multiline continuation with code block."""
        parser = InputParser()
        assert parser.needs_continuation("```python") is True
        assert parser.needs_continuation("```python\ncode\n```") is False


class TestREPL:
    """Tests for REPL class."""

    def test_create_repl(self, session, manager, config):
        """Test creating REPL with create_repl function."""
        with patch('src.cli.repl.PromptSession') as mock_session_class:
            mock_prompt_session = Mock()
            mock_session_class.return_value = mock_prompt_session

            repl = create_repl(session, manager, config)
            assert repl is not None
            assert repl.session == session

    def test_get_prompt_chat_mode(self, repl):
        """Test getting prompt in chat mode."""
        repl.session.mode = CLIMode.CHAT
        prompt = repl._get_prompt()
        assert "You:" in prompt or "💬" in prompt

    def test_get_prompt_task_mode(self, repl):
        """Test getting prompt in task mode."""
        repl.session.mode = CLIMode.TASK
        prompt = repl._get_prompt()
        assert "Task:" in prompt or "🤖" in prompt

    @pytest.mark.asyncio
    async def test_handle_command(self, repl):
        """Test handling a command."""
        # /clear should clear messages
        repl.session.add_message("user", "Hello")
        assert len(repl.session.messages) == 1

        await repl._handle_command("clear", "")

        assert len(repl.session.messages) == 0

    @pytest.mark.asyncio
    async def test_handle_unknown_command(self, repl):
        """Test handling unknown command."""
        # Should not raise
        await repl._handle_command("nonexistent_command", "")

    @pytest.mark.asyncio
    async def test_process_input_command(self, repl):
        """Test processing command input."""
        repl.session.add_message("user", "Hello")

        await repl._process_input("/clear")

        assert len(repl.session.messages) == 0

    @pytest.mark.asyncio
    async def test_process_once(self, repl):
        """Test one-shot processing."""
        # Without LLM service, should handle gracefully
        result = await repl.process_once("Hello")
        # May be empty without LLM
        assert isinstance(result, str)


class TestREPLWithMocks:
    """Tests for REPL with mocked services."""

    @pytest.mark.asyncio
    async def test_handle_message_chat_mode(self, repl):
        """Test handling message in chat mode."""
        repl.session.mode = CLIMode.CHAT

        # Mock chat mode process
        repl.chat_mode.process = AsyncMock()

        await repl._handle_message("Hello")

        repl.chat_mode.process.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_handle_message_task_mode(self, repl):
        """Test handling message in task mode."""
        repl.session.mode = CLIMode.TASK

        # Mock task mode process
        repl.task_mode.process = AsyncMock()

        await repl._handle_message("Create a script")

        repl.task_mode.process.assert_called_once_with("Create a script")

    @pytest.mark.asyncio
    async def test_auto_save(self, repl):
        """Test auto-save functionality."""
        repl.config.auto_save_sessions = True

        # Mock session manager
        repl.session_manager.save = AsyncMock()

        # Mock mode to avoid actual processing
        repl.chat_mode.process = AsyncMock()

        await repl._handle_message("Hello")

        repl.session_manager.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_auto_save_when_disabled(self, repl):
        """Test auto-save disabled."""
        repl.config.auto_save_sessions = False

        # Mock session manager
        repl.session_manager.save = AsyncMock()

        # Mock mode
        repl.chat_mode.process = AsyncMock()

        await repl._handle_message("Hello")

        repl.session_manager.save.assert_not_called()
