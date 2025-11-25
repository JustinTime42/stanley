"""Tests for CLI modes."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from rich.console import Console

from src.cli.modes.base import BaseMode
from src.cli.modes.chat import ChatMode
from src.cli.modes.task import TaskMode
from src.cli.session.state import SessionState, CLIMode
from src.cli.config.cli_config import CLIConfig
from src.cli.output.renderer import OutputRenderer


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
    config = CLIConfig()
    config.stream_output = False  # Disable streaming for tests
    return config


@pytest.fixture
def console():
    """Create test console."""
    return Console(force_terminal=True, width=80, record=True)


@pytest.fixture
def renderer(config, console):
    """Create test renderer."""
    return OutputRenderer(config, console)


class TestChatMode:
    """Tests for ChatMode."""

    @pytest.fixture
    def chat_mode(self, session, config, renderer, console):
        """Create ChatMode instance."""
        return ChatMode(
            session=session,
            config=config,
            renderer=renderer,
            console=console,
        )

    def test_get_prompt(self, chat_mode):
        """Test getting chat mode prompt."""
        prompt = chat_mode.get_prompt()
        assert "You:" in prompt or "💬" in prompt

    def test_default_system_prompt(self, chat_mode):
        """Test default system prompt."""
        prompt = chat_mode._get_system_prompt()
        assert prompt is not None
        assert "assistant" in prompt.lower() or "helpful" in prompt.lower()

    def test_custom_system_prompt(self, chat_mode):
        """Test custom system prompt override."""
        chat_mode.session.system_prompt = "You are a pirate."
        prompt = chat_mode._get_system_prompt()
        assert prompt == "You are a pirate."

    @pytest.mark.asyncio
    async def test_process_without_llm(self, chat_mode):
        """Test processing message without LLM service."""
        await chat_mode.process("Hello")

        # Message should be added to session
        assert len(chat_mode.session.messages) >= 1

    @pytest.mark.asyncio
    async def test_process_with_mock_llm(self, chat_mode):
        """Test processing message with mock LLM."""
        # Create mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Hello! How can I help?"
        mock_response.input_tokens = 10
        mock_response.output_tokens = 20
        mock_response.total_cost = 0.001

        mock_llm.generate_response = AsyncMock(return_value=mock_response)
        chat_mode.llm = mock_llm

        await chat_mode.process("Hello")

        # Check LLM was called
        mock_llm.generate_response.assert_called_once()

        # Check stats updated
        assert chat_mode.session.total_tokens == 30
        assert chat_mode.session.total_cost == 0.001

    @pytest.mark.asyncio
    async def test_on_enter(self, chat_mode):
        """Test entering chat mode."""
        await chat_mode.on_enter()
        # Should not raise


class TestTaskMode:
    """Tests for TaskMode."""

    @pytest.fixture
    def task_mode(self, session, config, renderer, console):
        """Create TaskMode instance."""
        return TaskMode(
            session=session,
            config=config,
            renderer=renderer,
            console=console,
        )

    def test_get_prompt(self, task_mode):
        """Test getting task mode prompt."""
        prompt = task_mode.get_prompt()
        assert "Task:" in prompt or "🤖" in prompt

    @pytest.mark.asyncio
    async def test_process_without_workflow(self, task_mode):
        """Test processing task without workflow service."""
        await task_mode.process("Create a hello world script")

        # Should handle gracefully without workflow service

    @pytest.mark.asyncio
    async def test_process_with_mock_workflow(self, task_mode):
        """Test processing task with mock workflow."""
        # Create mock workflow
        mock_workflow = Mock()
        mock_execution = Mock()
        mock_execution.status = Mock(value="complete")
        mock_execution.elapsed_time_seconds = 5.0
        mock_execution.total_cost_usd = 0.01
        mock_execution.total_tokens = 1000
        mock_execution.error = None

        mock_workflow.start_workflow = AsyncMock(return_value=mock_execution)
        task_mode.workflow = mock_workflow

        await task_mode.process("Create a hello world script")

        # Check workflow was called
        mock_workflow.start_workflow.assert_called_once()

        # Check session state updated
        assert task_mode.session.active_workflow_id is not None

    @pytest.mark.asyncio
    async def test_on_enter(self, task_mode):
        """Test entering task mode."""
        await task_mode.on_enter()
        # Should not raise

    @pytest.mark.asyncio
    async def test_on_exit_with_active_workflow(self, task_mode):
        """Test exiting task mode with active workflow."""
        task_mode.session.active_workflow_id = "wf_123"
        task_mode.session.workflow_status = "running"

        await task_mode.on_exit()
        # Should warn about active workflow


class TestModeProperties:
    """Tests for mode properties."""

    def test_chat_mode_name(self, session, config, renderer, console):
        """Test chat mode name."""
        mode = ChatMode(session, config, renderer, console)
        assert mode.name == "chat"

    def test_task_mode_name(self, session, config, renderer, console):
        """Test task mode name."""
        mode = TaskMode(session, config, renderer, console)
        assert mode.name == "task"

    def test_chat_mode_description(self, session, config, renderer, console):
        """Test chat mode description."""
        mode = ChatMode(session, config, renderer, console)
        assert "conversation" in mode.description.lower()

    def test_task_mode_description(self, session, config, renderer, console):
        """Test task mode description."""
        mode = TaskMode(session, config, renderer, console)
        assert "workflow" in mode.description.lower() or "agent" in mode.description.lower()
