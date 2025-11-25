"""Tests for CLI session management."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from src.cli.session.state import SessionState, Message, CLIMode
from src.cli.session.manager import SessionManager
from src.cli.config.cli_config import CLIConfig


class TestSessionState:
    """Tests for SessionState model."""

    def test_create_session(self):
        """Test creating a new session."""
        session = SessionState(
            session_id="test_123",
            working_directory="/tmp/test",
        )

        assert session.session_id == "test_123"
        assert session.working_directory == "/tmp/test"
        assert session.mode == CLIMode.CHAT
        assert len(session.messages) == 0
        assert session.turn_count == 0

    def test_add_message(self):
        """Test adding messages to session."""
        session = SessionState(
            session_id="test_123",
            working_directory="/tmp/test",
        )

        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")

        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Hello"
        assert session.messages[1].role == "assistant"
        assert session.turn_count == 2

    def test_clear_messages(self):
        """Test clearing messages."""
        session = SessionState(
            session_id="test_123",
            working_directory="/tmp/test",
        )

        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi!")
        session.clear_messages()

        assert len(session.messages) == 0
        assert session.turn_count == 0

    def test_get_messages_for_llm(self):
        """Test getting messages in LLM format."""
        session = SessionState(
            session_id="test_123",
            working_directory="/tmp/test",
        )

        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi!")

        llm_messages = session.get_messages_for_llm()

        assert len(llm_messages) == 2
        assert llm_messages[0] == {"role": "user", "content": "Hello"}
        assert llm_messages[1] == {"role": "assistant", "content": "Hi!"}

    def test_update_stats(self):
        """Test updating session statistics."""
        session = SessionState(
            session_id="test_123",
            working_directory="/tmp/test",
        )

        session.update_stats(input_tokens=100, output_tokens=50, cost=0.001)

        assert session.total_tokens == 150
        assert session.total_cost == 0.001

        session.update_stats(input_tokens=50, output_tokens=25, cost=0.0005)

        assert session.total_tokens == 225
        assert session.total_cost == 0.0015

    def test_session_serialization(self):
        """Test session can be serialized to JSON."""
        session = SessionState(
            session_id="test_123",
            working_directory="/tmp/test",
        )
        session.add_message("user", "Hello")

        json_str = session.model_dump_json()
        assert "test_123" in json_str
        assert "Hello" in json_str

        # Deserialize
        loaded = SessionState.model_validate_json(json_str)
        assert loaded.session_id == "test_123"
        assert len(loaded.messages) == 1

    def test_to_summary(self):
        """Test session summary."""
        session = SessionState(
            session_id="test_123",
            working_directory="/tmp/test",
            name="My Session",
        )
        session.add_message("user", "Hello")
        session.total_tokens = 100
        session.total_cost = 0.01

        summary = session.to_summary()

        assert summary["session_id"] == "test_123"
        assert summary["name"] == "My Session"
        assert summary["turn_count"] == 1
        assert summary["total_tokens"] == 100


class TestMessage:
    """Tests for Message model."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)

    def test_message_to_dict(self):
        """Test message to dict conversion."""
        msg = Message(role="assistant", content="Hi there!")

        d = msg.to_dict()

        assert d == {"role": "assistant", "content": "Hi there!"}


class TestSessionManager:
    """Tests for SessionManager."""

    @pytest.fixture
    def temp_config(self, tmp_path):
        """Create config with temp directory."""
        config = CLIConfig()
        config.sessions_dir = str(tmp_path / "sessions")
        return config

    @pytest.fixture
    def manager(self, temp_config):
        """Create session manager."""
        return SessionManager(temp_config)

    def test_generate_id(self, manager):
        """Test session ID generation."""
        id1 = manager.generate_id()
        id2 = manager.generate_id()

        assert id1 != id2
        assert "_" in id1  # Format: timestamp_uuid

    @pytest.mark.asyncio
    async def test_save_and_load(self, manager):
        """Test saving and loading sessions."""
        session = SessionState(
            session_id="test_save",
            working_directory="/tmp",
        )
        session.add_message("user", "Hello")

        await manager.save(session)
        loaded = await manager.load("test_save")

        assert loaded is not None
        assert loaded.session_id == "test_save"
        assert len(loaded.messages) == 1

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, manager):
        """Test loading nonexistent session."""
        loaded = await manager.load("nonexistent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_load_latest(self, manager):
        """Test loading latest session."""
        # Create two sessions
        session1 = SessionState(
            session_id="older",
            working_directory="/tmp",
        )
        await manager.save(session1)

        session2 = SessionState(
            session_id="newer",
            working_directory="/tmp",
        )
        await manager.save(session2)

        latest = await manager.load_latest()

        assert latest is not None
        assert latest.session_id == "newer"

    @pytest.mark.asyncio
    async def test_list_sessions(self, manager):
        """Test listing sessions."""
        # Create sessions
        for i in range(3):
            session = SessionState(
                session_id=f"session_{i}",
                working_directory="/tmp",
            )
            await manager.save(session)

        sessions = await manager.list_sessions(limit=10)

        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_delete_session(self, manager):
        """Test deleting a session."""
        session = SessionState(
            session_id="to_delete",
            working_directory="/tmp",
        )
        await manager.save(session)

        deleted = await manager.delete("to_delete")
        assert deleted is True

        loaded = await manager.load("to_delete")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_rename_session(self, manager):
        """Test renaming a session."""
        session = SessionState(
            session_id="to_rename",
            working_directory="/tmp",
        )
        await manager.save(session)

        renamed = await manager.rename("to_rename", "New Name")
        assert renamed is True

        # Verify name was updated
        sessions = await manager.list_sessions()
        found = [s for s in sessions if s["session_id"] == "to_rename"]
        assert len(found) == 1
        assert found[0]["name"] == "New Name"
