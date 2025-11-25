"""Unit tests for Redis working memory."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.memory.working import RedisWorkingMemory
from src.models.memory_models import MemoryItem, MemoryType
from src.config.memory_config import MemoryConfig


@pytest.fixture
def mock_config():
    """Create mock memory config."""
    config = MagicMock(spec=MemoryConfig)
    config.redis_url = "redis://localhost:6379/0"
    config.memory_ttl_working = 3600
    config.connection_pool_size = 10
    return config


@pytest.fixture
def memory_item():
    """Create sample memory item."""
    return MemoryItem(
        id="test-memory-1",
        content="Test memory content",
        agent_id="agent-1",
        session_id="session-1",
        memory_type=MemoryType.WORKING,
        importance=0.7,
        tags=["test", "unit"],
        timestamp=datetime.now(),
    )


@pytest.mark.asyncio
class TestRedisWorkingMemory:
    """Test suite for RedisWorkingMemory."""

    async def test_init(self, mock_config):
        """Test initialization."""
        memory = RedisWorkingMemory(config=mock_config)
        assert memory.config == mock_config
        assert memory.ttl == mock_config.memory_ttl_working
        assert memory._redis is None

    async def test_generate_key(self, mock_config, memory_item):
        """Test key generation."""
        memory = RedisWorkingMemory(config=mock_config)
        key = memory._generate_key(memory_item)
        expected = (
            f"working:{memory_item.agent_id}:{memory_item.session_id}:{memory_item.id}"
        )
        assert key == expected

    async def test_generate_key_default_session(self, mock_config):
        """Test key generation with default session."""
        memory = RedisWorkingMemory(config=mock_config)
        item = MemoryItem(
            id="test-1",
            content="Test",
            agent_id="agent-1",
            session_id=None,  # No session ID
            memory_type=MemoryType.WORKING,
        )
        key = memory._generate_key(item)
        assert key == "working:agent-1:default:test-1"

    @patch("src.memory.working.Redis")
    @patch("src.memory.working.ConnectionPool")
    async def test_add_memory(
        self, mock_pool_class, mock_redis_class, mock_config, memory_item
    ):
        """Test adding memory with TTL."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.setex = AsyncMock()
        mock_redis_class.return_value = mock_redis

        memory = RedisWorkingMemory(config=mock_config)

        # Add memory
        result = await memory.add_memory(memory_item)

        # Verify
        assert result == memory_item.id
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == mock_config.memory_ttl_working  # TTL

    @patch("src.memory.working.Redis")
    @patch("src.memory.working.ConnectionPool")
    async def test_get_memory(
        self, mock_pool_class, mock_redis_class, mock_config, memory_item
    ):
        """Test retrieving memory."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.get = AsyncMock(
            return_value=json.dumps(memory_item.model_dump(mode="json"))
        )
        mock_redis.setex = AsyncMock()  # For access count update
        mock_redis_class.return_value = mock_redis

        memory = RedisWorkingMemory(config=mock_config)

        # Get memory
        result = await memory.get_memory(
            memory_id=memory_item.id,
            agent_id=memory_item.agent_id,
            session_id=memory_item.session_id,
        )

        # Verify
        assert result is not None
        assert result.id == memory_item.id
        assert result.content == memory_item.content
        mock_redis.get.assert_called_once()

    @patch("src.memory.working.Redis")
    @patch("src.memory.working.ConnectionPool")
    async def test_get_memory_not_found(
        self, mock_pool_class, mock_redis_class, mock_config
    ):
        """Test retrieving non-existent memory."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis_class.return_value = mock_redis

        memory = RedisWorkingMemory(config=mock_config)

        # Get memory
        result = await memory.get_memory(
            memory_id="nonexistent",
            agent_id="agent-1",
            session_id="session-1",
        )

        # Verify
        assert result is None

    @patch("src.memory.working.Redis")
    @patch("src.memory.working.ConnectionPool")
    async def test_delete_memory(self, mock_pool_class, mock_redis_class, mock_config):
        """Test deleting memory."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.delete = AsyncMock(return_value=1)
        mock_redis_class.return_value = mock_redis

        memory = RedisWorkingMemory(config=mock_config)

        # Delete memory
        result = await memory.delete_memory(
            memory_id="test-1",
            agent_id="agent-1",
            session_id="session-1",
        )

        # Verify
        assert result is True
        mock_redis.delete.assert_called_once()

    @patch("src.memory.working.Redis")
    @patch("src.memory.working.ConnectionPool")
    async def test_clear_all(self, mock_pool_class, mock_redis_class, mock_config):
        """Test clearing all memories."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()

        # Mock scan_iter to return some keys
        async def mock_scan_iter(match, count):
            for key in [
                "working:agent-1:session-1:mem-1",
                "working:agent-1:session-1:mem-2",
            ]:
                yield key

        mock_redis.scan_iter = mock_scan_iter
        mock_redis.delete = AsyncMock()
        mock_redis_class.return_value = mock_redis

        memory = RedisWorkingMemory(config=mock_config)

        # Clear all
        count = await memory.clear_all(agent_id="agent-1", session_id="session-1")

        # Verify
        assert count == 2
        assert mock_redis.delete.call_count == 2

    def test_calculate_relevance(self, mock_config):
        """Test relevance calculation."""
        memory = RedisWorkingMemory(config=mock_config)

        # Exact match
        score = memory._calculate_relevance("test query", "test query")
        assert score == 1.0

        # Contains match
        score = memory._calculate_relevance("query", "this is a query string")
        assert score == 0.8

        # Word overlap
        score = memory._calculate_relevance("test query", "query test")
        assert score > 0.0

        # No match
        score = memory._calculate_relevance("xyz", "abc def")
        assert score == 0.0

    @patch("src.memory.working.Redis")
    @patch("src.memory.working.ConnectionPool")
    async def test_set_get_state(self, mock_pool_class, mock_redis_class, mock_config):
        """Test state storage and retrieval."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.setex = AsyncMock()
        mock_redis.get = AsyncMock(return_value=json.dumps({"key": "value"}))
        mock_redis_class.return_value = mock_redis

        memory = RedisWorkingMemory(config=mock_config)

        # Set state
        await memory.set_state("agent-1", "session-1", {"key": "value"})
        mock_redis.setex.assert_called_once()

        # Get state
        state = await memory.get_state("agent-1", "session-1")
        assert state == {"key": "value"}

    @patch("src.memory.working.Redis")
    @patch("src.memory.working.ConnectionPool")
    async def test_close(self, mock_pool_class, mock_redis_class, mock_config):
        """Test closing connections."""
        # Setup mocks
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_redis_class.return_value = mock_redis

        mock_pool = AsyncMock()
        mock_pool.aclose = AsyncMock()
        mock_pool_class.from_url.return_value = mock_pool

        memory = RedisWorkingMemory(config=mock_config)

        # Initialize connection
        await memory._get_redis()

        # Close
        await memory.close()

        # Verify
        mock_redis.aclose.assert_called_once()
        mock_pool.aclose.assert_called_once()
        assert memory._redis is None
        assert memory._pool is None
