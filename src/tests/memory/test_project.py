"""Unit tests for Qdrant project memory."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from src.memory.project import QdrantProjectMemory
from src.models.memory_models import MemoryItem, MemorySearchRequest, MemoryType
from src.config.memory_config import MemoryConfig


@pytest.fixture
def mock_config():
    """Create mock memory config."""
    config = MagicMock(spec=MemoryConfig)
    config.qdrant_url = "http://localhost:6333"
    config.qdrant_api_key = None
    config.project_collection_name = "project_memory"
    config.vector_size = 1536
    return config


@pytest.fixture
def memory_item():
    """Create sample memory item with embedding."""
    return MemoryItem(
        id="test-memory-1",
        content="Test memory content for project",
        embedding=[0.1] * 1536,  # Mock embedding
        agent_id="agent-1",
        project_id="project-1",
        memory_type=MemoryType.PROJECT,
        importance=0.7,
        tags=["test", "project"],
        timestamp=datetime.now(),
    )


@pytest.fixture
def mock_embedding_function():
    """Create mock embedding function."""

    async def embed(text: str) -> List[float]:
        return [0.1] * 1536

    return embed


@pytest.mark.asyncio
class TestQdrantProjectMemory:
    """Test suite for QdrantProjectMemory."""

    async def test_init(self, mock_config, mock_embedding_function):
        """Test initialization."""
        memory = QdrantProjectMemory(
            config=mock_config,
            embedding_function=mock_embedding_function,
        )
        assert memory.config == mock_config
        assert memory.collection_name == mock_config.project_collection_name
        assert memory.embedding_function == mock_embedding_function
        assert memory._client is None

    @patch("src.memory.project.AsyncQdrantClient")
    async def test_get_client_creates_collection(
        self, mock_client_class, mock_config, mock_embedding_function
    ):
        """Test client initialization creates collection if needed."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = []  # No existing collections
        mock_client.get_collections = AsyncMock(return_value=mock_collections)
        mock_client.create_collection = AsyncMock()
        mock_client_class.return_value = mock_client

        memory = QdrantProjectMemory(
            config=mock_config,
            embedding_function=mock_embedding_function,
        )

        # Get client (should create collection)
        client = await memory._get_client()

        # Verify
        assert client == mock_client
        mock_client.create_collection.assert_called_once()

    @patch("src.memory.project.AsyncQdrantClient")
    async def test_add_memory(
        self, mock_client_class, mock_config, mock_embedding_function, memory_item
    ):
        """Test adding memory with embedding."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="project_memory")]
        mock_client.get_collections = AsyncMock(return_value=mock_collections)
        mock_client.upsert = AsyncMock()
        mock_client_class.return_value = mock_client

        memory = QdrantProjectMemory(
            config=mock_config,
            embedding_function=mock_embedding_function,
        )

        # Add memory
        result = await memory.add_memory(memory_item)

        # Verify
        assert result == memory_item.id
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "project_memory"
        assert len(call_args[1]["points"]) == 1

    @patch("src.memory.project.AsyncQdrantClient")
    async def test_add_memory_generates_embedding(
        self, mock_client_class, mock_config, mock_embedding_function
    ):
        """Test adding memory generates embedding if missing."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="project_memory")]
        mock_client.get_collections = AsyncMock(return_value=mock_collections)
        mock_client.upsert = AsyncMock()
        mock_client_class.return_value = mock_client

        memory = QdrantProjectMemory(
            config=mock_config,
            embedding_function=mock_embedding_function,
        )

        # Create item without embedding
        item = MemoryItem(
            id="test-1",
            content="Test content",
            embedding=None,  # No embedding
            agent_id="agent-1",
            memory_type=MemoryType.PROJECT,
        )

        # Add memory
        result = await memory.add_memory(item)

        # Verify
        assert result == item.id
        assert item.embedding is not None  # Embedding was generated

    @patch("src.memory.project.AsyncQdrantClient")
    async def test_add_memories_batch(
        self, mock_client_class, mock_config, mock_embedding_function
    ):
        """Test batch adding memories."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="project_memory")]
        mock_client.get_collections = AsyncMock(return_value=mock_collections)
        mock_client.upsert = AsyncMock()
        mock_client_class.return_value = mock_client

        memory = QdrantProjectMemory(
            config=mock_config,
            embedding_function=mock_embedding_function,
        )

        # Create multiple items
        items = [
            MemoryItem(
                id=f"test-{i}",
                content=f"Test content {i}",
                embedding=[0.1] * 1536,
                agent_id="agent-1",
                memory_type=MemoryType.PROJECT,
            )
            for i in range(5)
        ]

        # Batch add
        result_ids = await memory.add_memories(items)

        # Verify
        assert len(result_ids) == 5
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert len(call_args[1]["points"]) == 5

    @patch("src.memory.project.AsyncQdrantClient")
    async def test_get_memory(
        self, mock_client_class, mock_config, mock_embedding_function, memory_item
    ):
        """Test retrieving memory by ID."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="project_memory")]
        mock_client.get_collections = AsyncMock(return_value=mock_collections)

        # Mock retrieved point
        mock_point = MagicMock()
        mock_point.id = memory_item.id
        mock_point.vector = memory_item.embedding
        mock_point.payload = {
            "content": memory_item.content,
            "agent_id": memory_item.agent_id,
            "project_id": memory_item.project_id,
            "memory_type": memory_item.memory_type.value,
            "importance": memory_item.importance,
            "access_count": memory_item.access_count,
            "tags": memory_item.tags,
            "timestamp": memory_item.timestamp.isoformat(),
        }

        mock_client.retrieve = AsyncMock(return_value=[mock_point])
        mock_client_class.return_value = mock_client

        memory = QdrantProjectMemory(
            config=mock_config,
            embedding_function=mock_embedding_function,
        )

        # Get memory
        result = await memory.get_memory(memory_item.id)

        # Verify
        assert result is not None
        assert result.id == memory_item.id
        assert result.content == memory_item.content

    @patch("src.memory.project.AsyncQdrantClient")
    async def test_search_memories(
        self, mock_client_class, mock_config, mock_embedding_function
    ):
        """Test vector search."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="project_memory")]
        mock_client.get_collections = AsyncMock(return_value=mock_collections)

        # Mock search results
        mock_result = MagicMock()
        mock_result.id = "test-1"
        mock_result.score = 0.95
        mock_result.vector = [0.1] * 1536
        mock_result.payload = {
            "content": "Test content",
            "agent_id": "agent-1",
            "project_id": "project-1",
            "memory_type": "project",
            "importance": 0.7,
            "access_count": 0,
            "tags": ["test"],
            "timestamp": datetime.now().isoformat(),
        }

        # Mock query_points to return QueryResponse-like object
        mock_query_response = MagicMock()
        mock_query_response.points = [mock_result]
        mock_client.query_points = AsyncMock(return_value=mock_query_response)
        mock_client_class.return_value = mock_client

        memory = QdrantProjectMemory(
            config=mock_config,
            embedding_function=mock_embedding_function,
        )

        # Search
        request = MemorySearchRequest(
            query="test query",
            k=5,
        )
        results = await memory.search_memories(request)

        # Verify
        assert len(results) == 1
        assert results[0].score == 0.95
        assert results[0].source == "project"

    @patch("src.memory.project.AsyncQdrantClient")
    async def test_delete_memory(
        self, mock_client_class, mock_config, mock_embedding_function
    ):
        """Test deleting memory."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="project_memory")]
        mock_client.get_collections = AsyncMock(return_value=mock_collections)
        mock_client.delete = AsyncMock()
        mock_client_class.return_value = mock_client

        memory = QdrantProjectMemory(
            config=mock_config,
            embedding_function=mock_embedding_function,
        )

        # Delete
        result = await memory.delete_memory("test-1")

        # Verify
        assert result is True
        mock_client.delete.assert_called_once()

    @patch("src.memory.project.AsyncQdrantClient")
    async def test_clear_all(
        self, mock_client_class, mock_config, mock_embedding_function
    ):
        """Test clearing all memories."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "project_memory"
        mock_collections.collections = [mock_collection]
        mock_client.get_collections = AsyncMock(return_value=mock_collections)

        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 100
        mock_client.get_collection = AsyncMock(return_value=mock_collection_info)
        mock_client.delete_collection = AsyncMock()
        mock_client.create_collection = AsyncMock()
        mock_client_class.return_value = mock_client

        memory = QdrantProjectMemory(
            config=mock_config,
            embedding_function=mock_embedding_function,
        )

        # Clear all
        count = await memory.clear_all()

        # Verify
        assert count == 100
        mock_client.delete_collection.assert_called_once()
        mock_client.create_collection.assert_called_once()

    @patch("src.memory.project.AsyncQdrantClient")
    async def test_close(self, mock_client_class, mock_config, mock_embedding_function):
        """Test closing client."""
        # Setup mocks
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="project_memory")]
        mock_client.get_collections = AsyncMock(return_value=mock_collections)
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        memory = QdrantProjectMemory(
            config=mock_config,
            embedding_function=mock_embedding_function,
        )

        # Initialize client
        await memory._get_client()

        # Close
        await memory.close()

        # Verify
        mock_client.close.assert_called_once()
        assert memory._client is None
