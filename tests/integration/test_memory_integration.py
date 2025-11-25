"""Integration tests for hierarchical memory system with live services.

These tests require Docker services to be running:
    docker-compose -f docker/docker-compose.yml up -d
"""

import pytest
import asyncio
import time
import uuid
from typing import List
from datetime import datetime

from src.config.memory_config import MemoryConfig
from src.memory.working import RedisWorkingMemory
from src.memory.project import QdrantProjectMemory
from src.models.memory_models import MemoryItem, MemorySearchRequest, MemoryType
from src.services.memory_service import MemoryOrchestrator


# Simple embedding function for testing
async def mock_embedding_function(text: str) -> List[float]:
    """Generate deterministic test embeddings."""
    # Simple hash-based embedding for testing
    hash_val = hash(text) % 1000
    return [float(hash_val % 100) / 100.0] * 1536


@pytest.fixture
def config():
    """Create test configuration."""
    return MemoryConfig(
        redis_url="redis://localhost:6379/0",
        qdrant_url="http://localhost:6333",
        project_collection_name="test_project_memory",
        global_collection_name="test_global_memory",
        vector_size=1536,
    )


@pytest.mark.asyncio
class TestRedisWorkingMemoryIntegration:
    """Integration tests for Redis working memory."""

    async def test_redis_connection(self, config):
        """Test Redis connection and basic operations."""
        memory = RedisWorkingMemory(config=config)

        # Create test memory item
        item = MemoryItem(
            id="test-redis-1",
            content="Integration test memory",
            agent_id="test-agent",
            session_id="test-session",
            memory_type=MemoryType.WORKING,
            importance=0.8,
            tags=["integration", "test"],
        )

        # Add memory
        memory_id = await memory.add_memory(item)
        assert memory_id == item.id

        # Retrieve memory
        retrieved = await memory.get_memory(
            memory_id=item.id,
            agent_id=item.agent_id,
            session_id=item.session_id,
        )
        assert retrieved is not None
        assert retrieved.content == item.content
        assert retrieved.agent_id == item.agent_id

        # Clean up
        await memory.delete_memory(item.id, item.agent_id, item.session_id)
        await memory.close()

    async def test_redis_ttl(self, config):
        """Test TTL functionality."""
        memory = RedisWorkingMemory(config=config)

        item = MemoryItem(
            id="test-ttl-1",
            content="TTL test",
            agent_id="test-agent",
            session_id="test-session",
            memory_type=MemoryType.WORKING,
        )

        # Add with short TTL
        await memory.add_memory(item, ttl=2)

        # Verify it exists
        retrieved = await memory.get_memory(item.id, item.agent_id, item.session_id)
        assert retrieved is not None

        # Wait for expiration
        await asyncio.sleep(3)

        # Verify it's gone
        expired = await memory.get_memory(item.id, item.agent_id, item.session_id)
        assert expired is None

        await memory.close()

    async def test_redis_performance(self, config):
        """Test Redis performance meets <100ms target."""
        memory = RedisWorkingMemory(config=config)

        item = MemoryItem(
            id="test-perf-1",
            content="Performance test",
            agent_id="test-agent",
            session_id="test-session",
            memory_type=MemoryType.WORKING,
        )

        # Test write performance
        start = time.time()
        await memory.add_memory(item)
        write_time = (time.time() - start) * 1000  # Convert to ms

        # Test read performance
        start = time.time()
        await memory.get_memory(item.id, item.agent_id, item.session_id)
        read_time = (time.time() - start) * 1000

        # Verify performance targets
        assert write_time < 100, f"Write took {write_time:.2f}ms (target: <100ms)"
        assert read_time < 100, f"Read took {read_time:.2f}ms (target: <100ms)"

        # Clean up
        await memory.delete_memory(item.id, item.agent_id, item.session_id)
        await memory.close()


@pytest.mark.asyncio
class TestQdrantProjectMemoryIntegration:
    """Integration tests for Qdrant project memory."""

    async def test_qdrant_connection(self, config):
        """Test Qdrant connection and collection creation."""
        memory = QdrantProjectMemory(
            config=config,
            embedding_function=mock_embedding_function,
        )

        # Create test memory item with UUID
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content="Qdrant integration test",
            agent_id="test-agent",
            project_id="test-project",
            memory_type=MemoryType.PROJECT,
            importance=0.9,
            tags=["qdrant", "integration"],
        )

        # Add memory (will auto-generate embedding)
        memory_id = await memory.add_memory(item)
        assert memory_id == item.id

        # Small delay for indexing
        await asyncio.sleep(0.5)

        # Retrieve memory
        retrieved = await memory.get_memory(item.id)
        assert retrieved is not None
        assert retrieved.content == item.content
        assert retrieved.embedding is not None

        # Clean up
        await memory.delete_memory(item.id)
        await memory.close()

    async def test_qdrant_vector_search(self, config):
        """Test vector similarity search."""
        memory = QdrantProjectMemory(
            config=config,
            embedding_function=mock_embedding_function,
        )

        # Add multiple memories with UUIDs
        items = [
            MemoryItem(
                id=str(uuid.uuid4()),
                content=f"Test content number {i}",
                agent_id="test-agent",
                project_id="test-project",
                memory_type=MemoryType.PROJECT,
                importance=0.7,
            )
            for i in range(5)
        ]

        for item in items:
            await memory.add_memory(item)

        # Wait for indexing
        await asyncio.sleep(1)

        # Search
        request = MemorySearchRequest(
            query="Test content",
            k=3,
            memory_types=[MemoryType.PROJECT],
        )

        results = await memory.search_memories(request)
        assert len(results) > 0
        assert len(results) <= 3

        # Clean up
        for item in items:
            await memory.delete_memory(item.id)
        await memory.close()

    async def test_qdrant_batch_operations(self, config):
        """Test batch insert performance."""
        memory = QdrantProjectMemory(
            config=config,
            embedding_function=mock_embedding_function,
        )

        # Create batch of memories with UUIDs
        items = [
            MemoryItem(
                id=str(uuid.uuid4()),
                content=f"Batch test item {i}",
                agent_id="test-agent",
                project_id="test-project",
                memory_type=MemoryType.PROJECT,
            )
            for i in range(20)
        ]

        # Insert individually (batch method not yet implemented)
        start = time.time()
        for item in items:
            await memory.add_memory(item)
        batch_time = time.time() - start

        print(f"Insert of 20 items took {batch_time:.2f}s")
        assert batch_time < 10.0, f"Insert too slow: {batch_time:.2f}s"

        # Clean up
        await memory.clear_all()
        await memory.close()


@pytest.mark.asyncio
class TestMemoryOrchestratorIntegration:
    """Integration tests for the full memory orchestrator."""

    async def test_orchestrator_initialization(self, config):
        """Test orchestrator with all memory tiers."""
        orchestrator = MemoryOrchestrator(
            config=config,
            embedding_function=mock_embedding_function,
        )

        # Verify all tiers initialized
        assert orchestrator.working_memory is not None
        assert orchestrator.project_memory is not None
        assert orchestrator.global_memory is not None

        await orchestrator.close()

    async def test_orchestrator_store_and_retrieve(self, config):
        """Test storing and retrieving across memory tiers."""
        orchestrator = MemoryOrchestrator(
            config=config,
            embedding_function=mock_embedding_function,
        )

        # Store in working memory
        working_id = await orchestrator.store_memory(
            content="Working memory test",
            agent_id="test-agent",
            memory_type=MemoryType.WORKING,
            session_id="test-session",
            importance=0.5,
        )
        assert working_id is not None

        # Store in project memory
        project_id = await orchestrator.store_memory(
            content="Project memory test",
            agent_id="test-agent",
            memory_type=MemoryType.PROJECT,
            project_id="test-project",
            importance=0.8,
            tags=["project", "test"],
        )
        assert project_id is not None

        # Wait for indexing
        await asyncio.sleep(0.5)

        # Retrieve from working memory
        working_mem = await orchestrator.retrieve_memory(
            memory_id=working_id,
            memory_type=MemoryType.WORKING,
            agent_id="test-agent",
            session_id="test-session",
        )
        assert working_mem is not None
        assert working_mem.content == "Working memory test"

        # Search across project memory
        results = await orchestrator.retrieve_relevant_memories(
            query="Project memory",
            memory_types=[MemoryType.PROJECT],
            k=5,
            use_hybrid=False,  # Just vector search for simplicity
        )
        assert len(results) > 0

        # Clean up
        await orchestrator.close()

    async def test_orchestrator_performance_mixed_operations(self, config):
        """Test performance with mixed read/write operations."""
        orchestrator = MemoryOrchestrator(
            config=config,
            embedding_function=mock_embedding_function,
        )

        start = time.time()

        # Perform mixed operations
        tasks = []

        # 5 working memory writes
        for i in range(5):
            task = orchestrator.store_memory(
                content=f"Working memory {i}",
                agent_id="perf-agent",
                memory_type=MemoryType.WORKING,
                session_id="perf-session",
            )
            tasks.append(task)

        # 5 project memory writes
        for i in range(5):
            task = orchestrator.store_memory(
                content=f"Project memory {i}",
                agent_id="perf-agent",
                memory_type=MemoryType.PROJECT,
                project_id="perf-project",
            )
            tasks.append(task)

        # Execute all in parallel
        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start
        print(f"10 parallel memory operations took {elapsed:.2f}s")

        assert all(r is not None for r in results)
        assert elapsed < 5.0, f"Mixed operations too slow: {elapsed:.2f}s"

        await orchestrator.close()


@pytest.mark.asyncio
class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    async def test_complete_agent_workflow(self, config):
        """Simulate a complete agent workflow with memory."""
        orchestrator = MemoryOrchestrator(
            config=config,
            embedding_function=mock_embedding_function,
        )

        agent_id = "workflow-agent"
        session_id = "workflow-session"
        project_id = "workflow-project"

        # Step 1: Store immediate context in working memory
        await orchestrator.store_memory(
            content="User asked: What is the weather?",
            agent_id=agent_id,
            memory_type=MemoryType.WORKING,
            session_id=session_id,
            importance=0.6,
            tags=["user-query"],
        )

        # Step 2: Store project knowledge
        knowledge_id = await orchestrator.store_memory(
            content="Weather API endpoint: https://api.weather.com/v1/current",
            agent_id=agent_id,
            memory_type=MemoryType.PROJECT,
            project_id=project_id,
            importance=0.9,
            tags=["api", "weather"],
        )

        # Step 3: Store global learnings
        await orchestrator.store_memory(
            content="Weather queries are common user requests",
            agent_id=agent_id,
            memory_type=MemoryType.GLOBAL,
            importance=0.7,
            tags=["patterns", "analytics"],
        )

        # Wait longer for indexing (Qdrant needs time to index)
        await asyncio.sleep(2)

        # Step 4: Retrieve relevant context for response
        relevant = await orchestrator.retrieve_relevant_memories(
            query="weather API information",
            memory_types=[MemoryType.PROJECT, MemoryType.GLOBAL],
            k=10,
            use_hybrid=False,
        )

        # At minimum, we should have stored memories
        assert len(relevant) > 0, "No memories retrieved"

        # Check if we can find relevant content (less strict check)
        has_weather_content = any("weather" in r.memory.content.lower() for r in relevant)
        assert has_weather_content, "Failed to retrieve weather-related memories"

        print(f"Retrieved {len(relevant)} relevant memories - workflow test passed")

        # Clean up
        await orchestrator.close()
