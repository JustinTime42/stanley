# PRP-01: Hierarchical Memory System

## Goal

**Feature Goal**: Implement a three-tier memory architecture (Working, Project, Global) with vector store integration, RAG capabilities, and persistent state management for the agent swarm system.

**Deliverable**: Memory management service with Redis for working memory, Qdrant/Pinecone for project/global memory, hybrid retrieval system, and LangGraph checkpoint integration.

**Success Definition**: 
- Successfully store and retrieve memories across all three tiers
- Achieve <100ms retrieval latency for working memory
- Support 10,000+ memories per project with 95%+ retrieval accuracy
- Enable checkpoint/resume for long-running agent sessions
- Reduce context window usage by 60% through intelligent memory management

## Why

- Current agent system lacks persistent memory across sessions, limiting complex project work
- No context sharing between agents, causing redundant work and inconsistent decisions  
- Limited to context window size, preventing work on large codebases
- No learning from past interactions, missing optimization opportunities
- High token costs from repeatedly processing same information

## What

Implement a hierarchical memory system that enables agents to maintain context across sessions, share knowledge between tasks, and learn from past interactions while optimizing token usage.

### Success Criteria

- [ ] Working memory provides sub-100ms access to immediate context
- [ ] Project memory supports semantic search across 10,000+ memories
- [ ] Global memory enables cross-project knowledge transfer
- [ ] Hybrid search achieves 20-40% better retrieval than pure vector search
- [ ] Checkpoint system allows resuming from any state
- [ ] Memory usage reduces overall token consumption by 60%

## All Needed Context

### Context Completeness Check

_If someone knew nothing about this codebase, would they have everything needed to implement this successfully?_ YES - This PRP includes complete memory architecture patterns, vector store configurations, and integration examples.

### Documentation & References

```yaml
- file: /home/claude/redis_memory_docs.md
  why: Complete Redis implementation patterns for working memory and caching
  pattern: RedisWorkingMemory class structure, connection pooling, error handling
  gotcha: Must use TTL for working memory, connection retry logic critical

- file: /home/claude/qdrant_memory_docs.md  
  why: Qdrant vector store setup for project/global memory tiers
  pattern: QdrantProjectMemory and hybrid search implementations
  gotcha: HNSW parameters must be tuned based on collection size

- url: https://docs.langchain.com/oss/python/langgraph/persistence#checkpointers
  why: LangGraph checkpoint configuration for state persistence
  critical: Must pass checkpointer to compile() method, thread_id required in config

- url: https://redis.io/docs/stack/search/reference/vectors/
  why: Redis vector search configuration and index management
  critical: Index must be created before adding vectors, proper distance metric selection

- url: https://qdrant.tech/documentation/concepts/collections/#collection-parameters
  why: Qdrant collection optimization parameters
  section: HNSW Configuration and Optimizers Config
```

### Current Codebase Tree

```bash
agent-swarm/
├── src/
│   ├── agents/
│   │   ├── coordinator.py
│   │   ├── planner.py
│   │   ├── architect.py
│   │   ├── implementer.py
│   │   ├── tester.py
│   │   ├── validator.py
│   │   └── debugger.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── state.py        # Agent state management
│   │   └── workflow.py     # LangGraph workflow
│   ├── models/
│   │   └── __init__.py
│   ├── utils/
│   │   └── __init__.py
│   └── main.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

### Desired Codebase Tree with Files to be Added

```bash
agent-swarm/
├── src/
│   ├── memory/                    # NEW: Memory subsystem
│   │   ├── __init__.py           # Memory manager factory
│   │   ├── base.py              # Abstract base classes
│   │   ├── working.py           # Redis working memory (immediate context)
│   │   ├── project.py           # Qdrant project memory (project-specific)
│   │   ├── global_memory.py     # Qdrant global memory (cross-project)
│   │   ├── hybrid.py            # Hybrid search implementation
│   │   └── cache.py             # Semantic caching layer
│   ├── models/
│   │   ├── memory_models.py     # NEW: Memory data models
│   │   └── checkpoint_models.py # NEW: Checkpoint schemas
│   ├── services/
│   │   ├── memory_service.py    # NEW: Memory orchestration service
│   │   ├── rag_service.py       # NEW: RAG implementation
│   │   └── checkpoint_service.py # NEW: Checkpoint management
│   ├── config/
│   │   └── memory_config.py     # NEW: Memory system configuration
│   └── tests/
│       └── memory/               # NEW: Memory system tests
│           ├── test_working.py
│           ├── test_project.py
│           ├── test_hybrid.py
│           └── test_checkpoints.py
├── docker/
│   └── docker-compose.yml       # MODIFY: Add Redis and Qdrant services
└── requirements.txt              # MODIFY: Add memory dependencies
```

### Known Gotchas of our Codebase & Library Quirks

```python
# CRITICAL: LangGraph requires checkpointer at compile time, not runtime
# Example: graph.compile(checkpointer=checkpointer) NOT graph.invoke(..., checkpointer=checkpointer)

# CRITICAL: Redis vector search requires index creation BEFORE adding vectors
# Must call create_index() before any add_documents() operations

# CRITICAL: Qdrant collections must exist before creating QdrantVectorStore
# Always check/create collection in __init__ methods

# CRITICAL: Hybrid search alpha parameter is inverse in some libraries
# Qdrant: alpha=0.7 means 70% vector, 30% keyword
# Some others: alpha=0.7 means 70% keyword, 30% vector - VERIFY!

# CRITICAL: Memory keys must follow consistent naming convention
# Pattern: {tier}:{agent_id}:{context_id}:{item_id}

# CRITICAL: TTL required for working memory to prevent memory bloat
# Default: 3600 seconds (1 hour) for working, no TTL for project/global
```

## Implementation Blueprint

### Data Models and Structure

```python
# src/models/memory_models.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

class MemoryType(str, Enum):
    WORKING = "working"
    PROJECT = "project"
    GLOBAL = "global"

class MemoryItem(BaseModel):
    """Base memory item model"""
    id: str = Field(description="Unique memory identifier")
    content: str = Field(description="Memory content/text")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Memory metadata")
    memory_type: MemoryType = Field(description="Memory tier")
    agent_id: str = Field(description="Agent that created memory")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    project_id: Optional[str] = Field(default=None, description="Project identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Creation time")
    importance: float = Field(default=0.5, ge=0, le=1, description="Memory importance score")
    access_count: int = Field(default=0, description="Number of times accessed")
    tags: List[str] = Field(default_factory=list, description="Memory tags")

class MemorySearchRequest(BaseModel):
    """Memory search request model"""
    query: str = Field(description="Search query")
    memory_types: List[MemoryType] = Field(default_factory=lambda: [MemoryType.PROJECT])
    k: int = Field(default=5, ge=1, le=100, description="Number of results")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
    use_hybrid: bool = Field(default=True, description="Use hybrid search")
    alpha: float = Field(default=0.7, ge=0, le=1, description="Hybrid search weight")
    score_threshold: float = Field(default=0.0, description="Minimum similarity score")

class MemorySearchResult(BaseModel):
    """Memory search result model"""
    memory: MemoryItem
    score: float = Field(description="Similarity/relevance score")
    source: str = Field(description="Source collection/tier")
    highlights: Optional[List[str]] = Field(default=None, description="Relevant excerpts")

# src/models/checkpoint_models.py  
class CheckpointMetadata(BaseModel):
    """Checkpoint metadata model"""
    checkpoint_id: str
    thread_id: str
    agent_id: str
    project_id: Optional[str]
    timestamp: datetime
    parent_checkpoint: Optional[str] = None
    checkpoint_type: str = "auto"  # auto, manual, error_recovery
    memory_stats: Dict[str, int] = Field(default_factory=dict)
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/config/memory_config.py
  - IMPLEMENT: MemoryConfig class with Redis/Qdrant connection settings
  - FOLLOW pattern: Environment variable loading with defaults
  - NAMING: REDIS_URL, QDRANT_URL, EMBEDDING_MODEL environment variables
  - PLACEMENT: Config module for centralized settings

Task 2: CREATE src/memory/base.py
  - IMPLEMENT: BaseMemory abstract class defining memory interface
  - FOLLOW pattern: ABC (Abstract Base Class) pattern
  - NAMING: add_memory, get_memory, search_memories, delete_memory methods
  - PLACEMENT: Memory subsystem base module

Task 3: CREATE src/memory/working.py
  - IMPLEMENT: RedisWorkingMemory class for immediate context storage
  - FOLLOW pattern: src/memory/redis_memory_docs.md RedisWorkingMemory implementation
  - NAMING: set_state, get_state, update_state methods  
  - DEPENDENCIES: Redis client, memory_models.MemoryItem
  - PLACEMENT: Memory subsystem working tier

Task 4: CREATE src/memory/project.py
  - IMPLEMENT: QdrantProjectMemory class for project-specific memory
  - FOLLOW pattern: src/memory/qdrant_memory_docs.md QdrantProjectMemory implementation
  - NAMING: add_memories, search_with_filters methods
  - DEPENDENCIES: Qdrant client, langchain_qdrant, embeddings
  - PLACEMENT: Memory subsystem project tier

Task 5: CREATE src/memory/hybrid.py
  - IMPLEMENT: HybridSearchManager combining vector and keyword search
  - FOLLOW pattern: Reciprocal Rank Fusion (RRF) for score combination
  - NAMING: hybrid_search, _reciprocal_rank_fusion methods
  - DEPENDENCIES: Both vector stores, BM25 implementation
  - PLACEMENT: Memory subsystem search module

Task 6: CREATE src/services/memory_service.py
  - IMPLEMENT: MemoryOrchestrator service coordinating all memory tiers
  - FOLLOW pattern: Facade pattern for unified memory interface
  - NAMING: store_memory, retrieve_relevant_memories, checkpoint_state methods
  - DEPENDENCIES: All memory tier implementations
  - PLACEMENT: Services layer for high-level orchestration

Task 7: CREATE src/services/rag_service.py
  - IMPLEMENT: RAGService for retrieval-augmented generation
  - FOLLOW pattern: LangChain RAG chain pattern
  - NAMING: generate_with_context, retrieve_and_generate methods
  - DEPENDENCIES: Memory service, LLM integration
  - PLACEMENT: Services layer for RAG functionality

Task 8: CREATE src/services/checkpoint_service.py
  - IMPLEMENT: CheckpointManager for state persistence
  - FOLLOW pattern: LangGraph checkpoint patterns
  - NAMING: save_checkpoint, load_checkpoint, list_checkpoints methods
  - DEPENDENCIES: Redis/Qdrant for storage, checkpoint models
  - PLACEMENT: Services layer for checkpoint management

Task 9: MODIFY src/core/workflow.py
  - INTEGRATE: Add checkpointer to graph compilation
  - FIND pattern: workflow.compile() calls
  - ADD: checkpointer parameter to compile method
  - PRESERVE: Existing workflow logic

Task 10: MODIFY docker/docker-compose.yml
  - ADD: Redis service with persistence volume
  - ADD: Qdrant service with storage volume
  - FOLLOW pattern: Existing service definitions
  - PRESERVE: Existing services

Task 11: CREATE src/tests/memory/test_working.py
  - IMPLEMENT: Unit tests for working memory operations
  - FOLLOW pattern: pytest fixtures and mocking
  - COVERAGE: CRUD operations, TTL, error handling
  - PLACEMENT: Test directory for memory subsystem

Task 12: CREATE src/tests/memory/test_hybrid.py
  - IMPLEMENT: Integration tests for hybrid search
  - FOLLOW pattern: Test with real vector/keyword data
  - COVERAGE: Score fusion, result ranking, edge cases
  - PLACEMENT: Test directory for search functionality
```

### Implementation Patterns & Key Details

```python
# Memory tier selection pattern
def select_memory_tier(
    memory_item: MemoryItem,
    context: Dict[str, Any]
) -> MemoryType:
    """
    PATTERN: Automatic tier selection based on context
    CRITICAL: Working memory MUST have TTL
    """
    if context.get("is_immediate_task"):
        return MemoryType.WORKING  # TTL will be applied
    elif context.get("project_id"):
        return MemoryType.PROJECT  # No TTL, project-scoped
    else:
        return MemoryType.GLOBAL   # No TTL, cross-project

# Hybrid search weight optimization
def optimize_alpha(
    query_type: str,
    document_characteristics: Dict
) -> float:
    """
    PATTERN: Dynamic alpha adjustment based on query analysis
    GOTCHA: Verify alpha interpretation in your vector store
    """
    if query_type == "exact_match":
        return 0.3  # Favor keyword search
    elif query_type == "semantic":
        return 0.8  # Favor vector search
    else:
        return 0.7  # Balanced default

# Checkpoint integration pattern
from langgraph.checkpoint.redis import RedisSaver

def create_checkpointer(redis_url: str) -> RedisSaver:
    """
    CRITICAL: Must be passed to compile(), not invoke()
    PATTERN: Create once, reuse across workflow
    """
    return RedisSaver(
        redis_url=redis_url,
        key_prefix="agent_swarm:checkpoint:"
    )

# Usage in workflow
checkpointer = create_checkpointer(redis_url)
app = workflow.compile(checkpointer=checkpointer)  # CRITICAL: At compile time!
```

### Integration Points

```yaml
REDIS:
  - connection: "redis://localhost:6379/0"
  - indexes:
    - working_memory_index: "CREATE INDEX ON working:*"
    - cache_index: "CREATE INDEX ON cache:*"
  - config: "maxmemory 2gb, maxmemory-policy allkeys-lru"

QDRANT:
  - connection: "http://localhost:6333"
  - collections:
    - project_memory: "vector_size=1536, distance=Cosine"
    - global_memory: "vector_size=1536, distance=Cosine, on_disk=true"

CONFIG:
  - add to: .env
  - variables: |
      REDIS_URL=redis://localhost:6379
      QDRANT_URL=http://localhost:6333
      EMBEDDING_MODEL=text-embedding-ada-002
      MEMORY_TTL_WORKING=3600
      MEMORY_CACHE_SIZE=1000
      HYBRID_SEARCH_ALPHA=0.7

DOCKER:
  - add to: docker-compose.yml
  - services: |
      redis:
        image: redis/redis-stack:latest
        ports:
          - "6379:6379"
        volumes:
          - ./data/redis:/data
      
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - "6333:6333"
        volumes:
          - ./data/qdrant:/qdrant/storage
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# After creating each Python file
ruff check src/memory/ --fix
mypy src/memory/ --strict
ruff format src/memory/

# Validate imports and dependencies
python -c "from src.memory import MemoryOrchestrator; print('Imports OK')"

# Expected: Zero errors, all imports resolve
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test memory components individually
pytest src/tests/memory/test_working.py -v --cov=src/memory/working
pytest src/tests/memory/test_project.py -v --cov=src/memory/project
pytest src/tests/memory/test_hybrid.py -v --cov=src/memory/hybrid

# Run all memory tests with coverage
pytest src/tests/memory/ -v --cov=src/memory --cov-report=term-missing

# Expected: 80%+ coverage, all tests pass
```

### Level 3: Integration Testing (System Validation)

```bash
# Start required services
docker-compose up -d redis qdrant
sleep 5  # Allow services to initialize

# Test Redis connectivity
redis-cli ping
# Expected: PONG

# Test Qdrant health
curl -f http://localhost:6333/health
# Expected: {"title":"qdrant - vector search engine","version":"..."}

# Test memory service integration
python -m pytest tests/integration/test_memory_integration.py -v

# Test checkpoint save/restore
python scripts/test_checkpoints.py
# Expected: Checkpoint saved and restored successfully

# Test hybrid search accuracy
python scripts/benchmark_hybrid_search.py
# Expected: Hybrid search 20-40% better than pure vector search
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Memory Performance Testing
python scripts/memory_benchmark.py \
  --working-memory-ops 10000 \
  --project-memory-docs 5000 \
  --measure-latency

# Expected: Working memory <100ms, Project memory <500ms

# Memory Persistence Validation
python scripts/test_persistence.py \
  --simulate-crash \
  --verify-recovery
  
# Expected: All memories recovered after restart

# Context Window Optimization Test
python scripts/measure_token_usage.py \
  --with-memory \
  --without-memory \
  --compare

# Expected: 60%+ reduction in token usage with memory

# Cross-Project Knowledge Transfer
python scripts/test_global_memory.py \
  --create-project "project_a" \
  --learn-pattern "error_handling" \
  --switch-project "project_b" \
  --verify-knowledge

# Expected: Knowledge successfully transferred between projects

# Long-Running Session Test
python scripts/test_long_session.py \
  --duration-hours 2 \
  --checkpoint-interval 300 \
  --simulate-interruptions 3

# Expected: Session resumes correctly from checkpoints
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] Memory tests achieve 80%+ coverage: `pytest src/tests/memory/ --cov=src/memory`
- [ ] No linting errors: `ruff check src/memory/`
- [ ] No type errors: `mypy src/memory/ --strict`
- [ ] Docker services healthy: `docker-compose ps`

### Feature Validation

- [ ] Working memory achieves <100ms latency
- [ ] Project memory handles 10,000+ documents
- [ ] Hybrid search shows 20-40% improvement over pure vector
- [ ] Checkpoints save and restore successfully
- [ ] Memory reduces token usage by 60%+
- [ ] Cross-project knowledge transfer works

### Code Quality Validation

- [ ] Follows existing agent swarm patterns
- [ ] Proper error handling with retries for connections
- [ ] TTL applied to working memory only
- [ ] Consistent key naming across all tiers
- [ ] Connection pooling implemented for Redis
- [ ] Batch operations used for Qdrant

### Documentation & Deployment

- [ ] Environment variables documented in .env.example
- [ ] Docker services added to docker-compose.yml
- [ ] README updated with memory system usage
- [ ] Migration guide for existing agent states

---

## Anti-Patterns to Avoid

- ❌ Don't store embeddings in working memory (use references)
- ❌ Don't skip TTL for working memory (causes memory bloat)  
- ❌ Don't use synchronous operations for batch inserts
- ❌ Don't hardcode connection strings (use environment variables)
- ❌ Don't mix memory tiers in same collection/database
- ❌ Don't ignore connection retry logic (services may restart)
- ❌ Don't checkpoint on every state change (use intervals)
- ❌ Don't store raw LLM responses without processing