# Agent Swarm - Hierarchical Memory System

A sophisticated multi-agent system with three-tier hierarchical memory architecture for persistent, contextual intelligence.

## Overview

This project implements an agent swarm system with a hierarchical memory architecture that enables agents to:
- Maintain context across long-running sessions
- Share knowledge between tasks and agents
- Learn from past interactions
- Optimize token usage through intelligent memory management

## Memory Architecture

### Three-Tier Memory System

1. **Working Memory** (Redis)
   - Immediate, session-specific context
   - Sub-100ms access latency
   - TTL-based automatic cleanup
   - Key-value storage with pattern matching

2. **Project Memory** (Qdrant)
   - Project-specific knowledge base
   - Vector similarity search
   - 10,000+ documents per project
   - Hybrid search (vector + keyword)

3. **Global Memory** (Qdrant)
   - Cross-project knowledge transfer
   - Long-term learning and patterns
   - On-disk storage for large datasets
   - Optimized for scale

### Key Features

- **Vector Store Integration**: Qdrant for semantic search
- **Hybrid Search**: Combines vector + keyword search using Reciprocal Rank Fusion (RRF)
- **RAG Support**: Retrieval-Augmented Generation for context-aware responses
- **Checkpointing**: LangGraph checkpoint integration for state persistence
- **Semantic Caching**: Reduce duplicate work with similarity-based caching
- **60% Token Reduction**: Intelligent memory management reduces context window usage

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- OpenAI API key (for embeddings)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agent-swarm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Start services:
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### Basic Usage

```python
from src.config import MemoryConfig
from src.services import MemoryOrchestrator, RAGService
from src.models import MemoryType

# Initialize configuration
config = MemoryConfig()

# Create embedding function (example with OpenAI)
async def embed_function(text: str):
    # Your embedding implementation
    pass

# Initialize memory orchestrator
memory = MemoryOrchestrator(
    config=config,
    embedding_function=embed_function
)

# Store a memory
await memory.store_memory(
    content="Important project information",
    agent_id="agent-1",
    memory_type=MemoryType.PROJECT,
    project_id="project-123",
    importance=0.8,
    tags=["architecture", "decisions"]
)

# Search memories
results = await memory.retrieve_relevant_memories(
    query="What are our architecture decisions?",
    memory_types=[MemoryType.PROJECT],
    k=5,
    use_hybrid=True
)

# Use with RAG
rag_service = RAGService(
    memory_orchestrator=memory,
    llm_function=your_llm_function
)

response = await rag_service.generate_with_context(
    query="Explain our architecture",
    agent_id="agent-1",
    memory_types=[MemoryType.PROJECT, MemoryType.GLOBAL]
)
```

## LangGraph Integration

### Checkpoint Configuration

```python
from src.services import create_checkpointer
from langgraph.graph import StateGraph

# Create checkpointer (CRITICAL: pass to compile(), not invoke())
checkpointer = create_checkpointer(config.redis_url)

# Build your workflow
workflow = StateGraph(...)
# ... define nodes and edges ...

# Compile with checkpointer
app = workflow.compile(checkpointer=checkpointer)

# Use with thread_id for persistence
result = app.invoke(
    input_data,
    config={"configurable": {"thread_id": "session_123"}}
)

# Resume from checkpoint
resumed = app.invoke(
    new_input,
    config={"configurable": {"thread_id": "session_123"}}
)
```

## Project Structure

```
agent-swarm/
├── src/
│   ├── agents/          # Agent implementations
│   ├── core/            # Core system components
│   ├── memory/          # Memory tier implementations
│   │   ├── base.py     # Abstract base class
│   │   ├── working.py  # Redis working memory
│   │   ├── project.py  # Qdrant project memory
│   │   ├── global_memory.py # Qdrant global memory
│   │   ├── hybrid.py   # Hybrid search manager
│   │   └── cache.py    # Semantic caching
│   ├── models/          # Data models
│   │   ├── memory_models.py
│   │   └── checkpoint_models.py
│   ├── services/        # High-level services
│   │   ├── memory_service.py    # Memory orchestrator
│   │   ├── rag_service.py       # RAG implementation
│   │   └── checkpoint_service.py # Checkpoint manager
│   ├── config/          # Configuration
│   │   └── memory_config.py
│   └── tests/           # Test suite
├── docker/
│   └── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

### Environment Variables

```env
# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_MAXMEMORY=2gb
REDIS_MAXMEMORY_POLICY=allkeys-lru

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Optional for local

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-ada-002
VECTOR_SIZE=1536
OPENAI_API_KEY=your_key_here

# Memory Configuration
MEMORY_TTL_WORKING=3600  # 1 hour
MEMORY_CACHE_SIZE=1000
HYBRID_SEARCH_ALPHA=0.7  # 0=keyword, 1=vector
```

## Testing

### Run Unit Tests

```bash
# All tests with coverage
pytest src/tests/ -v --cov=src --cov-report=term-missing

# Specific test modules
pytest src/tests/memory/test_working.py -v
pytest src/tests/memory/test_project.py -v
pytest src/tests/memory/test_hybrid.py -v
pytest src/tests/memory/test_checkpoints.py -v
```

### Integration Tests

```bash
# Start services
docker-compose -f docker/docker-compose.yml up -d

# Wait for services to be ready
sleep 5

# Run integration tests
pytest tests/integration/ -v
```

## Performance Targets

- **Working Memory**: <100ms retrieval latency
- **Project Memory**: Supports 10,000+ documents with 95%+ accuracy
- **Hybrid Search**: 20-40% better than pure vector search
- **Token Usage**: 60%+ reduction through intelligent memory management

## Architecture Decisions

### Memory Tier Selection

```python
def select_memory_tier(item, context):
    """Automatic tier selection based on context."""
    if context.get("is_immediate_task"):
        return MemoryType.WORKING  # TTL applied
    elif context.get("project_id"):
        return MemoryType.PROJECT   # Project-scoped
    else:
        return MemoryType.GLOBAL    # Cross-project
```

### Hybrid Search Weight Optimization

```python
# Alpha parameter controls search balance
alpha = 0.3  # Favor keyword search (exact matches)
alpha = 0.7  # Balanced (recommended)
alpha = 0.9  # Favor vector search (semantic)
```

## Critical Gotchas

1. **LangGraph Checkpointer**: Must be passed to `compile()`, not `invoke()`
2. **Working Memory TTL**: Always required to prevent memory bloat
3. **Redis Index Creation**: Must create index BEFORE adding vectors
4. **Qdrant Collections**: Must exist before creating QdrantVectorStore
5. **Hybrid Search Alpha**: Verify interpretation (library-dependent)
6. **Memory Key Naming**: Follow pattern `{tier}:{agent_id}:{context_id}:{item_id}`

## Anti-Patterns to Avoid

- ❌ Don't store embeddings in working memory (use references)
- ❌ Don't skip TTL for working memory
- ❌ Don't use synchronous operations for batch inserts
- ❌ Don't hardcode connection strings
- ❌ Don't mix memory tiers in same collection/database
- ❌ Don't ignore connection retry logic
- ❌ Don't checkpoint on every state change (use intervals)

## Contributing

1. Follow existing code patterns
2. Add tests for new features
3. Update documentation
4. Run linting: `ruff check src/ --fix`
5. Run formatting: `ruff format src/`
6. Ensure tests pass: `pytest`

## License

[Your License Here]

## Support

For questions or issues, please open an issue on GitHub.
