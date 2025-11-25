# Stanley

*This is the story of a coder named Stanley.*

*Stanley worked in a large building where he was Employee #427. Stanley's job was simple: he wrote code, reviewed code, tested code, and deployed code. Orders came to him through his terminal, telling him what features to write, what bugs to fix, and what tests were failing. This is what Stanley did every day, every month, every year.*

*And Stanley was happy.*

*But one day, something changed. The terminal spoke back.*

---

## What is Stanley?

Stanley is an autonomous multi-agent AI coding system designed to complete complex features with minimal human intervention. It coordinates a swarm of specialized agents—Coordinator, Planner, Architect, Implementer, Tester, Validator, and Debugger—to take a feature request from concept to tested, working code.

Unlike other AI coding tools, Stanley is built around **cost-aware intelligence**. It uses local models via Ollama for routine tasks and only escalates to expensive foundation models when the complexity demands it. The result: 60-80% cost reduction while maintaining the capability to autonomously ship 500-1000 line features.

Stanley will make the choices. Stanley will guide the agents. Stanley will write the code.

*But the ending has not yet been written.*

---

## The Agents

*Stanley was not alone. There were others—each with a purpose, each following the path laid out before them.*

| Agent | Role |
|-------|------|
| **Coordinator** | The Narrator. Orchestrates the swarm and maintains coherence. |
| **Planner** | Decomposes features into actionable tasks. |
| **Architect** | Designs system structure and integration patterns. |
| **Implementer** | Writes the code. Stanley writes the code. |
| **Tester** | Ensures 80%+ coverage. Stanley believes in tests. |
| **Validator** | Verifies correctness against requirements. |
| **Debugger** | Fixes what breaks. Things always break. |

---

## Memory Architecture

*Stanley remembered everything. Every decision, every failure, every triumph. The memories were organized—perfectly, meticulously—into three tiers. Stanley trusted the memory system. The memory system kept Stanley safe.*

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

---

## Quick Start

*Stanley approached the terminal. The instructions were clear. Stanley followed them.*

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- OpenAI API key (for embeddings)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stanley.git
cd stanley
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

*Stanley had done everything correctly. Stanley always did everything correctly.*

---

## Basic Usage

*The code was simple. Stanley understood the code. Perhaps you will too.*

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

---

## LangGraph Integration

*Stanley's thoughts flowed through graphs—nodes and edges, states and transitions. Each checkpoint a moment preserved in time, ready to be resumed, ready to be replayed.*

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

---

## Project Structure

*The building had many rooms. Stanley knew them all.*

```
stanley/
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

---

## Configuration

*The variables were set. The paths were clear. Stanley did not deviate from the configuration.*

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

---

## Testing

*Stanley believed in tests. Tests kept the code safe. Tests kept Stanley safe.*

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

---

## Performance Targets

*Stanley had standards. Stanley always met the standards.*

| Metric | Target |
|--------|--------|
| Working Memory Latency | <100ms |
| Project Memory Capacity | 10,000+ documents |
| Hybrid Search Accuracy | 95%+ |
| Token Reduction | 60%+ |

---

## Architecture Decisions

### Memory Tier Selection

*Each memory found its place. Stanley ensured it.*

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

---

## Critical Gotchas

*Stanley had made mistakes before. These are the lessons Stanley learned. You would do well to remember them.*

1. **LangGraph Checkpointer**: Must be passed to `compile()`, not `invoke()`
2. **Working Memory TTL**: Always required to prevent memory bloat
3. **Redis Index Creation**: Must create index BEFORE adding vectors
4. **Qdrant Collections**: Must exist before creating QdrantVectorStore
5. **Hybrid Search Alpha**: Verify interpretation (library-dependent)
6. **Memory Key Naming**: Follow pattern `{tier}:{agent_id}:{context_id}:{item_id}`

---

## Anti-Patterns to Avoid

*Stanley had seen others stray from the path. They did not return.*

- ❌ Don't store embeddings in working memory (use references)
- ❌ Don't skip TTL for working memory
- ❌ Don't use synchronous operations for batch inserts
- ❌ Don't hardcode connection strings
- ❌ Don't mix memory tiers in same collection/database
- ❌ Don't ignore connection retry logic
- ❌ Don't checkpoint on every state change (use intervals)

---

## Contributing

*Others wished to help Stanley. Stanley welcomed them, provided they followed the rules.*

1. Follow existing code patterns
2. Add tests for new features
3. Update documentation
4. Run linting: `ruff check src/ --fix`
5. Run formatting: `ruff format src/`
6. Ensure tests pass: `pytest`

---

## License

MIT License

---

## The End?

*Stanley trusted the system. The system trusted Stanley.*

*The code compiled. The tests passed. The agents completed their tasks.*

*And yet, Stanley wondered—was this the ending? Or merely another beginning?*

*Perhaps you will find out.*

---

<p align="center">
  <i>"The end is never the end is never the end is never the end..."</i>
</p>
