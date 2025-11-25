# Quick Start Guide - Using Agent Swarm for Real Tasks

This guide shows you how to use the Agent Swarm system for actual software development tasks.

## Prerequisites

1. **Python 3.9+** installed
2. **Dependencies installed**: `pip install -r requirements.txt`
3. **At least one LLM provider configured** (see options below)

## Option 1: Local Setup with Ollama (Recommended for Testing)

### Install Ollama
```bash
# Install Ollama from https://ollama.ai
# Then pull a model:
ollama pull qwen2.5-coder:14b
```

### Configure .env
```bash
# LLM Configuration - Ollama (local, free)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODELS=qwen2.5-coder:14b

# Routing preferences
PREFER_LOCAL=true
ROUTING_STRATEGY=cost_optimized
```

### Test it
```bash
python example_real_task.py
```

**Pros:** Free, fast, runs locally, no API costs
**Cons:** Requires ~8GB RAM, slower than cloud models

---

## Option 2: Cloud Setup with OpenAI (Best Quality)

### Get API Key
1. Sign up at https://platform.openai.com
2. Create an API key
3. Add credits to your account

### Configure .env
```bash
# LLM Configuration - OpenAI
OPENAI_API_KEY=sk-your-key-here
OPENAI_DEFAULT_MODEL=gpt-4o-mini  # or gpt-4o for better quality

# Cost limits
MAX_COST_PER_REQUEST=0.10
MAX_COST_PER_WORKFLOW=1.00
DAILY_COST_LIMIT=50.00

# Routing preferences
PREFER_LOCAL=false
ROUTING_STRATEGY=balanced
```

### Test it
```bash
python example_real_task.py
```

**Pros:** Best quality, no local resources needed, fast
**Cons:** Costs money (~$0.10-1.00 per workflow)

---

## Option 3: OpenRouter (Access Many Models)

### Get API Key
1. Sign up at https://openrouter.ai
2. Add credits ($5-10 recommended)
3. Get your API key

### Configure .env
```bash
# LLM Configuration - OpenRouter
OPENROUTER_API_KEY=your-key-here
OPENROUTER_DEFAULT_MODEL=anthropic/claude-3-sonnet

# Cost limits
MAX_COST_PER_REQUEST=0.10
MAX_COST_PER_WORKFLOW=1.00

PREFER_LOCAL=false
ROUTING_STRATEGY=cost_optimized
```

**Pros:** Access to many models, flexible pricing
**Cons:** Costs money, requires internet

---

## Optional: Enable Memory Features

The Agent Swarm has advanced memory capabilities (Redis + Qdrant) for context retention across tasks.

### Start Services
```bash
docker-compose -f docker/docker-compose.yml up -d
```

### Configure .env
```bash
# Redis (Working Memory)
REDIS_URL=redis://localhost:6379/0

# Qdrant (Vector Memory)
QDRANT_URL=http://localhost:6333

# Embeddings (for memory)
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_MODEL=text-embedding-ada-002
```

**When to use:**
- Long-running projects with context
- Tasks that need to reference past work
- Multi-session workflows

**When to skip:**
- Simple one-off tasks
- Testing/demos
- Quick experiments

---

## Running Your First Real Task

### 1. Verify Setup
```bash
# Run the demo to verify everything works
python run.py
```

### 2. Run Example Task
```bash
# Run the example calculator task
python example_real_task.py
```

This will:
- ✓ Connect to your LLM provider
- ✓ Break down the task into subtasks
- ✓ Generate implementation code
- ✓ Create unit tests
- ✓ Validate the output
- ✓ Save generated files

### 3. Check Output
```bash
# Generated code will be in:
ls output/calculator_project/
```

---

## Creating Your Own Tasks

### Simple Task Example
```python
import asyncio
from src.services.workflow_service import WorkflowOrchestrator
from src.models.workflow_models import WorkflowConfig

task = {
    "id": "my_task",
    "description": "Build a REST API for user management",
    "requirements": [
        "FastAPI framework",
        "SQLite database",
        "CRUD operations for users",
        "Input validation with Pydantic",
        "JWT authentication",
    ]
}

config = WorkflowConfig(
    project_id="user_api",
    enable_human_approval=False,
    max_retries=3,
)

# Run it
orchestrator = WorkflowOrchestrator(...)
execution = await orchestrator.start_workflow(task, config)
```

### Advanced Task Example
```python
task = {
    "id": "complex_system",
    "description": "Build a microservices architecture",
    "requirements": [
        "API Gateway service",
        "User service with authentication",
        "Product service with inventory",
        "Order service with payment integration",
        "Docker compose for local development",
        "Kubernetes manifests for deployment",
    ],
    "constraints": [
        "Python FastAPI for all services",
        "PostgreSQL for databases",
        "Redis for caching",
        "RabbitMQ for messaging",
    ],
    "context": {
        "team_size": 3,
        "timeline": "2 weeks",
        "focus": "MVP with basic features",
    }
}

config = WorkflowConfig(
    project_id="microservices_platform",
    enable_human_approval=True,  # Review before execution
    max_retries=5,
    save_artifacts=True,
    enable_testing=True,
    enable_validation=True,
)
```

---

## Understanding the Workflow

The agent swarm executes tasks through 7 specialized agents:

```
┌─────────────┐
│ Coordinator │ ← Entry point, analyzes task
└──────┬──────┘
       ↓
┌─────────────┐
│   Planner   │ ← Breaks down into subtasks
└──────┬──────┘
       ↓
┌─────────────┐
│  Architect  │ ← Designs system architecture
└──────┬──────┘
       ↓
┌─────────────┐
│ Implementer │ ← Generates code
└──────┬──────┘
       ↓
┌─────────────┐
│   Tester    │ ← Creates and runs tests
└──────┬──────┘
       ↓
┌─────────────┐
│  Validator  │ ← Quality assurance
└──────┬──────┘
       ↓
     (Done)

Note: Debugger intervenes if tests fail or validation issues found
```

---

## Cost Optimization

### Tips to Minimize Costs

1. **Use local models** (Ollama) for development/testing
2. **Set cost limits** in .env:
   ```bash
   MAX_COST_PER_REQUEST=0.10
   MAX_COST_PER_WORKFLOW=1.00
   DAILY_COST_LIMIT=50.00
   ```

3. **Enable intelligent routing**:
   ```bash
   ROUTING_STRATEGY=cost_optimized
   PREFER_LOCAL=true
   ```

4. **Start small** - Test with simple tasks first

5. **Use caching**:
   ```bash
   ENABLE_CACHE=true
   CACHE_TTL=3600
   ```

### Expected Costs (with OpenAI GPT-4o-mini)

- Simple task (calculator): ~$0.05-0.15
- Medium task (REST API): ~$0.20-0.50
- Complex task (microservices): ~$1.00-3.00

---

## Troubleshooting

### "No LLM service available"
- Check your .env file has API keys
- Verify Ollama is running: `ollama list`
- Test API key: `curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"`

### "Connection timeout"
- Check Redis/Qdrant are running (if enabled)
- Disable memory services if not needed
- Check firewall settings

### "Task failed - no implementation"
- Verify tool service is configured
- Check task requirements are clear
- Try with a simpler task first

### "Tests failing"
- This is normal! The debugger will fix them
- Check `max_retries` in config
- Review generated code in output directory

---

## Next Steps

1. ✅ Run `python run.py` to verify setup
2. ✅ Run `python example_real_task.py` for first real task
3. ✅ Modify the task in example_real_task.py
4. ✅ Create your own task script
5. ✅ Enable memory features for complex projects
6. ✅ Explore advanced features (PRP-* docs)

## Getting Help

- **Demo**: `python run.py` - System capabilities
- **Docs**: `README.md` - Full documentation
- **Tests**: `pytest src/tests/` - Verify installation
- **Issues**: Check GitHub issues or create new one

---

Happy building! 🚀
