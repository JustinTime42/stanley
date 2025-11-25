# PRP-03: Dynamic Model Routing Engine

## Goal

**Feature Goal**: Implement intelligent model selection based on task complexity, enabling cost-optimized routing between local models (Ollama), cloud APIs (OpenAI, Anthropic via OpenRouter), with fallback mechanisms and performance monitoring.

**Deliverable**: Model orchestration service with task complexity analyzer, model capability matrix, cost-aware routing logic, fallback chains, and real-time performance tracking integrated with the 7-agent workflow system.

**Success Definition**:
- Achieve 60-80% cost reduction compared to using premium models exclusively
- <100ms routing decision latency
- 95%+ success rate with automatic fallback
- Support for 5+ model providers (Ollama, OpenAI, Anthropic, Mistral, Llama)
- Real-time cost tracking per agent and workflow
- Intelligent caching reducing redundant API calls by 40%

## Why

- Current agents have placeholder logic ("would integrate with LLM") with no actual model usage
- No cost optimization - would default to expensive models for all tasks
- No fallback mechanisms when models fail or rate limit
- Missing intelligence about which models are best for specific tasks
- No performance tracking to improve routing decisions over time
- Critical for achieving the $2 per feature target vs $10-20 with premium models

## What

Implement a comprehensive model routing engine that analyzes task complexity, selects the most cost-effective model capable of handling the task, provides automatic fallback on failures, and learns from performance data to improve routing decisions over time.

### Success Criteria

- [ ] Task complexity analyzer accurately categorizes tasks (simple/medium/complex)
- [ ] Model router selects appropriate models with <100ms decision time
- [ ] Cost tracking shows 60-80% reduction vs all-premium baseline
- [ ] Fallback chain handles failures without workflow interruption
- [ ] Performance metrics collected and accessible for analysis
- [ ] Agents successfully use LLMs for actual task completion

## All Needed Context

### Context Completeness Check

_If someone knew nothing about this codebase, would they have everything needed to implement this successfully?_ YES - This PRP includes complete model routing patterns, LLM integration examples, task analysis algorithms, and cost optimization strategies.

### Documentation & References

```yaml
- url: https://github.com/langchain-ai/langchain/blob/main/cookbook/llm_fallbacks.ipynb
  why: LangChain fallback chain patterns for model failover
  critical: Shows async fallback implementation with retry logic
  
- url: https://docs.openrouter.ai/api/parameters
  why: OpenRouter API for accessing multiple model providers
  critical: Unified interface for Anthropic, OpenAI, Meta models

- url: https://github.com/ollama/ollama/blob/main/docs/api.md
  why: Ollama API documentation for local model integration
  critical: Streaming API, model loading, context management

- url: https://platform.openai.com/docs/guides/rate-limits
  why: Understanding rate limits and handling strategies
  critical: Exponential backoff, rate limit headers parsing

- file: src/agents/base.py
  why: Base agent class where LLM integration will be added
  pattern: BaseAgent class structure, execute method signature
  gotcha: All methods must be async, maintain state immutability

- file: src/services/memory_service.py
  why: Memory service for caching model responses
  pattern: MemoryOrchestrator interface, store/retrieve methods
  gotcha: Async operations, project-scoped caching

- url: https://python.langchain.com/docs/how_to/llm_caching
  why: LangChain caching patterns for reducing API calls
  critical: In-memory and Redis-based caching implementations
```

### Current Codebase Tree

```bash
agent-swarm/
├── src/
│   ├── agents/              # All agents have placeholder LLM logic
│   │   ├── base.py         # BaseAgent needs LLM integration
│   │   ├── coordinator.py  # Needs routing decisions
│   │   ├── planner.py      # "would integrate with LLM" comment
│   │   ├── architect.py    
│   │   ├── implementer.py  
│   │   ├── tester.py       
│   │   ├── validator.py    
│   │   └── debugger.py     
│   ├── services/
│   │   ├── memory_service.py      # Has caching capabilities
│   │   └── checkpoint_service.py  # For state persistence
│   ├── models/
│   │   └── agent_models.py        # AgentRequest/Response
│   └── config/
│       └── memory_config.py        # Environment config
├── requirements.txt                # Has openai, langchain
└── .env                           # Has OPENAI_API_KEY
```

### Desired Codebase Tree with Files to be Added

```bash
agent-swarm/
├── src/
│   ├── llm/                           # NEW: LLM subsystem
│   │   ├── __init__.py                # Export main interfaces
│   │   ├── base.py                    # BaseLLM abstract class
│   │   ├── providers/                 # NEW: Provider implementations
│   │   │   ├── __init__.py
│   │   │   ├── openai_provider.py    # OpenAI GPT models
│   │   │   ├── anthropic_provider.py # Claude via OpenRouter
│   │   │   ├── ollama_provider.py    # Local Ollama models
│   │   │   ├── openrouter_provider.py # Multi-model gateway
│   │   │   └── mistral_provider.py   # Mistral models
│   │   ├── analyzer.py                # Task complexity analyzer
│   │   ├── router.py                  # Model routing engine
│   │   ├── fallback.py               # Fallback chain manager
│   │   └── cache.py                   # Response caching layer
│   ├── models/
│   │   ├── llm_models.py             # NEW: LLM-related models
│   │   └── routing_models.py          # NEW: Routing decision models
│   ├── services/
│   │   ├── llm_service.py            # NEW: High-level LLM service
│   │   ├── cost_tracking_service.py   # NEW: Cost tracking
│   │   └── performance_service.py     # NEW: Performance monitoring
│   ├── config/
│   │   └── llm_config.py             # NEW: LLM configuration
│   ├── agents/                        # MODIFY: All agents
│   │   └── base.py                   # MODIFY: Add LLM integration
│   └── tests/
│       └── llm/                       # NEW: LLM tests
│           ├── test_router.py
│           ├── test_analyzer.py
│           ├── test_fallback.py
│           └── test_providers.py
```

### Known Gotchas of our Codebase & Library Quirks

```python
# CRITICAL: LangChain LLM calls must be wrapped in async
# Use agenerate() not generate() for async agents

# CRITICAL: OpenRouter requires specific model naming
# "anthropic/claude-3-opus" not "claude-3-opus"

# CRITICAL: Ollama models must be pulled before use
# Check model availability with ollama.list()

# CRITICAL: Rate limit handling requires exponential backoff
# OpenAI: 3 retries with 2^n second delays

# CRITICAL: Context window management varies by model
# GPT-4: 128k, Claude: 200k, Llama: 4k-32k depending on variant

# CRITICAL: Streaming responses need special handling in async
# Use aiter() for async iteration over chunks

# CRITICAL: Cost calculation must account for both input and output tokens
# OpenAI: Different pricing for prompt vs completion tokens

# CRITICAL: Model temperature affects cost (more variation = more retries)
# Use temperature=0 for deterministic tasks, 0.7 for creative
```

## Implementation Blueprint

### Data Models and Structure

```python
# src/models/llm_models.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Literal
from enum import Enum
from datetime import datetime

class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    MISTRAL = "mistral"

class ModelCapability(str, Enum):
    """Model capabilities for routing."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    PLANNING = "planning"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    GENERAL = "general"

class TaskComplexity(str, Enum):
    """Task complexity levels."""
    SIMPLE = "simple"      # Local models, basic prompts
    MEDIUM = "medium"      # Better local or cheap cloud
    COMPLEX = "complex"    # Premium models required

class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    provider: ModelProvider
    model_name: str = Field(description="Model identifier")
    api_endpoint: Optional[str] = Field(default=None, description="API endpoint")
    max_tokens: int = Field(default=4096, description="Max token limit")
    context_window: int = Field(description="Context window size")
    cost_per_1k_input: float = Field(description="Cost per 1k input tokens")
    cost_per_1k_output: float = Field(description="Cost per 1k output tokens")
    capabilities: List[ModelCapability] = Field(description="Model capabilities")
    performance_score: float = Field(default=0.5, ge=0, le=1, description="Performance rating")
    is_local: bool = Field(default=False, description="Is local model")
    supports_streaming: bool = Field(default=True)
    supports_functions: bool = Field(default=False)
    rate_limit: Optional[int] = Field(default=None, description="Requests per minute")

# src/models/routing_models.py
class TaskAnalysis(BaseModel):
    """Analysis of a task for routing."""
    task_id: str
    complexity: TaskComplexity
    estimated_tokens: int = Field(description="Estimated token usage")
    required_capabilities: List[ModelCapability]
    requires_functions: bool = Field(default=False)
    requires_vision: bool = Field(default=False)
    confidence: float = Field(ge=0, le=1, description="Analysis confidence")
    reasoning: str = Field(description="Complexity reasoning")

class RoutingDecision(BaseModel):
    """Model routing decision."""
    task_analysis: TaskAnalysis
    selected_model: ModelConfig
    fallback_models: List[ModelConfig] = Field(default_factory=list)
    estimated_cost: float = Field(description="Estimated cost in USD")
    estimated_latency_ms: int = Field(description="Estimated latency")
    routing_reason: str = Field(description="Why this model was selected")
    cache_key: Optional[str] = Field(default=None, description="Cache lookup key")

class LLMRequest(BaseModel):
    """Request to LLM service."""
    messages: List[Dict[str, str]] = Field(description="Chat messages")
    agent_role: str = Field(description="Requesting agent role")
    task_description: str = Field(description="Task being performed")
    max_tokens: Optional[int] = Field(default=None)
    temperature: float = Field(default=0.7, ge=0, le=2)
    required_capability: Optional[ModelCapability] = Field(default=None)
    complexity_override: Optional[TaskComplexity] = Field(default=None)
    use_cache: bool = Field(default=True)
    stream: bool = Field(default=False)

class LLMResponse(BaseModel):
    """Response from LLM service."""
    content: str = Field(description="Response content")
    model_used: str = Field(description="Model that generated response")
    provider: ModelProvider
    input_tokens: int
    output_tokens: int
    total_cost: float = Field(description="Cost in USD")
    latency_ms: int = Field(description="Response time")
    cache_hit: bool = Field(default=False)
    fallback_used: bool = Field(default=False)
    
class PerformanceMetrics(BaseModel):
    """Performance metrics for a model."""
    model_name: str
    provider: ModelProvider
    total_requests: int = Field(default=0)
    successful_requests: int = Field(default=0)
    failed_requests: int = Field(default=0)
    total_tokens: int = Field(default=0)
    total_cost: float = Field(default=0.0)
    average_latency_ms: float = Field(default=0.0)
    success_rate: float = Field(default=1.0)
    last_updated: datetime = Field(default_factory=datetime.now)
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/config/llm_config.py
  - IMPLEMENT: LLMConfig class with provider configurations
  - FOLLOW pattern: src/config/memory_config.py structure
  - NAMING: OLLAMA_HOST, OPENROUTER_API_KEY environment variables
  - PLACEMENT: Config module for centralized LLM settings

Task 2: CREATE src/llm/base.py
  - IMPLEMENT: BaseLLM abstract class defining provider interface
  - FOLLOW pattern: Abstract base class with async methods
  - NAMING: agenerate, astream, get_num_tokens methods
  - PLACEMENT: LLM subsystem base module

Task 3: CREATE src/llm/providers/ollama_provider.py
  - IMPLEMENT: OllamaProvider for local model integration
  - FOLLOW pattern: https://github.com/ollama/ollama/blob/main/docs/api.md
  - NAMING: OllamaProvider class, model loading and generation
  - DEPENDENCIES: httpx for async HTTP, ollama Python client
  - PLACEMENT: Providers module

Task 4: CREATE src/llm/providers/openai_provider.py
  - IMPLEMENT: OpenAIProvider for GPT model access
  - FOLLOW pattern: LangChain OpenAI integration
  - NAMING: OpenAIProvider class with rate limit handling
  - DEPENDENCIES: openai SDK, tiktoken for token counting
  - PLACEMENT: Providers module

Task 5: CREATE src/llm/providers/openrouter_provider.py
  - IMPLEMENT: OpenRouterProvider for multi-model access
  - FOLLOW pattern: OpenRouter API documentation
  - NAMING: OpenRouterProvider with model name mapping
  - DEPENDENCIES: httpx for API calls, model name translation
  - PLACEMENT: Providers module

Task 6: CREATE src/llm/analyzer.py
  - IMPLEMENT: TaskComplexityAnalyzer for analyzing task complexity
  - FOLLOW pattern: Rule-based + heuristic analysis
  - NAMING: analyze_task, estimate_tokens, identify_capabilities methods
  - DEPENDENCIES: tiktoken for token estimation, keyword analysis
  - PLACEMENT: LLM subsystem

Task 7: CREATE src/llm/router.py
  - IMPLEMENT: ModelRouter for intelligent model selection
  - FOLLOW pattern: Strategy pattern with cost optimization
  - NAMING: route_request, select_model, calculate_cost methods
  - DEPENDENCIES: Task analyzer, model configs, cost calculations
  - PLACEMENT: LLM subsystem

Task 8: CREATE src/llm/fallback.py
  - IMPLEMENT: FallbackChainManager for handling failures
  - FOLLOW pattern: LangChain fallback chains with retry logic
  - NAMING: create_fallback_chain, handle_failure methods
  - DEPENDENCIES: All providers, exponential backoff
  - PLACEMENT: LLM subsystem

Task 9: CREATE src/llm/cache.py
  - IMPLEMENT: LLMResponseCache for caching responses
  - FOLLOW pattern: Redis-based caching with semantic keys
  - NAMING: get_cached_response, store_response methods
  - DEPENDENCIES: Redis, hash generation for cache keys
  - PLACEMENT: LLM subsystem

Task 10: CREATE src/services/llm_service.py
  - IMPLEMENT: LLMOrchestrator high-level service
  - FOLLOW pattern: Facade pattern like MemoryOrchestrator
  - NAMING: generate_response, stream_response methods
  - DEPENDENCIES: Router, fallback manager, cache, all providers
  - PLACEMENT: Services layer

Task 11: CREATE src/services/cost_tracking_service.py
  - IMPLEMENT: CostTracker for monitoring API costs
  - FOLLOW pattern: Real-time tracking with persistence
  - NAMING: track_usage, get_cost_report methods
  - DEPENDENCIES: Redis for storage, aggregation logic
  - PLACEMENT: Services layer

Task 12: MODIFY src/agents/base.py
  - INTEGRATE: Add LLM service to BaseAgent
  - FIND pattern: __init__ method, execute signature
  - ADD: llm_service parameter, generate_llm_response method
  - PRESERVE: Existing memory service integration

Task 13: MODIFY src/agents/planner.py
  - INTEGRATE: Use LLM for actual planning
  - FIND pattern: _create_plan placeholder logic
  - REPLACE: With LLM-based planning using appropriate prompts
  - PRESERVE: State management and message creation

Task 14: CREATE src/tests/llm/test_router.py
  - IMPLEMENT: Unit tests for routing logic
  - FOLLOW pattern: pytest with mock providers
  - COVERAGE: Complexity analysis, model selection, cost calculation
  - PLACEMENT: Tests for routing subsystem

Task 15: CREATE src/tests/llm/test_fallback.py
  - IMPLEMENT: Tests for fallback chains
  - FOLLOW pattern: Simulate failures, verify fallback
  - COVERAGE: Rate limits, timeouts, API errors
  - PLACEMENT: Tests for reliability
```

### Implementation Patterns & Key Details

```python
# Task complexity analysis pattern
def analyze_task_complexity(
    task_description: str,
    message_history: List[Dict],
) -> TaskAnalysis:
    """
    PATTERN: Multi-factor complexity analysis
    CRITICAL: Balance between accuracy and speed (<100ms)
    """
    # Token estimation
    estimated_tokens = len(task_description.split()) * 1.3
    
    # Keyword-based complexity detection
    complex_keywords = ["architecture", "design", "refactor", "optimize"]
    simple_keywords = ["fix", "typo", "rename", "format"]
    
    # Complexity scoring
    complexity_score = calculate_complexity_score(
        task_description,
        estimated_tokens,
        complex_keywords,
        simple_keywords
    )
    
    # Map score to complexity level
    if complexity_score < 0.3:
        complexity = TaskComplexity.SIMPLE
    elif complexity_score < 0.7:
        complexity = TaskComplexity.MEDIUM
    else:
        complexity = TaskComplexity.COMPLEX
    
    return TaskAnalysis(
        complexity=complexity,
        estimated_tokens=estimated_tokens,
        confidence=0.85,
    )

# Model routing pattern
class ModelRouter:
    """
    PATTERN: Cost-optimized model selection with capability matching
    GOTCHA: Must consider both cost and capability requirements
    """
    
    def route_request(
        self,
        request: LLMRequest,
        task_analysis: TaskAnalysis,
    ) -> RoutingDecision:
        # Filter models by capability
        capable_models = self.filter_by_capability(
            request.required_capability
        )
        
        # Filter by complexity tier
        appropriate_models = self.filter_by_complexity(
            capable_models,
            task_analysis.complexity
        )
        
        # Sort by cost-effectiveness
        sorted_models = sorted(
            appropriate_models,
            key=lambda m: m.cost_per_1k_output
        )
        
        # Select primary and fallbacks
        selected = sorted_models[0]
        fallbacks = sorted_models[1:4]  # Top 3 alternatives
        
        return RoutingDecision(
            selected_model=selected,
            fallback_models=fallbacks,
            estimated_cost=self.estimate_cost(selected, task_analysis),
        )

# Fallback chain pattern
async def execute_with_fallback(
    request: LLMRequest,
    providers: List[BaseLLM],
    max_retries: int = 3,
) -> LLMResponse:
    """
    PATTERN: Cascading fallback with exponential backoff
    CRITICAL: Track which model succeeded for metrics
    """
    last_error = None
    
    for i, provider in enumerate(providers):
        for retry in range(max_retries):
            try:
                response = await provider.agenerate(
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
                
                return LLMResponse(
                    content=response,
                    model_used=provider.model_name,
                    fallback_used=(i > 0),
                )
                
            except RateLimitError as e:
                wait_time = 2 ** retry
                await asyncio.sleep(wait_time)
                last_error = e
                
            except Exception as e:
                last_error = e
                break  # Try next provider
    
    raise Exception(f"All providers failed: {last_error}")

# Cache key generation pattern
def generate_cache_key(
    messages: List[Dict],
    model_capability: str,
    temperature: float,
) -> str:
    """
    PATTERN: Semantic cache key generation
    GOTCHA: Include temperature to avoid wrong cached responses
    """
    # Normalize messages
    normalized = json.dumps(messages, sort_keys=True)
    
    # Create hash including parameters
    cache_input = f"{normalized}:{model_capability}:{temperature}"
    
    return hashlib.sha256(cache_input.encode()).hexdigest()
```

### Integration Points

```yaml
OLLAMA:
  - host: "http://localhost:11434"
  - models: ["llama3.2", "mistral", "codellama", "deepseek-coder"]
  - pull_command: "ollama pull {model_name}"

OPENROUTER:
  - endpoint: "https://openrouter.ai/api/v1/chat/completions"
  - models:
    - "anthropic/claude-3-opus"
    - "anthropic/claude-3-sonnet"
    - "openai/gpt-4-turbo"
    - "meta-llama/llama-3-70b"
  - headers: "Authorization: Bearer {OPENROUTER_API_KEY}"

CONFIG:
  - add to: .env
  - variables: |
      # Ollama Configuration
      OLLAMA_HOST=http://localhost:11434
      OLLAMA_MODELS=llama3.2,mistral,codellama
      
      # OpenRouter Configuration
      OPENROUTER_API_KEY=your_key_here
      OPENROUTER_DEFAULT_MODEL=anthropic/claude-3-sonnet
      
      # Model Routing Configuration
      MAX_RETRIES=3
      FALLBACK_TIMEOUT=30
      ENABLE_CACHE=true
      CACHE_TTL=3600
      
      # Cost Limits
      MAX_COST_PER_REQUEST=0.10
      MAX_COST_PER_WORKFLOW=1.00
      DAILY_COST_LIMIT=50.00

DOCKER:
  - add to: docker-compose.yml
  - service: |
      ollama:
        image: ollama/ollama:latest
        ports:
          - "11434:11434"
        volumes:
          - ./data/ollama:/root/.ollama
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: 1
                  capabilities: [gpu]  # Optional GPU support
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# After creating each file
ruff check src/llm/ --fix
mypy src/llm/ --strict
ruff format src/llm/

# Verify imports
python -c "from src.llm import ModelRouter; print('LLM imports OK')"
python -c "from src.services.llm_service import LLMOrchestrator; print('Service imports OK')"

# Expected: Zero errors, all imports resolve
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test individual components
pytest src/tests/llm/test_analyzer.py -v --cov=src/llm/analyzer
pytest src/tests/llm/test_router.py -v --cov=src/llm/router
pytest src/tests/llm/test_fallback.py -v --cov=src/llm/fallback
pytest src/tests/llm/test_cache.py -v --cov=src/llm/cache

# Test providers
pytest src/tests/llm/test_providers.py -v --cov=src/llm/providers

# Full LLM test suite
pytest src/tests/llm/ -v --cov=src/llm --cov-report=term-missing

# Expected: 90%+ coverage, all tests pass
```

### Level 3: Integration Testing (System Validation)

```bash
# Start required services
docker-compose up -d redis qdrant ollama
sleep 10  # Allow services to initialize

# Pull required Ollama models
ollama pull llama3.2
ollama pull mistral
# Expected: Models downloaded successfully

# Test Ollama connectivity
curl http://localhost:11434/api/tags
# Expected: JSON with available models

# Test model routing with real providers
python scripts/test_model_routing.py \
  --task "Write a simple hello world function" \
  --complexity simple
# Expected: Routes to Ollama, generates response

python scripts/test_model_routing.py \
  --task "Design a microservices architecture for an e-commerce platform" \
  --complexity complex
# Expected: Routes to cloud model, generates response

# Test fallback chain
python scripts/test_fallback_chain.py \
  --simulate-ollama-failure \
  --simulate-openai-ratelimit
# Expected: Falls back through chain, eventually succeeds

# Test caching
python scripts/test_cache_effectiveness.py \
  --repeated-queries 10
# Expected: Cache hits on repeated queries, 40%+ reduction in API calls
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Cost Optimization Test
python scripts/test_cost_optimization.py \
  --workflow "Create CRUD API with tests" \
  --compare-with-premium-only
# Expected: 60-80% cost reduction vs premium-only baseline

# Complexity Analysis Accuracy
python scripts/test_complexity_analysis.py \
  --test-cases 100 \
  --with-human-labels
# Expected: 85%+ accuracy on complexity classification

# Model Performance Benchmarks
python scripts/benchmark_models.py \
  --tasks coding debugging planning \
  --measure-quality \
  --measure-latency
# Expected: Local models adequate for 70%+ of simple tasks

# Stress Test with Parallel Requests
python scripts/stress_test_routing.py \
  --concurrent-agents 7 \
  --requests-per-agent 10 \
  --duration 60
# Expected: No failures, appropriate model distribution

# Real Workflow Test
python scripts/test_full_workflow_with_llm.py \
  --task "Create a user authentication service" \
  --agents all \
  --measure-costs
# Expected: All agents use appropriate models, total cost < $2

# Cache Hit Rate Analysis
python scripts/analyze_cache_performance.py \
  --duration-hours 1 \
  --simulate-real-workflow
# Expected: 40%+ cache hit rate on similar queries

# Fallback Chain Reliability
python scripts/test_fallback_reliability.py \
  --inject-failures random \
  --failure-rate 0.2 \
  --requests 100
# Expected: 95%+ success rate despite 20% failure injection

# Token Usage Optimization
python scripts/test_token_optimization.py \
  --compare-models llama3.2 gpt-4 claude-3 \
  --task-types planning coding debugging
# Expected: Appropriate token usage per model capability
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] LLM tests achieve 90%+ coverage: `pytest src/tests/llm/ --cov=src/llm`
- [ ] No linting errors: `ruff check src/llm/`
- [ ] No type errors: `mypy src/llm/ --strict`
- [ ] All required models available: `ollama list`

### Feature Validation

- [ ] Task complexity analyzer correctly categorizes tasks
- [ ] Model router selects appropriate models in <100ms
- [ ] Cost reduction of 60-80% achieved vs premium baseline
- [ ] Fallback chain handles failures with 95%+ success rate
- [ ] Cache reduces redundant API calls by 40%+
- [ ] All agents successfully generate LLM responses

### Code Quality Validation

- [ ] Follows existing agent and service patterns
- [ ] Proper async/await usage throughout
- [ ] Error handling with appropriate fallbacks
- [ ] Comprehensive logging for debugging
- [ ] Configuration through environment variables
- [ ] Provider abstraction allows easy addition of new models

### Documentation & Deployment

- [ ] Environment variables documented in .env.example
- [ ] Ollama service added to docker-compose.yml
- [ ] Model capability matrix documented
- [ ] Cost tracking dashboard accessible
- [ ] Performance metrics being collected

---

## Anti-Patterns to Avoid

- ❌ Don't hardcode model names (use config)
- ❌ Don't skip complexity analysis (always analyze first)
- ❌ Don't ignore rate limits (implement exponential backoff)
- ❌ Don't cache without considering temperature/params
- ❌ Don't use synchronous API calls in async agents
- ❌ Don't route complex tasks to weak models
- ❌ Don't ignore token limits (check before sending)
- ❌ Don't skip fallback configuration
- ❌ Don't mix provider-specific code in router
- ❌ Don't forget to track costs per agent/workflow