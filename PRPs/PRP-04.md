# PRP-04: Enhanced Tool Abstraction Layer

## Goal

**Feature Goal**: Implement a unified tool abstraction layer with standardized interfaces, comprehensive error handling, retry mechanisms, usage analytics, resource tracking, and async/parallel execution capabilities for all agent tools.

**Deliverable**: Tool management system that provides a consistent interface for agents to interact with external resources (file system, Git, code analyzers, test runners, APIs), with built-in resilience, monitoring, and optimization capabilities.

**Success Definition**:
- All agents successfully use tools through unified interface
- 99%+ tool reliability with automatic retry and fallback
- <50ms overhead for tool invocation
- Resource usage tracked per tool/agent
- Parallel tool execution reduces workflow time by 40%
- Tool usage analytics dashboard operational

## Why

- Agents currently have placeholder implementations (_implement_code, _create_tests) with no actual tool integration
- No standardized way for agents to interact with external systems
- Missing error handling and retry logic for tool failures
- No resource usage tracking or optimization
- No parallel execution capability for independent tools
- Critical for enabling agents to perform real work (file operations, testing, etc.)

## What

Implement a comprehensive tool abstraction layer that provides agents with a standardized interface to interact with various tools and external systems, including error handling, retry logic, resource tracking, and parallel execution capabilities.

### Success Criteria

- [ ] Tool registry with 10+ implemented tools
- [ ] Automatic retry with exponential backoff on failures
- [ ] Resource usage tracked (CPU, memory, I/O per tool)
- [ ] Parallel execution reduces task time by 40%+
- [ ] Tool usage analytics accessible via API
- [ ] 99%+ reliability with fallback mechanisms

## All Needed Context

### Context Completeness Check

_If someone knew nothing about this codebase, would they have everything needed to implement this successfully?_ YES - This PRP includes complete tool patterns, error handling strategies, resource tracking methods, and parallel execution patterns.

### Documentation & References

```yaml
- url: https://python.langchain.com/docs/how_to/tools_as_openai_functions/
  why: LangChain tool patterns for function calling
  critical: Shows how to create tool schemas and integrate with LLMs
  
- url: https://docs.python.org/3/library/asyncio-task.html#asyncio.gather
  why: Async parallel execution patterns
  critical: asyncio.gather for parallel tool execution, error handling

- url: https://github.com/Textualize/rich
  why: Rich library for tool output formatting and progress tracking
  critical: Progress bars for long-running tools, formatted output

- file: src/agents/base.py
  why: BaseAgent class where tool integration will be added
  pattern: BaseAgent structure, execute method pattern
  gotcha: All methods must be async, maintain state immutability

- file: src/agents/implementer.py
  why: Example agent that needs file system and code generation tools
  pattern: Current placeholder methods like _implement_code
  gotcha: Need to replace placeholders with actual tool calls

- file: src/services/llm_service.py
  why: LLM service pattern for high-level orchestration
  pattern: Service facade pattern, error handling
  gotcha: Async context managers, proper cleanup

- url: https://docs.python.org/3/library/resource.html
  why: Resource tracking for CPU and memory usage
  critical: resource.getrusage() for process resource tracking
```

### Current Codebase Tree

```bash
agent-swarm/
├── src/
│   ├── agents/               # All agents need tool integration
│   │   ├── implementer.py    # Needs: file ops, code gen
│   │   ├── tester.py         # Needs: test runners
│   │   ├── validator.py      # Needs: linters, analyzers
│   │   └── debugger.py       # Needs: debuggers, profilers
│   ├── services/            
│   │   └── llm_service.py    # Pattern for service orchestration
│   └── models/
│       └── agent_models.py   # AgentResponse model
```

### Desired Codebase Tree with Files to be Added

```bash
agent-swarm/
├── src/
│   ├── tools/                          # NEW: Tool subsystem
│   │   ├── __init__.py                # Export main interfaces
│   │   ├── base.py                    # BaseTool abstract class
│   │   ├── registry.py                # Tool registry and discovery
│   │   ├── executor.py                # Tool execution engine
│   │   ├── retry.py                   # Retry and fallback logic
│   │   ├── monitor.py                 # Resource monitoring
│   │   ├── parallel.py                # Parallel execution manager
│   │   └── implementations/           # NEW: Concrete tool implementations
│   │       ├── __init__.py
│   │       ├── file_tools.py         # File system operations
│   │       ├── code_tools.py         # Code generation/manipulation
│   │       ├── git_tools.py          # Version control operations
│   │       ├── test_tools.py         # Test execution (pytest, jest, etc.)
│   │       ├── validation_tools.py   # Linting, type checking
│   │       ├── build_tools.py        # Build and compilation
│   │       ├── docker_tools.py       # Container operations
│   │       ├── api_tools.py          # External API interactions
│   │       ├── search_tools.py       # Code/documentation search
│   │       └── analysis_tools.py     # Code analysis, metrics
│   ├── models/
│   │   ├── tool_models.py            # NEW: Tool-related models
│   │   └── execution_models.py        # NEW: Execution tracking models
│   ├── services/
│   │   ├── tool_service.py           # NEW: High-level tool orchestration
│   │   └── analytics_service.py       # NEW: Tool usage analytics
│   ├── agents/                        # MODIFY: All agents
│   │   └── base.py                   # MODIFY: Add tool manager integration
│   └── tests/
│       └── tools/                     # NEW: Tool tests
│           ├── test_executor.py
│           ├── test_retry.py
│           ├── test_parallel.py
│           └── test_implementations.py
```

### Known Gotchas of our Codebase & Library Quirks

```python
# CRITICAL: All tool operations must be async for agent integration
# Use async file operations, async subprocess, etc.

# CRITICAL: Tool results must be JSON-serializable for state storage
# Convert complex objects to dictionaries

# CRITICAL: Resource tracking requires proper context management
# Use async context managers for cleanup

# CRITICAL: Parallel execution must handle partial failures
# Return both successful results and errors

# CRITICAL: Tool schemas must be compatible with function calling
# Follow OpenAI function schema format for LLM integration

# CRITICAL: File paths must be sandboxed to project directory
# Prevent tools from accessing system files

# CRITICAL: Long-running tools need progress reporting
# Use callbacks or async generators for progress updates

# CRITICAL: Tool timeouts must be configurable
# Different tools have different acceptable durations
```

## Implementation Blueprint

### Data Models and Structure

```python
# src/models/tool_models.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Callable, Literal
from enum import Enum
from datetime import datetime

class ToolCategory(str, Enum):
    """Tool categories for organization and discovery."""
    FILE_SYSTEM = "file_system"
    CODE_GENERATION = "code_generation"
    VERSION_CONTROL = "version_control"
    TESTING = "testing"
    VALIDATION = "validation"
    BUILD = "build"
    CONTAINER = "container"
    EXTERNAL_API = "external_api"
    SEARCH = "search"
    ANALYSIS = "analysis"

class ToolStatus(str, Enum):
    """Tool execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"

class ToolParameter(BaseModel):
    """Tool parameter definition for schema."""
    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type (string, integer, etc.)")
    description: str = Field(description="Parameter description")
    required: bool = Field(default=True)
    default: Optional[Any] = Field(default=None)
    enum: Optional[List[Any]] = Field(default=None, description="Allowed values")

class ToolSchema(BaseModel):
    """Tool schema for function calling and validation."""
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    category: ToolCategory
    parameters: List[ToolParameter] = Field(default_factory=list)
    returns: str = Field(description="Return type description")
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    timeout_seconds: int = Field(default=30)
    max_retries: int = Field(default=3)
    requires_confirmation: bool = Field(default=False)

class ToolRequest(BaseModel):
    """Request to execute a tool."""
    tool_name: str = Field(description="Name of tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    agent_id: str = Field(description="Agent making the request")
    workflow_id: str = Field(description="Workflow context")
    timeout_override: Optional[int] = Field(default=None)
    async_execution: bool = Field(default=False)
    priority: int = Field(default=5, ge=1, le=10)

class ToolResult(BaseModel):
    """Result from tool execution."""
    tool_name: str
    status: ToolStatus
    result: Optional[Any] = Field(default=None, description="Tool output")
    error: Optional[str] = Field(default=None)
    execution_time_ms: int = Field(description="Execution duration")
    resource_usage: Dict[str, float] = Field(default_factory=dict)
    retry_count: int = Field(default=0)
    timestamp: datetime = Field(default_factory=datetime.now)

# src/models/execution_models.py
class ExecutionMetrics(BaseModel):
    """Metrics for tool execution."""
    tool_name: str
    total_executions: int = Field(default=0)
    successful_executions: int = Field(default=0)
    failed_executions: int = Field(default=0)
    total_retries: int = Field(default=0)
    average_execution_time_ms: float = Field(default=0.0)
    total_cpu_seconds: float = Field(default=0.0)
    total_memory_mb: float = Field(default=0.0)
    last_execution: Optional[datetime] = Field(default=None)
    error_types: Dict[str, int] = Field(default_factory=dict)

class ParallelExecutionPlan(BaseModel):
    """Plan for parallel tool execution."""
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_groups: List[List[ToolRequest]] = Field(
        description="Groups of tools that can run in parallel"
    )
    dependencies: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Tool dependencies (tool -> [depends_on])"
    )
    estimated_time_ms: int = Field(description="Estimated total execution time")
    max_parallelism: int = Field(default=5, description="Max concurrent executions")

class ResourceUsage(BaseModel):
    """Resource usage tracking."""
    cpu_percent: float = Field(description="CPU usage percentage")
    memory_mb: float = Field(description="Memory usage in MB")
    io_read_bytes: int = Field(default=0)
    io_write_bytes: int = Field(default=0)
    network_sent_bytes: int = Field(default=0)
    network_recv_bytes: int = Field(default=0)
    open_files: int = Field(default=0)
    thread_count: int = Field(default=1)
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/tools/base.py
  - IMPLEMENT: BaseTool abstract class defining tool interface
  - FOLLOW pattern: Abstract base class with async execute
  - NAMING: BaseTool, execute, validate_parameters, get_schema methods
  - PLACEMENT: Tool subsystem base module

Task 2: CREATE src/tools/registry.py
  - IMPLEMENT: ToolRegistry for tool discovery and management
  - FOLLOW pattern: Singleton registry pattern
  - NAMING: ToolRegistry, register_tool, get_tool, list_tools methods
  - DEPENDENCIES: BaseTool, ToolSchema
  - PLACEMENT: Tool subsystem

Task 3: CREATE src/tools/retry.py
  - IMPLEMENT: RetryManager with exponential backoff
  - FOLLOW pattern: Decorator pattern for retry logic
  - NAMING: RetryManager, with_retry decorator, handle_failure method
  - DEPENDENCIES: asyncio, exponential backoff algorithm
  - PLACEMENT: Tool subsystem

Task 4: CREATE src/tools/monitor.py
  - IMPLEMENT: ResourceMonitor for tracking tool resource usage
  - FOLLOW pattern: Context manager for resource tracking
  - NAMING: ResourceMonitor, track_resources, get_usage methods
  - DEPENDENCIES: psutil, resource module
  - PLACEMENT: Tool subsystem

Task 5: CREATE src/tools/executor.py
  - IMPLEMENT: ToolExecutor for running tools with monitoring
  - FOLLOW pattern: Async executor with timeout and retry
  - NAMING: ToolExecutor, execute_tool, validate_request methods
  - DEPENDENCIES: Registry, retry manager, resource monitor
  - PLACEMENT: Tool subsystem

Task 6: CREATE src/tools/parallel.py
  - IMPLEMENT: ParallelExecutor for concurrent tool execution
  - FOLLOW pattern: asyncio.gather with error handling
  - NAMING: ParallelExecutor, execute_parallel, build_execution_plan
  - DEPENDENCIES: ToolExecutor, dependency resolution
  - PLACEMENT: Tool subsystem

Task 7: CREATE src/tools/implementations/file_tools.py
  - IMPLEMENT: FileSystemTools for file operations
  - FOLLOW pattern: Individual tool classes inheriting BaseTool
  - NAMING: ReadFileTool, WriteFileTool, ListDirectoryTool, etc.
  - DEPENDENCIES: aiofiles for async file operations
  - PLACEMENT: Tool implementations

Task 8: CREATE src/tools/implementations/code_tools.py
  - IMPLEMENT: CodeGenerationTools for code creation/modification
  - FOLLOW pattern: Tool classes with LLM integration
  - NAMING: GenerateCodeTool, RefactorCodeTool, AddTestsTool
  - DEPENDENCIES: LLM service, AST manipulation
  - PLACEMENT: Tool implementations

Task 9: CREATE src/tools/implementations/test_tools.py
  - IMPLEMENT: TestingTools for running tests
  - FOLLOW pattern: Subprocess execution with output parsing
  - NAMING: PytestTool, JestTool, UnittestTool
  - DEPENDENCIES: asyncio.subprocess, output parsers
  - PLACEMENT: Tool implementations

Task 10: CREATE src/tools/implementations/git_tools.py
  - IMPLEMENT: GitTools for version control operations
  - FOLLOW pattern: GitPython or subprocess git commands
  - NAMING: GitStatusTool, GitCommitTool, GitDiffTool
  - DEPENDENCIES: GitPython or subprocess
  - PLACEMENT: Tool implementations

Task 11: CREATE src/tools/implementations/validation_tools.py
  - IMPLEMENT: ValidationTools for code quality checks
  - FOLLOW pattern: Subprocess execution of linters/formatters
  - NAMING: RuffTool, MypyTool, BlackTool, ESLintTool
  - DEPENDENCIES: Tool-specific packages
  - PLACEMENT: Tool implementations

Task 12: CREATE src/services/tool_service.py
  - IMPLEMENT: ToolOrchestrator high-level service
  - FOLLOW pattern: Facade pattern like LLMOrchestrator
  - NAMING: ToolOrchestrator, execute, execute_batch methods
  - DEPENDENCIES: All tool components
  - PLACEMENT: Services layer

Task 13: CREATE src/services/analytics_service.py
  - IMPLEMENT: ToolAnalytics for usage tracking and reporting
  - FOLLOW pattern: Metrics aggregation and storage
  - NAMING: ToolAnalytics, record_execution, get_metrics methods
  - DEPENDENCIES: Redis for storage, metrics calculation
  - PLACEMENT: Services layer

Task 14: MODIFY src/agents/base.py
  - INTEGRATE: Add tool_service to BaseAgent
  - FIND pattern: __init__ method signature
  - ADD: tool_service parameter, execute_tool helper method
  - PRESERVE: Existing memory and LLM service integration

Task 15: MODIFY src/agents/implementer.py
  - INTEGRATE: Replace placeholder with actual tool calls
  - FIND pattern: _implement_code placeholder method
  - REPLACE: With actual file and code generation tool calls
  - PRESERVE: State management and error handling

Task 16: CREATE src/tests/tools/test_executor.py
  - IMPLEMENT: Unit tests for tool execution
  - FOLLOW pattern: pytest-asyncio with mocks
  - COVERAGE: Execution, retry, timeout, resource tracking
  - PLACEMENT: Tests for tool subsystem

Task 17: CREATE src/tests/tools/test_parallel.py
  - IMPLEMENT: Tests for parallel execution
  - FOLLOW pattern: Mock tools with controlled delays
  - COVERAGE: Parallel execution, dependency resolution, partial failures
  - PLACEMENT: Tests for parallelization
```

### Implementation Patterns & Key Details

```python
# Tool implementation pattern
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTool(ABC):
    """
    PATTERN: Abstract base for all tools
    CRITICAL: Must be async and JSON-serializable
    """
    
    def __init__(self, name: str, category: ToolCategory):
        self.name = name
        self.category = category
        self.schema = self._build_schema()
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute tool with given parameters."""
        pass
    
    @abstractmethod
    def _build_schema(self) -> ToolSchema:
        """Build tool schema for discovery."""
        pass
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate parameters against schema."""
        # Parameter validation logic
        pass

# Retry decorator pattern
def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    retry_on: tuple = (Exception,)
):
    """
    PATTERN: Decorator for automatic retry with exponential backoff
    CRITICAL: Must handle async functions
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = backoff_factor ** attempt
                        await asyncio.sleep(delay)
                    continue
            raise last_error
        return wrapper
    return decorator

# Resource monitoring pattern
class ResourceMonitor:
    """
    PATTERN: Context manager for resource tracking
    GOTCHA: Must handle async context
    """
    
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.start_resources = None
        self.end_resources = None
    
    async def __aenter__(self):
        self.start_resources = self._get_current_resources()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_resources = self._get_current_resources()
        # Calculate and store resource usage
    
    def _get_current_resources(self) -> ResourceUsage:
        """Get current resource usage."""
        import psutil
        process = psutil.Process()
        return ResourceUsage(
            cpu_percent=process.cpu_percent(),
            memory_mb=process.memory_info().rss / 1024 / 1024,
            # ... other metrics
        )

# Parallel execution pattern
async def execute_parallel(
    tool_requests: List[ToolRequest],
    max_parallelism: int = 5
) -> List[ToolResult]:
    """
    PATTERN: Bounded parallel execution with error handling
    CRITICAL: Must handle partial failures
    """
    semaphore = asyncio.Semaphore(max_parallelism)
    
    async def bounded_execute(request: ToolRequest):
        async with semaphore:
            try:
                return await execute_tool(request)
            except Exception as e:
                return ToolResult(
                    tool_name=request.tool_name,
                    status=ToolStatus.FAILED,
                    error=str(e)
                )
    
    # Execute all tools, collecting both successes and failures
    results = await asyncio.gather(
        *[bounded_execute(req) for req in tool_requests],
        return_exceptions=False  # Don't raise, return as results
    )
    
    return results

# File tool implementation example
class ReadFileTool(BaseTool):
    """
    PATTERN: Concrete tool implementation
    GOTCHA: Must use async file operations
    """
    
    def __init__(self):
        super().__init__(
            name="read_file",
            category=ToolCategory.FILE_SYSTEM
        )
    
    async def execute(self, path: str, **kwargs) -> ToolResult:
        """Read file contents asynchronously."""
        import aiofiles
        
        try:
            async with aiofiles.open(path, 'r') as f:
                content = await f.read()
            
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                result={"content": content, "path": path},
                execution_time_ms=10,  # Track actual time
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error=str(e),
                execution_time_ms=5,
            )
```

### Integration Points

```yaml
DEPENDENCIES:
  - add to: requirements.txt
  - packages: |
      aiofiles>=23.0.0       # Async file operations
      psutil>=5.9.0          # Resource monitoring
      GitPython>=3.1.0       # Git operations
      rich>=13.0.0           # Progress bars and formatting
      asyncio-throttle>=1.0.0  # Rate limiting

CONFIG:
  - add to: .env
  - variables: |
      # Tool Configuration
      TOOL_MAX_RETRIES=3
      TOOL_RETRY_BACKOFF=2.0
      TOOL_DEFAULT_TIMEOUT=30
      TOOL_MAX_PARALLELISM=5
      
      # Resource Limits
      TOOL_MAX_CPU_PERCENT=80
      TOOL_MAX_MEMORY_MB=1024
      TOOL_MAX_FILE_SIZE_MB=100
      
      # Sandboxing
      TOOL_SANDBOX_PATH=/tmp/agent_workspace
      TOOL_ALLOWED_PATHS=/home/user/projects
      
      # Analytics
      TOOL_METRICS_ENABLED=true
      TOOL_METRICS_INTERVAL=60

MONITORING:
  - endpoint: "/metrics/tools"
  - metrics: |
      - tool_execution_count
      - tool_success_rate  
      - tool_execution_time
      - tool_resource_usage
      - tool_error_count
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# After creating each file
ruff check src/tools/ --fix
mypy src/tools/ --strict
ruff format src/tools/

# Verify imports
python -c "from src.tools import ToolExecutor; print('Tool imports OK')"
python -c "from src.services.tool_service import ToolOrchestrator; print('Service imports OK')"

# Expected: Zero errors, all imports resolve
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test individual components
pytest src/tests/tools/test_executor.py -v --cov=src/tools/executor
pytest src/tests/tools/test_retry.py -v --cov=src/tools/retry
pytest src/tests/tools/test_parallel.py -v --cov=src/tools/parallel
pytest src/tests/tools/test_monitor.py -v --cov=src/tools/monitor

# Test tool implementations
pytest src/tests/tools/test_implementations.py -v --cov=src/tools/implementations

# Full tool test suite
pytest src/tests/tools/ -v --cov=src/tools --cov-report=term-missing

# Expected: 95%+ coverage, all tests pass
```

### Level 3: Integration Testing (System Validation)

```bash
# Test file operations
python scripts/test_file_tools.py \
  --create-file test.txt \
  --write-content "Hello World" \
  --read-back \
  --verify
# Expected: File created, written, and read successfully

# Test code generation tools
python scripts/test_code_tools.py \
  --generate "Calculator class with add method" \
  --language python \
  --verify-syntax
# Expected: Valid Python code generated

# Test parallel execution
python scripts/test_parallel_tools.py \
  --tools "read_file,list_directory,git_status" \
  --measure-time
# Expected: Parallel execution faster than sequential

# Test retry mechanism
python scripts/test_retry_logic.py \
  --inject-failures 2 \
  --max-retries 3
# Expected: Succeeds on third attempt after 2 failures

# Test resource monitoring
python scripts/test_resource_tracking.py \
  --execute-heavy-tool \
  --monitor-resources \
  --verify-limits
# Expected: Resources tracked and limits enforced
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Full Agent Workflow with Tools
python scripts/test_agent_with_tools.py \
  --agent implementer \
  --task "Create a REST API with CRUD operations" \
  --verify-tool-usage
# Expected: Agent uses multiple tools to complete task

# Tool Performance Benchmark
python scripts/benchmark_tools.py \
  --tools all \
  --iterations 100 \
  --measure-overhead
# Expected: <50ms overhead per tool invocation

# Parallel Execution Optimization
python scripts/test_parallel_optimization.py \
  --workflow "Build and test project" \
  --compare-sequential \
  --measure-speedup
# Expected: 40%+ speedup with parallel execution

# Resource Usage Analysis
python scripts/analyze_resource_usage.py \
  --run-workflow \
  --track-per-tool \
  --generate-report
# Expected: Detailed resource usage report per tool

# Tool Reliability Test
python scripts/test_tool_reliability.py \
  --continuous-hours 1 \
  --random-failures 0.05 \
  --verify-recovery
# Expected: 99%+ success rate despite 5% failure injection

# Analytics Dashboard Test
python scripts/test_analytics_dashboard.py \
  --generate-traffic \
  --verify-metrics \
  --check-api
# Expected: Metrics collected and accessible via API

# Sandbox Security Test
python scripts/test_sandbox_security.py \
  --attempt-breakout \
  --verify-containment
# Expected: All breakout attempts blocked

# Tool Discovery and Schema
python scripts/test_tool_discovery.py \
  --list-all-tools \
  --validate-schemas \
  --check-examples
# Expected: All tools discoverable with valid schemas
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] Tool tests achieve 95%+ coverage: `pytest src/tests/tools/ --cov=src/tools`
- [ ] No linting errors: `ruff check src/tools/`
- [ ] No type errors: `mypy src/tools/ --strict`
- [ ] All tool implementations working

### Feature Validation

- [ ] 10+ tools implemented and registered
- [ ] Retry mechanism working with exponential backoff
- [ ] Resource tracking accurate for all tools
- [ ] Parallel execution achieves 40%+ speedup
- [ ] Analytics dashboard operational
- [ ] 99%+ reliability with automatic recovery

### Code Quality Validation

- [ ] Follows existing service patterns
- [ ] All tools async-compatible
- [ ] Proper error handling and logging
- [ ] Sandboxing prevents system access
- [ ] JSON-serializable results
- [ ] Tool schemas follow function calling format

### Documentation & Deployment

- [ ] Environment variables documented
- [ ] Tool schemas documented
- [ ] Usage examples for each tool
- [ ] Resource limits configured
- [ ] Analytics endpoints documented

---

## Anti-Patterns to Avoid

- ❌ Don't use synchronous file operations (use aiofiles)
- ❌ Don't skip parameter validation (always validate against schema)
- ❌ Don't ignore resource limits (enforce CPU/memory limits)
- ❌ Don't allow unbounded parallelism (use semaphores)
- ❌ Don't return non-serializable objects (convert to dict)
- ❌ Don't hardcode timeouts (make configurable)
- ❌ Don't skip retry logic (implement exponential backoff)
- ❌ Don't allow file access outside sandbox
- ❌ Don't forget to track resource usage
- ❌ Don't mix tool logic with agent logic
