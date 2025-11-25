# PRP-02: Advanced State Management with LangGraph

## Goal

**Feature Goal**: Migrate from basic agent orchestration to LangGraph-based workflows with stateful agent management, checkpoint/resume capabilities, graph-based workflow representation, rollback mechanisms, and human-in-the-loop intervention points.

**Deliverable**: Complete LangGraph workflow implementation for the 7-agent swarm system (Coordinator, Planner, Architect, Implementer, Tester, Validator, Debugger) with state persistence, conditional edges, rollback capabilities, and human-in-the-loop checkpoints.

**Success Definition**:
- All 7 agents functioning as LangGraph nodes with proper state management
- Checkpoint/resume working across sessions with <2s recovery time
- Rollback to any previous state within workflow history
- Human-in-the-loop approval gates at critical decision points
- Graph visualization and debugging capabilities operational
- 90%+ state recovery accuracy after interruptions

## Why

- Current system lacks proper workflow orchestration between agents
- No state persistence across sessions, losing progress on interruptions
- No rollback capabilities when agents make poor decisions
- Missing human oversight at critical decision points
- Limited debugging and workflow visualization
- No proper graph-based representation of complex agent interactions

## What

Implement a comprehensive LangGraph-based workflow system that manages state across all 7 agents, enables checkpoint/resume functionality, provides rollback capabilities, and supports human intervention at configurable points in the workflow.

### Success Criteria

- [ ] All 7 agents integrated as LangGraph nodes
- [ ] State persistence with <2s checkpoint save/load time
- [ ] Rollback to any checkpoint within last 100 states
- [ ] Human-in-the-loop gates functional at 5 key decision points
- [ ] Graph visualization generates Mermaid diagrams
- [ ] 95%+ test coverage on critical paths

## All Needed Context

### Context Completeness Check

_If someone knew nothing about this codebase, would they have everything needed to implement this successfully?_ YES - This PRP includes complete LangGraph patterns, state schemas, agent definitions, and integration examples with the existing memory system.

### Documentation & References

```yaml
- url: https://langchain-ai.github.io/langgraph/how-tos/persistence/
  why: LangGraph persistence patterns and checkpoint configuration
  critical: Checkpointer must be passed to compile(), not invoke()
  section: Setting up checkpointers

- url: https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/
  why: Human-in-the-loop patterns for approval gates
  critical: Use interrupt_before for human approval points
  section: Breakpoints and approval patterns

- url: https://langchain-ai.github.io/langgraph/how-tos/subgraph/
  why: Subgraph patterns for complex agent workflows
  critical: Subgraphs help organize complex multi-agent flows
  
- url: https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.StateGraph
  why: StateGraph API reference for conditional edges and state management
  critical: TypedDict for state schema, add_conditional_edges for branching

- file: src/services/checkpoint_service.py
  why: Existing checkpoint manager from PRP-01 to integrate
  pattern: CheckpointManager class, get_checkpointer() method
  gotcha: Redis checkpointer already configured, reuse don't recreate

- file: src/services/memory_service.py
  why: Memory orchestrator to integrate with state management
  pattern: MemoryOrchestrator facade, store_memory and retrieve methods
  gotcha: Async operations throughout, maintain consistency
```

### Current Codebase Tree

```bash
agent-swarm/
├── src/
│   ├── agents/           # Minimal implementation currently
│   │   └── __init__.py
│   ├── core/             # Empty, needs workflow implementation
│   │   └── __init__.py
│   ├── memory/           # Complete from PRP-01
│   │   ├── base.py
│   │   ├── cache.py
│   │   ├── global_memory.py
│   │   ├── hybrid.py
│   │   ├── project.py
│   │   └── working.py
│   ├── models/
│   │   ├── checkpoint_models.py
│   │   └── memory_models.py
│   ├── services/
│   │   ├── checkpoint_service.py  # Has basic checkpointer
│   │   ├── memory_service.py
│   │   └── rag_service.py
│   ├── config/
│   │   └── memory_config.py
│   └── tests/
│       └── memory/
├── docker/
│   └── docker-compose.yml
└── requirements.txt
```

### Desired Codebase Tree with Files to be Added

```bash
agent-swarm/
├── src/
│   ├── agents/                       # EXPAND: Individual agent implementations
│   │   ├── __init__.py              # MODIFY: Export all agents
│   │   ├── base.py                  # NEW: BaseAgent abstract class
│   │   ├── coordinator.py           # NEW: Coordinator agent (orchestrates workflow)
│   │   ├── planner.py              # NEW: Planner agent (task decomposition)
│   │   ├── architect.py            # NEW: Architect agent (system design)
│   │   ├── implementer.py          # NEW: Implementer agent (code generation)
│   │   ├── tester.py               # NEW: Tester agent (test creation/execution)
│   │   ├── validator.py            # NEW: Validator agent (quality checks)
│   │   └── debugger.py             # NEW: Debugger agent (error resolution)
│   ├── core/
│   │   ├── __init__.py              # MODIFY: Export workflow components
│   │   ├── state.py                 # NEW: AgentState and state schemas
│   │   ├── workflow.py              # NEW: Main LangGraph workflow
│   │   ├── edges.py                 # NEW: Conditional edge logic
│   │   ├── nodes.py                 # NEW: Node wrapper functions
│   │   └── checkpoints.py           # NEW: Enhanced checkpoint management
│   ├── graphs/                      # NEW: Graph-specific implementations
│   │   ├── __init__.py
│   │   ├── main_graph.py            # NEW: Primary agent workflow graph
│   │   ├── planning_subgraph.py    # NEW: Planning phase subgraph
│   │   ├── implementation_subgraph.py # NEW: Implementation subgraph
│   │   └── validation_subgraph.py  # NEW: Validation/testing subgraph
│   ├── models/
│   │   ├── state_models.py          # NEW: State-related data models
│   │   ├── agent_models.py          # NEW: Agent request/response models
│   │   └── workflow_models.py       # NEW: Workflow configuration models
│   ├── services/
│   │   ├── workflow_service.py      # NEW: High-level workflow orchestration
│   │   ├── rollback_service.py      # NEW: State rollback management
│   │   └── human_approval_service.py # NEW: Human-in-the-loop service
│   ├── utils/
│   │   ├── __init__.py              # MODIFY: Export utilities
│   │   ├── visualization.py         # NEW: Graph visualization (Mermaid)
│   │   └── state_utils.py           # NEW: State manipulation utilities
│   ├── tests/
│   │   ├── agents/                  # NEW: Agent-specific tests
│   │   ├── core/                    # NEW: Workflow and state tests
│   │   └── graphs/                  # NEW: Graph integration tests
│   └── main.py                      # NEW: Main application entry point
```

### Known Gotchas of our Codebase & Library Quirks

```python
# CRITICAL: LangGraph requires specific patterns for state management
# StateGraph expects TypedDict for state schema, not Pydantic models directly

# CRITICAL: Checkpointer must be passed at compile time
# graph.compile(checkpointer=checkpointer) NOT during invoke

# CRITICAL: Conditional edges must return next node name as string
# def route(state): return "next_node" NOT return next_node_function

# CRITICAL: Human-in-the-loop requires interrupt_before configuration
# graph.add_node("approval", node, interrupt_before=True)

# CRITICAL: State updates are immutable, return new state dict
# return {"messages": state["messages"] + [new_message]} NOT state["messages"].append()

# CRITICAL: Async nodes require proper async context
# All agent execute methods must be async def execute(state) -> Dict

# CRITICAL: Memory integration needs proper async context management
# Use async with for memory operations within nodes

# CRITICAL: Rollback requires versioned state management
# Store state version in metadata for proper rollback tracking
```

## Implementation Blueprint

### Data Models and Structure

```python
# src/models/state_models.py
from typing import Dict, List, Any, Optional, TypedDict
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field

class AgentRole(str, Enum):
    """Agent roles in the workflow."""
    COORDINATOR = "coordinator"
    PLANNER = "planner"
    ARCHITECT = "architect"
    IMPLEMENTER = "implementer"
    TESTER = "tester"
    VALIDATOR = "validator"
    DEBUGGER = "debugger"

class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    PLANNING = "planning"
    DESIGNING = "designing"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    VALIDATING = "validating"
    DEBUGGING = "debugging"
    COMPLETE = "complete"
    FAILED = "failed"
    HUMAN_REVIEW = "human_review"

# TypedDict for LangGraph state (required format)
class AgentState(TypedDict, total=False):
    """
    Main state schema for LangGraph workflow.
    Uses TypedDict as required by StateGraph.
    """
    # Core workflow state
    workflow_id: str
    project_id: str
    session_id: str
    status: str  # WorkflowStatus value
    
    # Agent communication
    messages: List[Dict[str, Any]]
    current_agent: str  # AgentRole value
    next_agent: Optional[str]
    
    # Task management
    task: Dict[str, Any]
    subtasks: List[Dict[str, Any]]
    completed_subtasks: List[str]
    
    # Results and artifacts
    plan: Optional[Dict[str, Any]]
    architecture: Optional[Dict[str, Any]]
    implementation: Optional[Dict[str, Any]]
    test_results: Optional[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]]
    debug_info: Optional[Dict[str, Any]]
    
    # Control flow
    retry_count: int
    max_retries: int
    should_continue: bool
    requires_human_approval: bool
    human_feedback: Optional[str]
    
    # Checkpoint and rollback
    checkpoint_id: str
    parent_checkpoint_id: Optional[str]
    state_version: int
    
    # Memory references
    memory_ids: List[str]
    context: Dict[str, Any]
    
    # Metadata
    created_at: str  # ISO format datetime
    updated_at: str
    elapsed_time: float
    token_count: int
    cost: float

# src/models/agent_models.py
class AgentRequest(BaseModel):
    """Request model for agent execution."""
    state: Dict[str, Any] = Field(description="Current workflow state")
    task: Dict[str, Any] = Field(description="Task to execute")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    memory_ids: List[str] = Field(default_factory=list, description="Relevant memory IDs")

class AgentResponse(BaseModel):
    """Response model from agent execution."""
    success: bool = Field(description="Execution success status")
    result: Dict[str, Any] = Field(description="Execution result")
    next_agent: Optional[AgentRole] = Field(default=None, description="Suggested next agent")
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="New messages")
    state_updates: Dict[str, Any] = Field(default_factory=dict, description="State updates")
    requires_human_approval: bool = Field(default=False, description="Needs human review")
    error: Optional[str] = Field(default=None, description="Error message if failed")

# src/models/workflow_models.py
class WorkflowConfig(BaseModel):
    """Workflow configuration model."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    enable_human_approval: bool = Field(default=True)
    approval_points: List[AgentRole] = Field(
        default_factory=lambda: [AgentRole.PLANNER, AgentRole.ARCHITECT, AgentRole.VALIDATOR]
    )
    max_retries: int = Field(default=3)
    timeout_seconds: int = Field(default=3600)
    checkpoint_interval: int = Field(default=5, description="Checkpoint every N nodes")
    enable_rollback: bool = Field(default=True)
    max_rollback_states: int = Field(default=100)
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/models/state_models.py
  - IMPLEMENT: AgentState TypedDict, WorkflowStatus, AgentRole enums
  - FOLLOW pattern: LangGraph TypedDict requirements for state
  - NAMING: AgentState for main state, WorkflowStatus for status tracking
  - PLACEMENT: Models layer for state definitions

Task 2: CREATE src/agents/base.py
  - IMPLEMENT: BaseAgent abstract class with execute method
  - FOLLOW pattern: Abstract base class with async execute signature
  - NAMING: BaseAgent, execute(state: AgentState) -> AgentResponse
  - PLACEMENT: Agents module base class

Task 3: CREATE src/agents/coordinator.py
  - IMPLEMENT: CoordinatorAgent orchestrating overall workflow
  - FOLLOW pattern: Inherit from BaseAgent, implement execute
  - NAMING: CoordinatorAgent class, route_to_agent method
  - DEPENDENCIES: BaseAgent, AgentState, memory_service
  - PLACEMENT: Agents module

Task 4: CREATE src/agents/[planner|architect|implementer|tester|validator|debugger].py
  - IMPLEMENT: Individual agent implementations
  - FOLLOW pattern: BaseAgent inheritance, async execute methods
  - NAMING: {Role}Agent classes matching AgentRole enum
  - DEPENDENCIES: BaseAgent, memory integration
  - PLACEMENT: One file per agent in agents module

Task 5: CREATE src/core/state.py
  - IMPLEMENT: State management utilities and state factory
  - FOLLOW pattern: State initialization, validation, serialization
  - NAMING: create_initial_state, validate_state, serialize_state
  - DEPENDENCIES: AgentState, state_models
  - PLACEMENT: Core module for state management

Task 6: CREATE src/core/nodes.py
  - IMPLEMENT: Node wrapper functions for LangGraph integration
  - FOLLOW pattern: Async functions wrapping agent execute methods
  - NAMING: {agent}_node functions for each agent
  - DEPENDENCIES: All agent classes, state management
  - PLACEMENT: Core module for node definitions

Task 7: CREATE src/core/edges.py
  - IMPLEMENT: Conditional edge logic for workflow routing
  - FOLLOW pattern: Router functions returning next node names
  - NAMING: route_from_{agent} functions, should_retry, needs_human_approval
  - DEPENDENCIES: AgentState, workflow logic
  - PLACEMENT: Core module for edge logic

Task 8: CREATE src/core/workflow.py
  - IMPLEMENT: Main LangGraph workflow using StateGraph
  - FOLLOW pattern: StateGraph construction with nodes and edges
  - NAMING: create_workflow, compile_graph methods
  - DEPENDENCIES: Nodes, edges, checkpoint_service
  - PLACEMENT: Core module for main workflow

Task 9: CREATE src/graphs/main_graph.py
  - IMPLEMENT: Primary workflow graph orchestrating all agents
  - FOLLOW pattern: StateGraph with all agents as nodes
  - NAMING: MainWorkflowGraph class, build() method
  - DEPENDENCIES: All nodes, edges, state management
  - PLACEMENT: Graphs module for main workflow

Task 10: CREATE src/services/workflow_service.py
  - IMPLEMENT: High-level workflow orchestration service
  - FOLLOW pattern: Service pattern with run, pause, resume methods
  - NAMING: WorkflowOrchestrator class, execute_workflow method
  - DEPENDENCIES: Main graph, checkpoint service, memory service
  - PLACEMENT: Services layer

Task 11: CREATE src/services/rollback_service.py
  - IMPLEMENT: State rollback management service
  - FOLLOW pattern: Version tracking, state restoration
  - NAMING: RollbackManager, rollback_to_checkpoint, list_checkpoints
  - DEPENDENCIES: Checkpoint service, state management
  - PLACEMENT: Services layer for rollback

Task 12: CREATE src/services/human_approval_service.py
  - IMPLEMENT: Human-in-the-loop approval management
  - FOLLOW pattern: Approval queue, feedback integration
  - NAMING: HumanApprovalService, request_approval, submit_feedback
  - DEPENDENCIES: Workflow state, notification system
  - PLACEMENT: Services layer

Task 13: CREATE src/utils/visualization.py
  - IMPLEMENT: Graph visualization generating Mermaid diagrams
  - FOLLOW pattern: Graph traversal, Mermaid syntax generation
  - NAMING: generate_mermaid_diagram, visualize_workflow
  - DEPENDENCIES: Graph structure, state
  - PLACEMENT: Utils module

Task 14: MODIFY src/services/checkpoint_service.py
  - ENHANCE: Add versioning, rollback support, state history
  - FIND pattern: Existing CheckpointManager
  - ADD: get_checkpoint_history, rollback_to methods
  - PRESERVE: Existing checkpointer functionality

Task 15: CREATE src/main.py
  - IMPLEMENT: Main application entry point
  - FOLLOW pattern: FastAPI or CLI interface
  - NAMING: app (FastAPI) or main() for CLI
  - DEPENDENCIES: All services, workflow orchestrator
  - PLACEMENT: Root of src directory

Task 16: CREATE src/tests/core/test_workflow.py
  - IMPLEMENT: Comprehensive workflow tests
  - FOLLOW pattern: pytest-asyncio, mock agents
  - COVERAGE: State transitions, checkpoints, rollback
  - PLACEMENT: Tests for core workflow

Task 17: CREATE src/tests/agents/test_*.py
  - IMPLEMENT: Unit tests for each agent
  - FOLLOW pattern: Mock state, verify outputs
  - COVERAGE: Each agent's execute method, error handling
  - PLACEMENT: One test file per agent
```

### Implementation Patterns & Key Details

```python
# LangGraph StateGraph pattern
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver

def create_workflow(checkpointer: RedisSaver) -> CompiledGraph:
    """
    PATTERN: Build graph with nodes and edges, compile with checkpointer
    CRITICAL: Pass checkpointer to compile(), not invoke()
    """
    # Create graph with state schema
    graph = StateGraph(AgentState)
    
    # Add nodes (agents)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("planner", planner_node, interrupt_before=True)  # Human approval
    graph.add_node("architect", architect_node, interrupt_before=True)
    graph.add_node("implementer", implementer_node)
    graph.add_node("tester", tester_node)
    graph.add_node("validator", validator_node, interrupt_before=True)
    graph.add_node("debugger", debugger_node)
    
    # Add conditional edges
    graph.add_conditional_edges(
        "coordinator",
        route_from_coordinator,  # Returns next node name
        {
            "planner": "planner",
            "architect": "architect",
            "implementer": "implementer",
            "end": END
        }
    )
    
    # Set entry point
    graph.set_entry_point("coordinator")
    
    # Compile with checkpointer
    return graph.compile(checkpointer=checkpointer)

# Agent node wrapper pattern
async def coordinator_node(state: AgentState) -> Dict[str, Any]:
    """
    PATTERN: Node wrapper for agent execution
    CRITICAL: Must return state updates, not mutate state
    """
    agent = CoordinatorAgent(memory_service=memory_orchestrator)
    response = await agent.execute(state)
    
    # Return state updates (immutable)
    return {
        "messages": state.get("messages", []) + response.messages,
        "current_agent": response.next_agent,
        "status": response.result.get("status", state.get("status")),
        **response.state_updates
    }

# Conditional edge pattern
def route_from_coordinator(state: AgentState) -> str:
    """
    PATTERN: Router function for conditional edges
    CRITICAL: Must return string node name
    """
    if state.get("status") == "planning":
        return "planner"
    elif state.get("status") == "designing":
        return "architect"
    elif state.get("requires_human_approval"):
        return "human_review"
    else:
        return "implementer"

# Rollback pattern
async def rollback_to_checkpoint(
    workflow_id: str,
    checkpoint_id: str,
    graph: CompiledGraph
) -> AgentState:
    """
    PATTERN: Restore state from checkpoint
    GOTCHA: Must reload graph with checkpoint config
    """
    config = {
        "configurable": {
            "thread_id": workflow_id,
            "checkpoint_id": checkpoint_id
        }
    }
    
    # Get state at checkpoint
    state = await graph.aget_state(config)
    return state.values
```

### Integration Points

```yaml
LANGGRAPH:
  - dependency: "langgraph>=0.0.20"
  - checkpoint: "Redis-based persistence already configured"
  - visualization: "Built-in graph.get_graph().draw_mermaid()"

MEMORY:
  - integration: "Use MemoryOrchestrator from PRP-01"
  - pattern: "Store agent decisions and context in project memory"
  - checkpoint: "Link memory IDs to checkpoint metadata"

CONFIG:
  - add to: .env
  - variables: |
      WORKFLOW_TIMEOUT=3600
      ENABLE_HUMAN_APPROVAL=true
      MAX_ROLLBACK_STATES=100
      CHECKPOINT_INTERVAL=5

FASTAPI:
  - add to: requirements.txt
  - endpoints: |
      POST /workflow/start
      GET /workflow/{id}/status
      POST /workflow/{id}/approve
      POST /workflow/{id}/rollback
      GET /workflow/{id}/graph
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# Check each new file
ruff check src/agents/ src/core/ src/graphs/ --fix
mypy src/agents/ src/core/ src/graphs/ --strict
ruff format src/

# Verify imports
python -c "from src.core.workflow import create_workflow; print('Core imports OK')"
python -c "from src.agents import *; print('Agent imports OK')"

# Expected: Zero errors, all imports resolve
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test individual agents
pytest src/tests/agents/test_coordinator.py -v
pytest src/tests/agents/test_planner.py -v
pytest src/tests/agents/ -v --cov=src/agents

# Test core workflow components
pytest src/tests/core/test_state.py -v
pytest src/tests/core/test_workflow.py -v
pytest src/tests/core/test_edges.py -v

# Test services
pytest src/tests/services/test_workflow_service.py -v
pytest src/tests/services/test_rollback_service.py -v

# Full test suite
pytest src/tests/ -v --cov=src --cov-report=term-missing

# Expected: 95%+ coverage on critical paths
```

### Level 3: Integration Testing (System Validation)

```bash
# Start required services
docker-compose -f docker/docker-compose.yml up -d
sleep 5

# Test workflow creation and compilation
python -c "
from src.core.workflow import create_workflow
from src.services.checkpoint_service import CheckpointManager
from src.config.memory_config import MemoryConfig

config = MemoryConfig()
checkpoint_manager = CheckpointManager(config)
graph = create_workflow(checkpoint_manager.get_checkpointer())
print('Workflow compiled successfully')
"

# Test basic workflow execution
python scripts/test_workflow.py \
  --task "Create a simple calculator" \
  --agents all \
  --timeout 60

# Test checkpoint and resume
python scripts/test_checkpoint_resume.py \
  --workflow-id test-123 \
  --interrupt-after planner \
  --resume

# Test rollback functionality
python scripts/test_rollback.py \
  --workflow-id test-123 \
  --rollback-to checkpoint-2

# Expected: All workflows execute, checkpoint/resume works
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Graph Visualization Test
python scripts/visualize_workflow.py \
  --output workflow.mermaid \
  --include-subgraphs

# Verify Mermaid output
cat workflow.mermaid | head -20
# Expected: Valid Mermaid graph syntax

# Human-in-the-Loop Simulation
python scripts/test_human_approval.py \
  --workflow-id approval-test \
  --auto-approve false \
  --approval-delay 5

# Expected: Workflow pauses at approval points

# Complex Multi-Agent Workflow
python scripts/test_complex_workflow.py \
  --task "Build a REST API with authentication" \
  --enable-all-agents \
  --max-iterations 50 \
  --measure-performance

# Expected: All agents collaborate successfully

# State Recovery After Crash
python scripts/test_crash_recovery.py \
  --simulate-crash-after 10 \
  --recover-and-continue

# Expected: Workflow resumes from last checkpoint

# Performance Benchmark
python scripts/benchmark_workflow.py \
  --tasks 10 \
  --parallel false \
  --measure-latency \
  --measure-throughput

# Expected: <2s checkpoint save/load, <500ms state transitions

# Rollback Stress Test
python scripts/test_rollback_stress.py \
  --create-checkpoints 100 \
  --random-rollbacks 20 \
  --verify-state-integrity

# Expected: All rollbacks successful, state integrity maintained
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] All agents functioning as LangGraph nodes
- [ ] State persistence working with Redis checkpointer
- [ ] Rollback functionality operational
- [ ] Human-in-the-loop gates functional
- [ ] 95%+ test coverage on critical paths

### Feature Validation

- [ ] 7 agents integrated and communicating
- [ ] Checkpoint/resume < 2s recovery time
- [ ] Rollback to any of last 100 states works
- [ ] Human approval at 5 decision points
- [ ] Graph visualization generates valid Mermaid
- [ ] State recovery accuracy > 95%

### Code Quality Validation

- [ ] Follows LangGraph patterns (TypedDict, compile with checkpointer)
- [ ] Proper async/await usage throughout
- [ ] Error handling and retry logic implemented
- [ ] Memory integration from PRP-01 utilized
- [ ] Clear separation of concerns (agents, nodes, edges)
- [ ] Comprehensive logging and monitoring

### Documentation & Deployment

- [ ] API endpoints documented (if FastAPI)
- [ ] Workflow configuration documented
- [ ] Agent responsibilities clearly defined
- [ ] Rollback procedures documented
- [ ] Human approval process documented

---

## Anti-Patterns to Avoid

- ❌ Don't mutate state directly (return new state dict)
- ❌ Don't pass checkpointer to invoke (pass to compile)
- ❌ Don't use Pydantic models directly for state (use TypedDict)
- ❌ Don't return node functions from routers (return node names)
- ❌ Don't skip interrupt_before for human approval points
- ❌ Don't create synchronous agent methods (all async)
- ❌ Don't ignore state versioning for rollback
- ❌ Don't checkpoint on every node (use intervals)
- ❌ Don't store large objects in state (use memory references)