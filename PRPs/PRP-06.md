# PRP-06: Fractal Task Decomposition System

## Goal

**Feature Goal**: Implement a fractal task decomposition system that recursively breaks down complex problems into manageable subtasks, with dependency management, complexity estimation, model assignment, and progress tracking capabilities integrated with the 7-agent workflow.

**Deliverable**: Task decomposition service with recursive planning algorithms, dependency graph management, complexity inheritance, granular task assignment to appropriate models/agents, and hierarchical progress rollup integrated with the existing Planner agent.

**Success Definition**:
- Decompose complex tasks into subtasks with <500ms processing time
- Generate dependency graphs with topological sorting for execution order
- Estimate complexity accurately (85%+ correlation with actual execution complexity)
- Assign tasks to optimal models based on subtask complexity (60%+ cost savings)
- Track progress hierarchically with real-time rollup to parent tasks
- Support 10+ levels of decomposition depth without performance degradation
- Handle circular dependency detection and resolution

## Why

- Current planner agent has placeholder logic for task decomposition (_create_plan method)
- No way to break down complex 500-1000 line features into manageable chunks
- Missing dependency tracking between subtasks for optimal execution ordering
- No complexity inheritance from parent to child tasks
- Cannot assign different models to subtasks based on their complexity
- No hierarchical progress tracking for long-running workflows
- Critical for achieving autonomous feature completion and cost optimization goals

## What

Implement a comprehensive fractal task decomposition system that recursively analyzes and breaks down complex tasks into atomic, executable subtasks with proper dependency management, complexity estimation, and progress tracking, enabling the agent swarm to tackle large features autonomously.

### Success Criteria

- [ ] Complex tasks decomposed into executable subtasks recursively
- [ ] Dependency graphs generated with no circular dependencies
- [ ] Subtask complexity estimated with 85%+ accuracy
- [ ] Tasks routed to appropriate models based on decomposition
- [ ] Progress tracked and rolled up through task hierarchy
- [ ] 10+ decomposition levels supported efficiently

## All Needed Context

### Context Completeness Check

_If someone knew nothing about this codebase, would they have everything needed to implement this successfully?_ YES - This PRP includes complete fractal decomposition algorithms, dependency management patterns, progress tracking methods, and integration with existing systems.

### Documentation & References

```yaml
- url: https://en.wikipedia.org/wiki/Work_breakdown_structure
  why: WBS concepts for hierarchical task decomposition
  critical: Shows hierarchical decomposition patterns and dependency tracking
  section: Structure and levels

- url: https://networkx.org/documentation/stable/reference/algorithms/dag.html
  why: Directed Acyclic Graph algorithms for dependency management
  critical: topological_sort, is_directed_acyclic_graph functions for dependency ordering
  
- url: https://docs.python.org/3/library/graphlib.html#graphlib.TopologicalSorter
  why: Built-in Python topological sorting for task ordering
  critical: Efficient dependency resolution without external libraries

- file: src/agents/planner.py
  why: Existing planner agent that will use decomposition service
  pattern: BaseAgent inheritance, execute method, _create_plan placeholder
  gotcha: Must maintain async execution, state immutability

- file: src/llm/analyzer.py
  why: Task complexity analyzer for complexity estimation
  pattern: TaskComplexityAnalyzer, analyze_task method
  gotcha: <100ms performance requirement

- file: src/models/state_models.py
  why: AgentState TypedDict with subtasks field
  pattern: subtasks, completed_subtasks lists in state
  gotcha: TypedDict required for LangGraph, not Pydantic

- file: src/services/llm_service.py
  why: LLM service for generating decomposition suggestions
  pattern: Service facade, generate_response method
  gotcha: Use appropriate model based on decomposition complexity

- file: src/analysis/complexity_analyzer.py
  why: Code complexity metrics that inform task complexity
  pattern: ComplexityAnalyzer for estimating implementation difficulty
  gotcha: Different metrics for different task types
```

### Current Codebase Tree

```bash
agent-swarm/
├── src/
│   ├── agents/
│   │   ├── planner.py           # Has _create_plan placeholder
│   │   └── base.py              # BaseAgent class
│   ├── llm/
│   │   ├── analyzer.py          # TaskComplexityAnalyzer
│   │   └── router.py            # Model routing logic
│   ├── models/
│   │   ├── state_models.py      # Has subtasks field
│   │   └── routing_models.py    # TaskAnalysis model
│   ├── services/
│   │   ├── llm_service.py       # LLM orchestration
│   │   └── workflow_service.py  # Workflow management
│   └── analysis/
│       └── complexity_analyzer.py # Code complexity
```

### Desired Codebase Tree with Files to be Added

```bash
agent-swarm/
├── src/
│   ├── decomposition/                    # NEW: Task decomposition subsystem
│   │   ├── __init__.py                  # Export main interfaces
│   │   ├── base.py                      # BaseDecomposer abstract class
│   │   ├── fractal_decomposer.py        # Recursive decomposition engine
│   │   ├── dependency_manager.py         # Dependency graph management
│   │   ├── complexity_estimator.py       # Subtask complexity estimation
│   │   ├── task_assigner.py             # Model/agent assignment logic
│   │   ├── progress_tracker.py          # Hierarchical progress tracking
│   │   └── strategies/                  # Decomposition strategies
│   │       ├── __init__.py
│   │       ├── code_strategy.py         # Code-specific decomposition
│   │       ├── research_strategy.py     # Research task decomposition
│   │       ├── testing_strategy.py      # Test creation decomposition
│   │       └── refactor_strategy.py     # Refactoring decomposition
│   ├── models/
│   │   └── decomposition_models.py      # NEW: Decomposition-related models
│   ├── services/
│   │   └── decomposition_service.py     # NEW: High-level decomposition service
│   ├── agents/
│   │   └── planner.py                   # MODIFY: Integrate decomposition
│   └── tests/
│       └── decomposition/               # NEW: Decomposition tests
│           ├── test_fractal.py
│           ├── test_dependencies.py
│           ├── test_complexity.py
│           └── test_progress.py
```

### Known Gotchas of our Codebase & Library Quirks

```python
# CRITICAL: Task decomposition must be deterministic for checkpointing
# Same input should produce same decomposition for rollback

# CRITICAL: Subtask IDs must be unique across entire workflow
# Use hierarchical naming: parent_id.subtask_1.subtask_1_1

# CRITICAL: Circular dependencies must be detected and prevented
# Use topological sort to verify DAG property

# CRITICAL: Progress updates must be atomic to prevent race conditions
# Use Redis transactions for progress updates

# CRITICAL: Decomposition depth must be limited to prevent stack overflow
# Default max depth: 10 levels

# CRITICAL: Model assignment must respect cost constraints
# Check remaining budget before assigning expensive models

# CRITICAL: State updates for subtasks must maintain immutability
# Return new state dict, don't modify in place

# CRITICAL: Memory usage grows exponentially with depth
# Implement streaming/pagination for large decompositions
```

## Implementation Blueprint

### Data Models and Structure

```python
# src/models/decomposition_models.py
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
from uuid import uuid4

class TaskType(str, Enum):
    """Types of tasks for strategy selection."""
    CODE_GENERATION = "code_generation"
    CODE_MODIFICATION = "code_modification"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    ARCHITECTURE = "architecture"
    ANALYSIS = "analysis"

class TaskStatus(str, Enum):
    """Status of a task in the decomposition tree."""
    PENDING = "pending"
    DECOMPOSING = "decomposing"
    READY = "ready"           # Ready for execution (leaf node)
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"        # Waiting on dependencies

class Task(BaseModel):
    """Represents a task in the decomposition tree."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique task ID")
    parent_id: Optional[str] = Field(default=None, description="Parent task ID")
    name: str = Field(description="Task name/title")
    description: str = Field(description="Detailed task description")
    type: TaskType = Field(description="Task type for strategy selection")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    
    # Decomposition metadata
    depth: int = Field(default=0, description="Depth in decomposition tree")
    is_leaf: bool = Field(default=False, description="Whether task is atomic")
    can_decompose: bool = Field(default=True, description="Whether further decomposition possible")
    
    # Complexity and assignment
    estimated_complexity: float = Field(default=0.5, ge=0, le=1, description="Estimated complexity")
    actual_complexity: Optional[float] = Field(default=None, description="Actual complexity after execution")
    assigned_model: Optional[str] = Field(default=None, description="Assigned model/agent")
    estimated_tokens: int = Field(default=0, description="Estimated token usage")
    estimated_cost: float = Field(default=0.0, description="Estimated cost in USD")
    
    # Dependencies
    dependencies: Set[str] = Field(default_factory=set, description="Task IDs this depends on")
    dependents: Set[str] = Field(default_factory=set, description="Task IDs that depend on this")
    
    # Progress tracking
    progress: float = Field(default=0.0, ge=0, le=100, description="Progress percentage")
    subtask_count: int = Field(default=0, description="Number of subtasks")
    completed_subtask_count: int = Field(default=0, description="Completed subtasks")
    
    # Execution metadata
    start_time: Optional[datetime] = Field(default=None)
    end_time: Optional[datetime] = Field(default=None)
    execution_time_ms: Optional[int] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    retry_count: int = Field(default=0)
    
    # Results and artifacts
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task execution result")
    artifacts: List[str] = Field(default_factory=list, description="Generated artifact IDs")

class DecompositionTree(BaseModel):
    """Represents the entire task decomposition tree."""
    root_task_id: str = Field(description="Root task ID")
    tasks: Dict[str, Task] = Field(default_factory=dict, description="All tasks by ID")
    execution_order: List[str] = Field(default_factory=list, description="Topological sort order")
    
    # Tree metadata
    total_tasks: int = Field(default=1)
    max_depth: int = Field(default=0)
    leaf_tasks: List[str] = Field(default_factory=list, description="Executable leaf task IDs")
    
    # Progress tracking
    overall_progress: float = Field(default=0.0, description="Overall progress percentage")
    completed_tasks: int = Field(default=0)
    failed_tasks: int = Field(default=0)
    blocked_tasks: int = Field(default=0)
    
    # Cost tracking
    estimated_total_cost: float = Field(default=0.0)
    actual_total_cost: float = Field(default=0.0)
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class DecompositionStrategy(BaseModel):
    """Strategy for decomposing specific task types."""
    name: str = Field(description="Strategy name")
    task_type: TaskType = Field(description="Task type this strategy handles")
    max_depth: int = Field(default=5, description="Maximum decomposition depth")
    min_complexity_for_decomposition: float = Field(default=0.3, description="Minimum complexity to decompose")
    decomposition_patterns: List[str] = Field(default_factory=list, description="Patterns for decomposition")

class DecompositionRequest(BaseModel):
    """Request to decompose a task."""
    task_description: str = Field(description="High-level task description")
    task_type: Optional[TaskType] = Field(default=None, description="Force specific task type")
    max_depth: int = Field(default=5, ge=1, le=10)
    max_subtasks_per_level: int = Field(default=5, ge=2, le=10)
    complexity_threshold: float = Field(default=0.2, ge=0, le=1, description="Min complexity to decompose")
    include_dependencies: bool = Field(default=True)
    estimate_costs: bool = Field(default=True)
    target_model_routing: bool = Field(default=True, description="Route subtasks to specific models")

class DecompositionResult(BaseModel):
    """Result of task decomposition."""
    tree: DecompositionTree = Field(description="Complete decomposition tree")
    execution_plan: List[List[str]] = Field(
        default_factory=list,
        description="Batches of tasks that can run in parallel"
    )
    estimated_duration_ms: int = Field(description="Estimated total execution time")
    warnings: List[str] = Field(default_factory=list, description="Decomposition warnings")
    
class ProgressUpdate(BaseModel):
    """Progress update for a task."""
    task_id: str = Field(description="Task being updated")
    progress: float = Field(ge=0, le=100, description="New progress percentage")
    status: TaskStatus = Field(description="New status")
    message: Optional[str] = Field(default=None, description="Progress message")
    partial_result: Optional[Dict[str, Any]] = Field(default=None, description="Partial results")
    
class DependencyValidation(BaseModel):
    """Result of dependency validation."""
    is_valid: bool = Field(description="Whether dependencies form valid DAG")
    has_cycles: bool = Field(default=False, description="Whether circular dependencies exist")
    cycles: List[List[str]] = Field(default_factory=list, description="Circular dependency chains")
    missing_dependencies: List[str] = Field(default_factory=list, description="Referenced but missing tasks")
    execution_order: List[str] = Field(default_factory=list, description="Valid execution order if DAG")
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/decomposition/base.py
  - IMPLEMENT: BaseDecomposer abstract class
  - FOLLOW pattern: src/tools/base.py abstract pattern
  - NAMING: BaseDecomposer, decompose, estimate_complexity methods
  - PLACEMENT: Decomposition subsystem base module

Task 2: CREATE src/models/decomposition_models.py
  - IMPLEMENT: Task, DecompositionTree, and related models
  - FOLLOW pattern: Pydantic models with validation
  - NAMING: Models as specified in data models section
  - PLACEMENT: Models directory

Task 3: CREATE src/decomposition/complexity_estimator.py
  - IMPLEMENT: ComplexityEstimator for subtask complexity
  - FOLLOW pattern: src/llm/analyzer.py complexity patterns
  - NAMING: ComplexityEstimator, estimate_task_complexity, propagate_complexity
  - DEPENDENCIES: TaskComplexityAnalyzer, Task model
  - PLACEMENT: Decomposition subsystem

Task 4: CREATE src/decomposition/dependency_manager.py
  - IMPLEMENT: DependencyManager for DAG management
  - FOLLOW pattern: graphlib.TopologicalSorter usage
  - NAMING: DependencyManager, add_dependency, validate_dependencies, get_execution_order
  - DEPENDENCIES: graphlib, Task model
  - PLACEMENT: Decomposition subsystem

Task 5: CREATE src/decomposition/strategies/code_strategy.py
  - IMPLEMENT: CodeDecompositionStrategy for code tasks
  - FOLLOW pattern: BaseDecomposer inheritance
  - NAMING: CodeDecompositionStrategy, specific decomposition rules
  - DEPENDENCIES: BaseDecomposer, AST analysis service
  - PLACEMENT: Strategies module

Task 6: CREATE src/decomposition/strategies/testing_strategy.py
  - IMPLEMENT: TestingDecompositionStrategy for test tasks
  - FOLLOW pattern: BaseDecomposer inheritance
  - NAMING: TestingDecompositionStrategy, test-specific patterns
  - DEPENDENCIES: BaseDecomposer, test coverage analysis
  - PLACEMENT: Strategies module

Task 7: CREATE src/decomposition/fractal_decomposer.py
  - IMPLEMENT: FractalDecomposer main decomposition engine
  - FOLLOW pattern: Recursive decomposition with depth limiting
  - NAMING: FractalDecomposer, decompose_recursive, select_strategy
  - DEPENDENCIES: All strategies, ComplexityEstimator, DependencyManager
  - PLACEMENT: Core decomposition module

Task 8: CREATE src/decomposition/task_assigner.py
  - IMPLEMENT: TaskAssigner for model/agent assignment
  - FOLLOW pattern: src/llm/router.py routing logic
  - NAMING: TaskAssigner, assign_model, optimize_assignments
  - DEPENDENCIES: ModelRouter, Task complexity
  - PLACEMENT: Decomposition subsystem

Task 9: CREATE src/decomposition/progress_tracker.py
  - IMPLEMENT: ProgressTracker for hierarchical tracking
  - FOLLOW pattern: Bottom-up progress aggregation
  - NAMING: ProgressTracker, update_progress, calculate_rollup
  - DEPENDENCIES: DecompositionTree, Redis for atomic updates
  - PLACEMENT: Decomposition subsystem

Task 10: CREATE src/services/decomposition_service.py
  - IMPLEMENT: DecompositionOrchestrator high-level service
  - FOLLOW pattern: src/services/llm_service.py facade pattern
  - NAMING: DecompositionOrchestrator, decompose_task, track_progress
  - DEPENDENCIES: All decomposition components
  - PLACEMENT: Services layer

Task 11: MODIFY src/agents/planner.py
  - INTEGRATE: Replace _create_plan with decomposition service
  - FIND pattern: _create_plan placeholder method
  - REPLACE: With actual decomposition service calls
  - PRESERVE: Async execution, state management

Task 12: CREATE src/tests/decomposition/test_fractal.py
  - IMPLEMENT: Unit tests for fractal decomposition
  - FOLLOW pattern: pytest-asyncio with fixtures
  - COVERAGE: Recursive decomposition, depth limiting, strategies
  - PLACEMENT: Decomposition test directory

Task 13: CREATE src/tests/decomposition/test_dependencies.py
  - IMPLEMENT: Tests for dependency management
  - FOLLOW pattern: Graph algorithm testing
  - COVERAGE: Cycle detection, topological sort, validation
  - PLACEMENT: Decomposition test directory

Task 14: CREATE src/tests/decomposition/test_progress.py
  - IMPLEMENT: Tests for progress tracking
  - FOLLOW pattern: Hierarchical aggregation testing
  - COVERAGE: Progress rollup, atomic updates, accuracy
  - PLACEMENT: Decomposition test directory

Task 15: MODIFY src/models/state_models.py
  - ENHANCE: Add decomposition_tree field to AgentState
  - FIND pattern: subtasks field in AgentState
  - ADD: decomposition_tree_id for tree reference
  - PRESERVE: TypedDict structure for LangGraph
```

### Implementation Patterns & Key Details

```python
# Fractal decomposition pattern
class FractalDecomposer:
    """
    PATTERN: Recursive decomposition with strategy selection
    CRITICAL: Must limit depth to prevent stack overflow
    """
    
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.strategies = {}
        self._register_strategies()
        
    async def decompose_recursive(
        self,
        task: Task,
        depth: int = 0
    ) -> List[Task]:
        """
        Recursively decompose task into subtasks.
        GOTCHA: Check complexity threshold and depth limit
        """
        # Base cases
        if depth >= self.max_depth:
            task.is_leaf = True
            task.can_decompose = False
            return [task]
            
        if task.estimated_complexity < self.complexity_threshold:
            task.is_leaf = True
            return [task]
            
        # Select strategy based on task type
        strategy = self._select_strategy(task.type)
        
        # Generate subtasks
        subtasks = await strategy.decompose(task)
        
        # Recursively decompose each subtask
        all_tasks = []
        for subtask in subtasks:
            subtask.depth = depth + 1
            subtask.parent_id = task.id
            
            # Estimate complexity for subtask
            subtask.estimated_complexity = await self._estimate_complexity(subtask)
            
            # Recursive decomposition
            decomposed = await self.decompose_recursive(subtask, depth + 1)
            all_tasks.extend(decomposed)
            
        return all_tasks

# Dependency management pattern
from graphlib import TopologicalSorter

class DependencyManager:
    """
    PATTERN: DAG management with cycle detection
    CRITICAL: Must maintain topological ordering
    """
    
    def __init__(self):
        self.graph = {}  # task_id -> set of dependencies
        
    def add_dependency(self, task_id: str, depends_on: str):
        """Add dependency relationship."""
        if task_id not in self.graph:
            self.graph[task_id] = set()
        self.graph[task_id].add(depends_on)
        
    def validate_dependencies(self) -> DependencyValidation:
        """
        Validate DAG and detect cycles.
        PATTERN: Use TopologicalSorter for validation
        """
        try:
            sorter = TopologicalSorter(self.graph)
            execution_order = list(sorter.static_order())
            
            return DependencyValidation(
                is_valid=True,
                has_cycles=False,
                execution_order=execution_order
            )
        except CycleError as e:
            # Extract cycle information
            cycles = self._extract_cycles(self.graph)
            
            return DependencyValidation(
                is_valid=False,
                has_cycles=True,
                cycles=cycles
            )
    
    def get_execution_batches(self) -> List[List[str]]:
        """
        Get batches of tasks that can run in parallel.
        PATTERN: Tasks with no dependencies can run together
        """
        sorter = TopologicalSorter(self.graph)
        batches = []
        
        while sorter.is_active():
            batch = sorter.get_ready()
            if batch:
                batches.append(list(batch))
                for task_id in batch:
                    sorter.done(task_id)
                    
        return batches

# Complexity estimation pattern
class ComplexityEstimator:
    """
    PATTERN: Multi-factor complexity estimation
    GOTCHA: Balance between accuracy and speed
    """
    
    def __init__(self, task_analyzer: TaskComplexityAnalyzer):
        self.task_analyzer = task_analyzer
        
    async def estimate_task_complexity(
        self,
        task: Task,
        parent_complexity: float = 0.5
    ) -> float:
        """
        Estimate subtask complexity.
        PATTERN: Inherit and adjust from parent
        """
        # Base estimation from task description
        analysis = await self.task_analyzer.analyze_task(
            task.description,
            task.type
        )
        
        base_complexity = self._map_to_normalized_score(analysis.complexity)
        
        # Adjust based on parent complexity (inheritance)
        inherited_factor = 0.3  # 30% inherited from parent
        estimated = (base_complexity * 0.7) + (parent_complexity * inherited_factor)
        
        # Apply task type modifiers
        type_modifiers = {
            TaskType.ARCHITECTURE: 1.2,
            TaskType.REFACTORING: 1.1,
            TaskType.DEBUGGING: 1.15,
            TaskType.TESTING: 0.8,
            TaskType.DOCUMENTATION: 0.6,
        }
        
        modifier = type_modifiers.get(task.type, 1.0)
        final_complexity = min(1.0, estimated * modifier)
        
        return final_complexity

# Progress tracking pattern
class ProgressTracker:
    """
    PATTERN: Hierarchical progress aggregation
    CRITICAL: Must be atomic for concurrent updates
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        
    async def update_progress(
        self,
        tree: DecompositionTree,
        task_id: str,
        progress: float
    ):
        """
        Update task progress and propagate to parents.
        PATTERN: Bottom-up aggregation
        CRITICAL: Use Redis transaction for atomicity
        """
        task = tree.tasks[task_id]
        task.progress = progress
        
        # If task completed, update status
        if progress >= 100:
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            
        # Propagate to parent
        if task.parent_id:
            await self._update_parent_progress(tree, task.parent_id)
            
    async def _update_parent_progress(
        self,
        tree: DecompositionTree,
        parent_id: str
    ):
        """
        Calculate parent progress from children.
        PATTERN: Weighted average based on complexity
        """
        parent = tree.tasks[parent_id]
        children = [
            t for t in tree.tasks.values()
            if t.parent_id == parent_id
        ]
        
        if not children:
            return
            
        # Calculate weighted progress
        total_weight = sum(c.estimated_complexity for c in children)
        if total_weight == 0:
            # Simple average if no complexity weights
            parent.progress = sum(c.progress for c in children) / len(children)
        else:
            # Weighted average
            weighted_sum = sum(
                c.progress * c.estimated_complexity
                for c in children
            )
            parent.progress = weighted_sum / total_weight
            
        # Update counts
        parent.completed_subtask_count = sum(
            1 for c in children if c.status == TaskStatus.COMPLETED
        )
        
        # Recursively update parent's parent
        if parent.parent_id:
            await self._update_parent_progress(tree, parent.parent_id)

# Task assignment pattern
class TaskAssigner:
    """
    PATTERN: Optimal model assignment based on complexity
    GOTCHA: Must respect budget constraints
    """
    
    def __init__(self, model_router: ModelRouter, cost_tracker: CostTracker):
        self.model_router = model_router
        self.cost_tracker = cost_tracker
        
    async def assign_models(
        self,
        tree: DecompositionTree,
        budget_remaining: float
    ) -> None:
        """
        Assign optimal models to leaf tasks.
        PATTERN: Greedy assignment with budget constraint
        """
        # Get leaf tasks sorted by complexity (high to low)
        leaf_tasks = [
            tree.tasks[tid] for tid in tree.leaf_tasks
        ]
        leaf_tasks.sort(key=lambda t: t.estimated_complexity, reverse=True)
        
        for task in leaf_tasks:
            # Route to appropriate model
            routing_decision = await self.model_router.route_request(
                task_description=task.description,
                complexity=task.estimated_complexity,
                budget_constraint=budget_remaining
            )
            
            task.assigned_model = routing_decision.selected_model.model_name
            task.estimated_cost = routing_decision.estimated_cost
            task.estimated_tokens = routing_decision.estimated_tokens
            
            # Update remaining budget
            budget_remaining -= task.estimated_cost
            
            if budget_remaining <= 0:
                # Fall back to cheapest model for remaining tasks
                self._assign_fallback_models(leaf_tasks[leaf_tasks.index(task)+1:])
                break
```

### Integration Points

```yaml
DEPENDENCIES:
  - graphlib: Built-in Python for topological sorting
  - networkx: Optional for advanced graph algorithms
  
LLM_SERVICE:
  - integration: "Use for decomposition suggestions"
  - pattern: "Generate subtask descriptions"
  - model_selection: "Use medium complexity model for decomposition"

TASK_COMPLEXITY:
  - integration: "Reuse TaskComplexityAnalyzer"
  - enhancement: "Add task type specific patterns"
  
MEMORY_SERVICE:
  - integration: "Store decomposition trees in project memory"
  - pattern: "Cache successful decomposition patterns"
  
REDIS:
  - usage: "Atomic progress updates"
  - keys: |
      decomposition:{tree_id} - Serialized tree
      progress:{tree_id}:{task_id} - Task progress
      execution:{tree_id} - Execution state

CONFIG:
  - add to: .env
  - variables: |
      # Decomposition Configuration
      DECOMPOSITION_MAX_DEPTH=10
      DECOMPOSITION_MAX_SUBTASKS=5
      DECOMPOSITION_COMPLEXITY_THRESHOLD=0.2
      
      # Progress Tracking
      PROGRESS_UPDATE_INTERVAL=5
      PROGRESS_BATCH_SIZE=10
      
      # Task Assignment
      TASK_ASSIGNMENT_STRATEGY=greedy
      TASK_BUDGET_RESERVE=0.1

WORKFLOW:
  - integration: "Planner agent uses decomposition service"
  - state_update: "Add decomposition_tree to AgentState"
  - checkpoint: "Include tree in checkpoint data"
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# After creating each file
ruff check src/decomposition/ --fix
mypy src/decomposition/ --strict
ruff format src/decomposition/

# Verify imports
python -c "from src.decomposition import FractalDecomposer; print('Decomposition imports OK')"
python -c "from src.services.decomposition_service import DecompositionOrchestrator; print('Service imports OK')"

# Expected: Zero errors, all imports resolve
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test fractal decomposition
pytest src/tests/decomposition/test_fractal.py -v --cov=src/decomposition/fractal_decomposer

# Test dependency management
pytest src/tests/decomposition/test_dependencies.py -v --cov=src/decomposition/dependency_manager

# Test complexity estimation
pytest src/tests/decomposition/test_complexity.py -v --cov=src/decomposition/complexity_estimator

# Test progress tracking
pytest src/tests/decomposition/test_progress.py -v --cov=src/decomposition/progress_tracker

# Full decomposition test suite
pytest src/tests/decomposition/ -v --cov=src/decomposition --cov-report=term-missing

# Expected: 90%+ coverage, all tests pass
```

### Level 3: Integration Testing (System Validation)

```bash
# Test simple task decomposition
python scripts/test_simple_decomposition.py \
  --task "Create a user authentication system" \
  --max-depth 3 \
  --visualize
# Expected: Tree with 10-20 subtasks, clear dependencies

# Test complex feature decomposition
python scripts/test_complex_decomposition.py \
  --task "Build a real-time chat application with WebSockets" \
  --max-depth 5 \
  --estimate-costs \
  --assign-models
# Expected: 50+ subtasks, model assignments, cost < $2

# Test dependency validation
python scripts/test_dependency_validation.py \
  --create-circular \
  --detect-cycles \
  --resolve
# Expected: Cycles detected and resolved

# Test progress tracking
python scripts/test_progress_tracking.py \
  --create-tree \
  --simulate-execution \
  --verify-rollup
# Expected: Progress accurately rolled up to root

# Test execution ordering
python scripts/test_execution_ordering.py \
  --complex-dependencies \
  --generate-batches \
  --verify-parallel
# Expected: Valid parallel execution batches
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Large Feature Decomposition Test
python scripts/test_large_feature.py \
  --task "Implement complete e-commerce backend with payment processing" \
  --measure-time \
  --measure-memory
# Expected: <500ms decomposition, <100MB memory

# Strategy Effectiveness Test
python scripts/test_strategy_effectiveness.py \
  --test-all-strategies \
  --compare-decompositions \
  --measure-quality
# Expected: Each strategy produces appropriate decompositions

# Cost Optimization Test
python scripts/test_cost_optimization.py \
  --task "Refactor legacy codebase" \
  --budget 1.00 \
  --compare-with-flat \
  --measure-savings
# Expected: 60%+ cost savings vs non-decomposed

# Depth vs Quality Analysis
python scripts/test_depth_analysis.py \
  --vary-depth 1 10 \
  --measure-granularity \
  --measure-execution-success
# Expected: Optimal depth around 4-6 levels

# Circular Dependency Stress Test
python scripts/test_circular_dependencies.py \
  --generate-random-graphs 100 \
  --inject-cycles 20 \
  --verify-detection
# Expected: 100% cycle detection rate

# Progress Accuracy Test
python scripts/test_progress_accuracy.py \
  --create-deep-tree \
  --random-updates 1000 \
  --verify-consistency
# Expected: Progress always consistent across hierarchy

# Model Assignment Optimization
python scripts/test_model_assignment.py \
  --vary-complexity \
  --measure-cost \
  --measure-quality \
  --find-optimal
# Expected: Optimal assignment balances cost and quality

# Agent Integration Test
python scripts/test_planner_with_decomposition.py \
  --task "Create REST API with authentication and testing" \
  --run-workflow \
  --verify-decomposition-used
# Expected: Planner successfully uses decomposition service

# Concurrent Progress Updates
python scripts/test_concurrent_progress.py \
  --parallel-updates 100 \
  --verify-atomicity \
  --check-race-conditions
# Expected: No race conditions, all updates atomic

# Memory and Caching Test
python scripts/test_decomposition_caching.py \
  --repeated-tasks 10 \
  --measure-cache-hits \
  --verify-consistency
# Expected: 80%+ cache hit rate for similar tasks
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] Decomposition tests achieve 90%+ coverage: `pytest src/tests/decomposition/ --cov=src/decomposition`
- [ ] No linting errors: `ruff check src/decomposition/`
- [ ] No type errors: `mypy src/decomposition/ --strict`
- [ ] All strategies implemented and tested

### Feature Validation

- [ ] Complex tasks decomposed in <500ms
- [ ] Dependency graphs validated as DAGs
- [ ] Complexity estimation 85%+ accurate
- [ ] Model assignment achieves 60%+ cost savings
- [ ] Progress tracking accurate through hierarchy
- [ ] 10+ decomposition levels supported efficiently

### Code Quality Validation

- [ ] Follows existing service and agent patterns
- [ ] All operations async-compatible
- [ ] Circular dependencies detected and handled
- [ ] Progress updates atomic via Redis
- [ ] Deterministic decomposition for checkpointing
- [ ] Memory usage controlled for deep trees

### Documentation & Deployment

- [ ] Environment variables documented
- [ ] Decomposition strategies documented
- [ ] API endpoints for decomposition service
- [ ] Progress tracking webhook configured
- [ ] Visualization tools for decomposition trees

---

## Anti-Patterns to Avoid

- ❌ Don't decompose below complexity threshold (creates overhead)
- ❌ Don't allow circular dependencies (breaks execution order)
- ❌ Don't update progress without transactions (causes inconsistency)
- ❌ Don't ignore depth limits (causes stack overflow)
- ❌ Don't assign expensive models to simple tasks (wastes budget)
- ❌ Don't skip dependency validation (causes deadlocks)
- ❌ Don't decompose synchronously (blocks workflow)
- ❌ Don't store entire tree in memory (use pagination)
- ❌ Don't hardcode decomposition patterns (use strategies)
- ❌ Don't ignore parent complexity (loses context)
