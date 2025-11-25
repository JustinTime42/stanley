# LangGraph State Management Documentation for PRP-02

## Critical Implementation Patterns

This document provides essential patterns and examples for implementing LangGraph-based state management in the agent swarm system.

## 1. State Schema Definition

### TypedDict Pattern (Required by LangGraph)

```python
from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict, total=False):
    """
    LangGraph requires TypedDict for state schema.
    Use total=False to make all fields optional.
    """
    # Required fields
    workflow_id: str
    messages: List[Dict[str, Any]]
    
    # Optional fields (total=False allows this)
    current_agent: Optional[str]
    task: Optional[Dict[str, Any]]
    results: Optional[Dict[str, Any]]
```

### State Update Pattern (Immutable)

```python
async def agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Nodes must return state updates, not mutate state.
    LangGraph merges returned dict with existing state.
    """
    # DON'T: Mutate state directly
    # state["messages"].append(new_message)  # ❌ Wrong!
    
    # DO: Return new state values
    return {
        "messages": state.get("messages", []) + [new_message],  # ✅ Correct
        "current_agent": "next_agent"
    }
```

## 2. Graph Construction Pattern

### Basic Graph Setup

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver

def build_agent_graph(redis_url: str):
    # 1. Create checkpointer (CRITICAL: Pass to compile, not invoke!)
    checkpointer = RedisSaver(
        redis_url=redis_url,
        key_prefix="agent_swarm:"
    )
    
    # 2. Create graph with state schema
    graph = StateGraph(AgentState)
    
    # 3. Add nodes
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("planner", planner_node)
    graph.add_node("implementer", implementer_node)
    
    # 4. Add edges (conditional or direct)
    graph.add_edge("coordinator", "planner")  # Direct edge
    graph.add_conditional_edges(  # Conditional routing
        "planner",
        route_from_planner,  # Router function
        {
            "implementer": "implementer",
            "coordinator": "coordinator",
            "end": END
        }
    )
    
    # 5. Set entry point
    graph.set_entry_point("coordinator")
    
    # 6. CRITICAL: Compile with checkpointer!
    return graph.compile(checkpointer=checkpointer)
```

## 3. Conditional Edge Patterns

### Router Functions

```python
def route_from_planner(state: AgentState) -> str:
    """
    Router functions MUST return string node names.
    """
    last_message = state["messages"][-1] if state.get("messages") else {}
    
    if last_message.get("type") == "error":
        return "coordinator"  # Go back to coordinator
    elif last_message.get("requires_implementation"):
        return "implementer"  # Proceed to implementation
    else:
        return "end"  # Finish workflow

# Alternative: Multiple conditions
def complex_router(state: AgentState) -> str:
    """
    Complex routing logic with multiple conditions.
    """
    status = state.get("status", "unknown")
    retry_count = state.get("retry_count", 0)
    
    if retry_count > 3:
        return "error_handler"
    elif status == "needs_review":
        return "human_review"
    elif status == "ready":
        return "next_agent"
    else:
        return "coordinator"
```

## 4. Human-in-the-Loop Pattern

### Interrupt Before Node

```python
def create_graph_with_human_approval():
    graph = StateGraph(AgentState)
    
    # Add node with interrupt_before for human approval
    graph.add_node(
        "architect", 
        architect_node,
        interrupt_before=True  # Pauses here for approval
    )
    
    # Human approval handled externally via API
    # Workflow resumes when approval is given
    
    return graph.compile(checkpointer=checkpointer)

# Resume after human approval
async def resume_after_approval(
    graph: CompiledGraph,
    workflow_id: str,
    approval_data: dict
):
    """
    Resume workflow after human approval.
    """
    config = {
        "configurable": {
            "thread_id": workflow_id
        }
    }
    
    # Update state with approval
    await graph.aupdate_state(
        config,
        {"human_feedback": approval_data}
    )
    
    # Resume execution
    result = await graph.ainvoke(None, config)
    return result
```

## 5. Subgraph Pattern

### Creating Subgraphs for Complex Workflows

```python
def create_planning_subgraph():
    """
    Subgraph for planning phase.
    """
    subgraph = StateGraph(AgentState)
    
    subgraph.add_node("analyze", analyze_node)
    subgraph.add_node("decompose", decompose_node)
    subgraph.add_node("prioritize", prioritize_node)
    
    subgraph.add_edge("analyze", "decompose")
    subgraph.add_edge("decompose", "prioritize")
    
    subgraph.set_entry_point("analyze")
    subgraph.set_finish_point("prioritize")
    
    return subgraph

def create_main_graph_with_subgraph():
    """
    Main graph incorporating subgraph.
    """
    main_graph = StateGraph(AgentState)
    planning_subgraph = create_planning_subgraph()
    
    # Add subgraph as a node
    main_graph.add_node("planning", planning_subgraph.compile())
    main_graph.add_node("implementation", implementation_node)
    
    main_graph.add_edge("planning", "implementation")
    
    return main_graph.compile(checkpointer=checkpointer)
```

## 6. Checkpoint and State Management

### Working with Checkpoints

```python
async def execute_with_checkpoint(
    graph: CompiledGraph,
    initial_state: dict,
    workflow_id: str
):
    """
    Execute workflow with checkpoint support.
    """
    config = {
        "configurable": {
            "thread_id": workflow_id,  # Required for checkpointing
            "checkpoint_ns": "main"    # Optional namespace
        }
    }
    
    # Execute workflow
    result = await graph.ainvoke(initial_state, config)
    
    # Get current state
    current_state = await graph.aget_state(config)
    
    # Get checkpoint history
    history = []
    async for checkpoint in graph.aget_state_history(config):
        history.append({
            "checkpoint_id": checkpoint.config["configurable"]["checkpoint_id"],
            "state": checkpoint.values,
            "next": checkpoint.next
        })
    
    return result, history

async def rollback_to_checkpoint(
    graph: CompiledGraph,
    workflow_id: str,
    checkpoint_id: str
):
    """
    Rollback to specific checkpoint.
    """
    config = {
        "configurable": {
            "thread_id": workflow_id,
            "checkpoint_id": checkpoint_id  # Specify checkpoint
        }
    }
    
    # Get state at checkpoint
    state = await graph.aget_state(config)
    
    # Resume from that checkpoint
    result = await graph.ainvoke(None, config)
    
    return result
```

## 7. Error Handling Pattern

### Retry and Error Recovery

```python
async def node_with_retry(state: AgentState) -> Dict[str, Any]:
    """
    Node with built-in retry logic.
    """
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    try:
        # Attempt operation
        result = await perform_operation(state)
        
        return {
            "result": result,
            "retry_count": 0,  # Reset on success
            "status": "success"
        }
    except Exception as e:
        if retry_count < max_retries:
            return {
                "retry_count": retry_count + 1,
                "status": "retry",
                "error": str(e),
                "next_agent": state.get("current_agent")  # Retry same agent
            }
        else:
            return {
                "status": "failed",
                "error": f"Failed after {max_retries} retries: {e}",
                "next_agent": "error_handler"
            }
```

## 8. State Streaming Pattern

### Real-time State Updates

```python
async def stream_workflow_execution(
    graph: CompiledGraph,
    initial_state: dict,
    workflow_id: str
):
    """
    Stream state updates as workflow executes.
    """
    config = {
        "configurable": {"thread_id": workflow_id}
    }
    
    async for event in graph.astream_events(
        initial_state,
        config,
        version="v2"
    ):
        if event["event"] == "on_chain_start":
            print(f"Starting: {event['name']}")
        elif event["event"] == "on_chain_end":
            print(f"Completed: {event['name']}")
        elif event["event"] == "on_chain_stream":
            print(f"Output: {event['data']}")
        
        # Yield events for real-time updates
        yield event
```

## 9. Memory Integration Pattern

### Integrating with PRP-01 Memory System

```python
async def agent_node_with_memory(state: AgentState) -> Dict[str, Any]:
    """
    Agent node that uses memory system from PRP-01.
    """
    from src.services.memory_service import MemoryOrchestrator
    from src.models.memory_models import MemoryType
    
    memory = MemoryOrchestrator(config)
    
    # Store decision in memory
    memory_id = await memory.store_memory(
        content=f"Agent decision: {state.get('decision')}",
        agent_id=state.get('current_agent'),
        memory_type=MemoryType.PROJECT,
        session_id=state.get('session_id'),
        project_id=state.get('project_id'),
        metadata={
            "workflow_id": state.get('workflow_id'),
            "checkpoint_id": state.get('checkpoint_id')
        }
    )
    
    # Retrieve relevant memories
    relevant_memories = await memory.retrieve_relevant_memories(
        query=state.get('task', {}).get('description', ''),
        k=5,
        memory_types=[MemoryType.PROJECT, MemoryType.GLOBAL]
    )
    
    # Use memories in decision making
    context = [m.memory.content for m in relevant_memories]
    
    return {
        "memory_ids": state.get("memory_ids", []) + [memory_id],
        "context": {"relevant_memories": context}
    }
```

## 10. Visualization Pattern

### Generate Mermaid Diagram

```python
def visualize_workflow(graph: CompiledGraph):
    """
    Generate Mermaid diagram of workflow.
    """
    # Get graph structure
    mermaid_diagram = graph.get_graph().draw_mermaid()
    
    # Save to file
    with open("workflow.mermaid", "w") as f:
        f.write(mermaid_diagram)
    
    # Alternative: Get as PNG
    png_data = graph.get_graph().draw_mermaid_png()
    with open("workflow.png", "wb") as f:
        f.write(png_data)
    
    return mermaid_diagram
```

## Common Pitfalls and Solutions

### Pitfall 1: Passing Checkpointer to invoke

```python
# ❌ WRONG: Don't pass checkpointer to invoke
result = await graph.ainvoke(state, checkpointer=checkpointer)

# ✅ CORRECT: Pass checkpointer to compile
compiled = graph.compile(checkpointer=checkpointer)
result = await compiled.ainvoke(state, config)
```

### Pitfall 2: Using Pydantic Models for State

```python
# ❌ WRONG: Don't use Pydantic BaseModel
class AgentState(BaseModel):
    workflow_id: str
    messages: List[dict]

# ✅ CORRECT: Use TypedDict
class AgentState(TypedDict):
    workflow_id: str
    messages: List[Dict[str, Any]]
```

### Pitfall 3: Mutating State in Nodes

```python
# ❌ WRONG: Don't mutate state
async def bad_node(state: AgentState):
    state["messages"].append({"text": "new"})
    return state

# ✅ CORRECT: Return new values
async def good_node(state: AgentState):
    return {
        "messages": state.get("messages", []) + [{"text": "new"}]
    }
```

### Pitfall 4: Router Functions Returning Wrong Type

```python
# ❌ WRONG: Don't return function or object
def bad_router(state):
    return implementer_node  # Returns function

# ✅ CORRECT: Return string node name
def good_router(state):
    return "implementer"  # Returns string
```

## Testing Patterns

### Unit Testing Nodes

```python
import pytest
from unittest.mock import Mock, AsyncMock

@pytest.mark.asyncio
async def test_agent_node():
    """
    Test individual node functionality.
    """
    # Create mock state
    state = {
        "workflow_id": "test-123",
        "messages": [],
        "task": {"description": "Test task"}
    }
    
    # Execute node
    result = await coordinator_node(state)
    
    # Verify state updates
    assert "messages" in result
    assert len(result["messages"]) > 0
    assert "next_agent" in result

@pytest.mark.asyncio
async def test_router_function():
    """
    Test conditional edge router.
    """
    state = {"status": "planning"}
    next_node = route_from_coordinator(state)
    assert next_node == "planner"
```

### Integration Testing Workflow

```python
@pytest.mark.asyncio
async def test_full_workflow():
    """
    Test complete workflow execution.
    """
    # Create graph
    graph = create_workflow(test_checkpointer)
    
    # Initial state
    initial_state = {
        "workflow_id": "test-workflow",
        "task": {"description": "Create calculator"},
        "messages": []
    }
    
    # Execute workflow
    config = {"configurable": {"thread_id": "test-thread"}}
    result = await graph.ainvoke(initial_state, config)
    
    # Verify completion
    assert result["status"] == "complete"
    assert len(result["messages"]) > 0
```

## Performance Considerations

1. **Checkpoint Frequency**: Don't checkpoint every node, use intervals
2. **State Size**: Keep state lean, use memory references instead of large objects
3. **Async Operations**: Use async/await throughout for non-blocking execution
4. **Batch Operations**: Group memory operations when possible
5. **Connection Pooling**: Reuse connections (already in PRP-01 memory system)

## Next Steps for Implementation

1. Start with `src/models/state_models.py` - Define AgentState TypedDict
2. Create `src/agents/base.py` - BaseAgent abstract class
3. Implement one agent completely (e.g., CoordinatorAgent)
4. Create corresponding node wrapper in `src/core/nodes.py`
5. Build minimal workflow in `src/core/workflow.py`
6. Test with single agent before adding others
7. Incrementally add agents and test integration
8. Add human-in-the-loop once basic flow works
9. Implement rollback after core workflow is stable
10. Add visualization and monitoring last

Remember: Start simple, test often, add complexity gradually!