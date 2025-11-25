# Agent Implementation Examples for PRP-02

## Complete Working Example: Coordinator Agent

This file provides a complete, production-ready implementation example for the Coordinator Agent that can be used as a template for all other agents.

```python
# src/agents/base.py
"""Base agent abstract class for all agents in the swarm."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import uuid

from ..models.agent_models import AgentRequest, AgentResponse
from ..models.state_models import AgentRole, WorkflowStatus
from ..services.memory_service import MemoryOrchestrator
from ..models.memory_models import MemoryType

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides common functionality and enforces interface contract.
    """
    
    def __init__(
        self,
        agent_role: AgentRole,
        memory_service: Optional[MemoryOrchestrator] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base agent.
        
        Args:
            agent_role: Role of this agent
            memory_service: Memory orchestrator for context
            config: Additional configuration
        """
        self.agent_role = agent_role
        self.memory_service = memory_service
        self.config = config or {}
        self.agent_id = f"{agent_role.value}_{uuid.uuid4().hex[:8]}"
        
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> AgentResponse:
        """
        Execute agent logic on current state.
        
        Args:
            state: Current workflow state
            
        Returns:
            AgentResponse with results and state updates
        """
        pass
    
    async def store_decision(
        self,
        decision: str,
        state: Dict[str, Any],
        importance: float = 0.5
    ) -> Optional[str]:
        """
        Store agent decision in memory.
        
        Args:
            decision: Decision description
            state: Current state
            importance: Importance score (0-1)
            
        Returns:
            Memory ID if stored
        """
        if not self.memory_service:
            return None
            
        try:
            memory_id = await self.memory_service.store_memory(
                content=f"[{self.agent_role.value}] Decision: {decision}",
                agent_id=self.agent_id,
                memory_type=MemoryType.PROJECT,
                session_id=state.get("session_id"),
                project_id=state.get("project_id"),
                importance=importance,
                metadata={
                    "workflow_id": state.get("workflow_id"),
                    "checkpoint_id": state.get("checkpoint_id"),
                    "agent_role": self.agent_role.value
                }
            )
            logger.info(f"Stored decision with memory_id: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Failed to store decision: {e}")
            return None
    
    async def retrieve_context(
        self,
        query: str,
        state: Dict[str, Any],
        k: int = 5
    ) -> List[str]:
        """
        Retrieve relevant context from memory.
        
        Args:
            query: Search query
            state: Current state
            k: Number of memories to retrieve
            
        Returns:
            List of relevant memory contents
        """
        if not self.memory_service:
            return []
            
        try:
            memories = await self.memory_service.retrieve_relevant_memories(
                query=query,
                k=k,
                memory_types=[MemoryType.PROJECT, MemoryType.GLOBAL],
                filters={
                    "project_id": state.get("project_id")
                }
            )
            return [m.memory.content for m in memories]
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []
```

```python
# src/agents/coordinator.py
"""Coordinator agent implementation - orchestrates overall workflow."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base import BaseAgent
from ..models.agent_models import AgentRequest, AgentResponse
from ..models.state_models import AgentRole, WorkflowStatus

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    """
    Coordinator agent orchestrates the overall workflow.
    
    Responsibilities:
    - Analyze incoming tasks
    - Route to appropriate agents
    - Monitor workflow progress
    - Handle workflow-level errors
    - Determine completion status
    """
    
    def __init__(self, memory_service=None, config=None):
        """Initialize coordinator agent."""
        super().__init__(
            agent_role=AgentRole.COORDINATOR,
            memory_service=memory_service,
            config=config
        )
        
    async def execute(self, state: Dict[str, Any]) -> AgentResponse:
        """
        Execute coordinator logic.
        
        Analyzes current state and determines next steps.
        """
        logger.info(f"Coordinator executing with state status: {state.get('status')}")
        
        try:
            # Retrieve relevant context
            task_description = state.get("task", {}).get("description", "")
            context = await self.retrieve_context(
                query=task_description,
                state=state,
                k=3
            )
            
            # Analyze current workflow status
            status = state.get("status", WorkflowStatus.PENDING.value)
            messages = state.get("messages", [])
            
            # Determine next action based on status
            if status == WorkflowStatus.PENDING.value:
                return await self._handle_new_task(state, context)
            elif status == WorkflowStatus.FAILED.value:
                return await self._handle_failure(state, context)
            elif status == WorkflowStatus.COMPLETE.value:
                return await self._handle_completion(state)
            else:
                return await self._route_to_next_agent(state, context)
                
        except Exception as e:
            logger.error(f"Coordinator execution failed: {e}")
            return AgentResponse(
                success=False,
                result={"error": str(e)},
                error=str(e),
                state_updates={
                    "status": WorkflowStatus.FAILED.value,
                    "updated_at": datetime.now().isoformat()
                }
            )
    
    async def _handle_new_task(
        self,
        state: Dict[str, Any],
        context: List[str]
    ) -> AgentResponse:
        """
        Handle a new task by analyzing and routing.
        
        Args:
            state: Current state
            context: Retrieved context
            
        Returns:
            Response directing to planning phase
        """
        task = state.get("task", {})
        
        # Analyze task complexity
        complexity = self._analyze_complexity(task)
        
        # Store analysis decision
        await self.store_decision(
            f"New task analyzed - Complexity: {complexity}",
            state,
            importance=0.7
        )
        
        # Create analysis message
        analysis_message = {
            "role": self.agent_role.value,
            "content": f"Task analysis complete. Complexity: {complexity}",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "complexity": complexity,
                "estimated_agents": self._estimate_required_agents(complexity)
            }
        }
        
        return AgentResponse(
            success=True,
            result={
                "analysis": "Task analyzed successfully",
                "complexity": complexity
            },
            next_agent=AgentRole.PLANNER,
            messages=[analysis_message],
            state_updates={
                "status": WorkflowStatus.PLANNING.value,
                "current_agent": AgentRole.PLANNER.value,
                "updated_at": datetime.now().isoformat()
            }
        )
    
    async def _handle_failure(
        self,
        state: Dict[str, Any],
        context: List[str]
    ) -> AgentResponse:
        """
        Handle workflow failure by analyzing and recovering.
        
        Args:
            state: Current state with failure info
            context: Retrieved context
            
        Returns:
            Response with recovery strategy
        """
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        
        if retry_count < max_retries:
            # Attempt recovery
            return AgentResponse(
                success=True,
                result={"action": "retry"},
                next_agent=AgentRole.DEBUGGER,
                messages=[{
                    "role": self.agent_role.value,
                    "content": f"Attempting recovery (retry {retry_count + 1}/{max_retries})",
                    "timestamp": datetime.now().isoformat()
                }],
                state_updates={
                    "status": WorkflowStatus.DEBUGGING.value,
                    "retry_count": retry_count + 1,
                    "current_agent": AgentRole.DEBUGGER.value
                }
            )
        else:
            # Max retries exceeded
            return AgentResponse(
                success=False,
                result={"action": "failed"},
                error="Max retries exceeded",
                requires_human_approval=True,
                state_updates={
                    "status": WorkflowStatus.HUMAN_REVIEW.value,
                    "requires_human_approval": True
                }
            )
    
    async def _handle_completion(
        self,
        state: Dict[str, Any]
    ) -> AgentResponse:
        """
        Handle workflow completion.
        
        Args:
            state: Current state
            
        Returns:
            Final response
        """
        # Store completion
        await self.store_decision(
            "Workflow completed successfully",
            state,
            importance=1.0
        )
        
        return AgentResponse(
            success=True,
            result={
                "status": "complete",
                "summary": self._generate_summary(state)
            },
            next_agent=None,  # No next agent, workflow ends
            messages=[{
                "role": self.agent_role.value,
                "content": "Workflow completed successfully",
                "timestamp": datetime.now().isoformat()
            }],
            state_updates={
                "status": WorkflowStatus.COMPLETE.value,
                "completed_at": datetime.now().isoformat()
            }
        )
    
    async def _route_to_next_agent(
        self,
        state: Dict[str, Any],
        context: List[str]
    ) -> AgentResponse:
        """
        Route to appropriate next agent based on current status.
        
        Args:
            state: Current state
            context: Retrieved context
            
        Returns:
            Response with next agent routing
        """
        status = state.get("status", "")
        
        # Status-based routing map
        routing_map = {
            WorkflowStatus.PLANNING.value: AgentRole.ARCHITECT,
            WorkflowStatus.DESIGNING.value: AgentRole.IMPLEMENTER,
            WorkflowStatus.IMPLEMENTING.value: AgentRole.TESTER,
            WorkflowStatus.TESTING.value: AgentRole.VALIDATOR,
            WorkflowStatus.VALIDATING.value: AgentRole.COORDINATOR,  # Back to coordinator for completion
            WorkflowStatus.DEBUGGING.value: AgentRole.IMPLEMENTER,  # After debugging, retry implementation
        }
        
        next_agent = routing_map.get(status, AgentRole.PLANNER)
        
        return AgentResponse(
            success=True,
            result={"routing": f"Routing to {next_agent.value}"},
            next_agent=next_agent,
            messages=[{
                "role": self.agent_role.value,
                "content": f"Routing workflow to {next_agent.value}",
                "timestamp": datetime.now().isoformat()
            }],
            state_updates={
                "current_agent": next_agent.value,
                "updated_at": datetime.now().isoformat()
            }
        )
    
    def _analyze_complexity(self, task: Dict[str, Any]) -> str:
        """
        Analyze task complexity.
        
        Args:
            task: Task details
            
        Returns:
            Complexity level (low, medium, high)
        """
        description = task.get("description", "")
        requirements = task.get("requirements", [])
        
        # Simple heuristic based on description length and requirements
        if len(description) < 100 and len(requirements) < 3:
            return "low"
        elif len(description) < 500 and len(requirements) < 10:
            return "medium"
        else:
            return "high"
    
    def _estimate_required_agents(self, complexity: str) -> List[str]:
        """
        Estimate which agents will be needed.
        
        Args:
            complexity: Task complexity
            
        Returns:
            List of required agent roles
        """
        if complexity == "low":
            return ["coordinator", "planner", "implementer", "tester"]
        elif complexity == "medium":
            return ["coordinator", "planner", "architect", "implementer", "tester", "validator"]
        else:
            return ["coordinator", "planner", "architect", "implementer", "tester", "validator", "debugger"]
    
    def _generate_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate workflow summary.
        
        Args:
            state: Final state
            
        Returns:
            Summary dictionary
        """
        return {
            "workflow_id": state.get("workflow_id"),
            "task": state.get("task", {}).get("description"),
            "total_messages": len(state.get("messages", [])),
            "elapsed_time": state.get("elapsed_time", 0),
            "token_count": state.get("token_count", 0),
            "agents_involved": list(set(
                msg.get("role") for msg in state.get("messages", [])
                if msg.get("role")
            ))
        }
```

```python
# src/core/nodes.py
"""Node wrapper functions for LangGraph integration."""

import logging
from typing import Dict, Any
from datetime import datetime

from ..agents import (
    CoordinatorAgent,
    PlannerAgent,
    ArchitectAgent,
    ImplementerAgent,
    TesterAgent,
    ValidatorAgent,
    DebuggerAgent
)
from ..services.memory_service import MemoryOrchestrator
from ..config.memory_config import MemoryConfig

logger = logging.getLogger(__name__)

# Initialize memory service (singleton)
memory_config = MemoryConfig()
memory_orchestrator = MemoryOrchestrator(memory_config)


async def coordinator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coordinator node for LangGraph.
    
    Args:
        state: Current workflow state
        
    Returns:
        State updates (immutable pattern)
    """
    logger.info("Executing coordinator node")
    
    # Create agent instance
    agent = CoordinatorAgent(memory_service=memory_orchestrator)
    
    # Execute agent logic
    response = await agent.execute(state)
    
    # Build state updates (immutable)
    updates = {
        "messages": state.get("messages", []) + response.messages,
        "updated_at": datetime.now().isoformat(),
        **response.state_updates
    }
    
    # Add next agent if specified
    if response.next_agent:
        updates["next_agent"] = response.next_agent.value
    
    # Add error if failed
    if not response.success and response.error:
        updates["error"] = response.error
    
    # Add human approval flag
    if response.requires_human_approval:
        updates["requires_human_approval"] = True
    
    return updates


async def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planner node for LangGraph.
    
    Decomposes tasks into subtasks and creates execution plan.
    """
    logger.info("Executing planner node")
    
    agent = PlannerAgent(memory_service=memory_orchestrator)
    response = await agent.execute(state)
    
    return {
        "messages": state.get("messages", []) + response.messages,
        "plan": response.result.get("plan"),
        "subtasks": response.result.get("subtasks", []),
        "status": WorkflowStatus.DESIGNING.value,
        "current_agent": AgentRole.ARCHITECT.value,
        "updated_at": datetime.now().isoformat(),
        **response.state_updates
    }


# Additional node implementations follow the same pattern...
```

```python
# src/core/edges.py
"""Conditional edge logic for workflow routing."""

from typing import Dict, Any
import logging

from ..models.state_models import WorkflowStatus, AgentRole

logger = logging.getLogger(__name__)


def route_from_coordinator(state: Dict[str, Any]) -> str:
    """
    Route from coordinator to next node.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name as string
    """
    status = state.get("status", "")
    next_agent = state.get("next_agent", "")
    requires_approval = state.get("requires_human_approval", False)
    
    # Check for human approval requirement
    if requires_approval:
        return "human_approval"
    
    # Check for explicit next agent
    if next_agent:
        return next_agent
    
    # Status-based routing
    if status == WorkflowStatus.COMPLETE.value:
        return "end"
    elif status == WorkflowStatus.FAILED.value:
        if state.get("retry_count", 0) < state.get("max_retries", 3):
            return "debugger"
        else:
            return "human_approval"
    elif status == WorkflowStatus.PLANNING.value:
        return "planner"
    else:
        return "planner"  # Default to planner


def route_from_planner(state: Dict[str, Any]) -> str:
    """
    Route from planner to next node.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name
    """
    plan = state.get("plan", {})
    subtasks = state.get("subtasks", [])
    
    if not plan or not subtasks:
        # Planning failed, go back to coordinator
        return "coordinator"
    
    # Check if architecture is needed
    complexity = plan.get("complexity", "low")
    if complexity in ["medium", "high"]:
        return "architect"
    else:
        return "implementer"


def should_continue(state: Dict[str, Any]) -> str:
    """
    Determine if workflow should continue.
    
    Args:
        state: Current workflow state
        
    Returns:
        "continue" or "end"
    """
    status = state.get("status", "")
    should_continue_flag = state.get("should_continue", True)
    
    if not should_continue_flag:
        return "end"
    
    if status == WorkflowStatus.COMPLETE.value:
        return "end"
    
    if status == WorkflowStatus.FAILED.value:
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        if retry_count >= max_retries:
            return "end"
    
    return "continue"
```

## Testing the Implementation

```python
# tests/agents/test_coordinator.py
"""Tests for coordinator agent."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.agents.coordinator import CoordinatorAgent
from src.models.state_models import WorkflowStatus, AgentRole
from src.models.agent_models import AgentResponse


@pytest.fixture
def mock_memory_service():
    """Create mock memory service."""
    service = Mock()
    service.store_memory = AsyncMock(return_value="memory-123")
    service.retrieve_relevant_memories = AsyncMock(return_value=[])
    return service


@pytest.fixture
def coordinator_agent(mock_memory_service):
    """Create coordinator agent with mocked dependencies."""
    return CoordinatorAgent(memory_service=mock_memory_service)


@pytest.fixture
def base_state():
    """Create base state for testing."""
    return {
        "workflow_id": "test-workflow",
        "project_id": "test-project",
        "session_id": "test-session",
        "status": WorkflowStatus.PENDING.value,
        "task": {
            "description": "Create a calculator application",
            "requirements": ["basic operations", "clean UI"]
        },
        "messages": [],
        "retry_count": 0,
        "max_retries": 3
    }


@pytest.mark.asyncio
async def test_coordinator_handles_new_task(coordinator_agent, base_state):
    """Test coordinator handles new task correctly."""
    response = await coordinator_agent.execute(base_state)
    
    assert response.success is True
    assert response.next_agent == AgentRole.PLANNER
    assert response.state_updates["status"] == WorkflowStatus.PLANNING.value
    assert len(response.messages) == 1
    assert response.messages[0]["role"] == AgentRole.COORDINATOR.value


@pytest.mark.asyncio
async def test_coordinator_handles_failure_with_retry(coordinator_agent, base_state):
    """Test coordinator handles failure with retry."""
    base_state["status"] = WorkflowStatus.FAILED.value
    base_state["retry_count"] = 1
    
    response = await coordinator_agent.execute(base_state)
    
    assert response.success is True
    assert response.next_agent == AgentRole.DEBUGGER
    assert response.state_updates["status"] == WorkflowStatus.DEBUGGING.value
    assert response.state_updates["retry_count"] == 2


@pytest.mark.asyncio
async def test_coordinator_handles_max_retries_exceeded(coordinator_agent, base_state):
    """Test coordinator handles max retries exceeded."""
    base_state["status"] = WorkflowStatus.FAILED.value
    base_state["retry_count"] = 3
    
    response = await coordinator_agent.execute(base_state)
    
    assert response.success is False
    assert response.requires_human_approval is True
    assert response.state_updates["status"] == WorkflowStatus.HUMAN_REVIEW.value


@pytest.mark.asyncio
async def test_coordinator_handles_completion(coordinator_agent, base_state):
    """Test coordinator handles workflow completion."""
    base_state["status"] = WorkflowStatus.COMPLETE.value
    
    response = await coordinator_agent.execute(base_state)
    
    assert response.success is True
    assert response.next_agent is None
    assert "completed_at" in response.state_updates


@pytest.mark.asyncio
async def test_coordinator_stores_decisions(coordinator_agent, mock_memory_service, base_state):
    """Test coordinator stores decisions in memory."""
    await coordinator_agent.execute(base_state)
    
    # Verify memory storage was called
    assert mock_memory_service.store_memory.called
    call_args = mock_memory_service.store_memory.call_args
    assert "Decision:" in call_args.kwargs["content"]
    assert call_args.kwargs["agent_id"] == coordinator_agent.agent_id


@pytest.mark.asyncio
async def test_coordinator_retrieves_context(coordinator_agent, mock_memory_service, base_state):
    """Test coordinator retrieves context from memory."""
    # Setup mock to return some memories
    mock_memory = Mock()
    mock_memory.memory.content = "Previous decision about calculators"
    mock_memory_service.retrieve_relevant_memories.return_value = [mock_memory]
    
    await coordinator_agent.execute(base_state)
    
    # Verify context retrieval was called
    assert mock_memory_service.retrieve_relevant_memories.called
    call_args = mock_memory_service.retrieve_relevant_memories.call_args
    assert "calculator" in call_args.kwargs["query"].lower()
```

## Running the Implementation

```python
# scripts/run_agent_workflow.py
"""Script to run agent workflow with LangGraph."""

import asyncio
import logging
from typing import Dict, Any

from src.core.workflow import create_workflow
from src.services.checkpoint_service import CheckpointManager
from src.config.memory_config import MemoryConfig
from src.models.state_models import WorkflowStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_workflow(task_description: str):
    """
    Run a complete workflow for a task.
    
    Args:
        task_description: Description of task to execute
    """
    # Initialize services
    config = MemoryConfig()
    checkpoint_manager = CheckpointManager(config)
    
    # Create workflow graph
    graph = create_workflow(checkpoint_manager.get_checkpointer())
    
    # Create initial state
    initial_state = {
        "workflow_id": "demo-workflow",
        "project_id": "demo-project",
        "session_id": "demo-session",
        "status": WorkflowStatus.PENDING.value,
        "task": {
            "description": task_description,
            "requirements": []
        },
        "messages": [],
        "retry_count": 0,
        "max_retries": 3,
        "should_continue": True
    }
    
    # Configuration for execution
    config = {
        "configurable": {
            "thread_id": "demo-thread"
        }
    }
    
    # Execute workflow
    logger.info(f"Starting workflow for task: {task_description}")
    result = await graph.ainvoke(initial_state, config)
    
    logger.info(f"Workflow completed with status: {result.get('status')}")
    logger.info(f"Total messages: {len(result.get('messages', []))}")
    
    return result


if __name__ == "__main__":
    # Example task
    task = "Create a simple calculator with basic arithmetic operations"
    
    # Run workflow
    result = asyncio.run(run_workflow(task))
    
    # Print summary
    print("\n=== Workflow Summary ===")
    print(f"Status: {result.get('status')}")
    print(f"Messages: {len(result.get('messages', []))}")
    print(f"Agents involved: {set(m.get('role') for m in result.get('messages', []))}")
```

This complete example provides:

1. **Base Agent Class**: Abstract class with common functionality
2. **Coordinator Agent**: Full implementation with all methods
3. **Node Wrappers**: LangGraph integration functions
4. **Edge Logic**: Routing functions for conditional edges
5. **Comprehensive Tests**: Unit tests with mocking
6. **Run Script**: Example of how to execute the workflow

Use this as a template for implementing the other 6 agents (Planner, Architect, Implementer, Tester, Validator, Debugger), following the same patterns and structure.