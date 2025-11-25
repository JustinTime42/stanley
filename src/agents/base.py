"""Base agent class for all workflow agents."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..models.state_models import AgentState, AgentRole
from ..models.agent_models import AgentResponse
from ..services.memory_service import MemoryOrchestrator

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all workflow agents.

    All agents must:
    - Implement async execute() method
    - Return AgentResponse with state updates
    - Follow immutable state pattern (no direct mutations)
    - Integrate with memory system for context
    """

    def __init__(
        self,
        role: AgentRole,
        memory_service: Optional[MemoryOrchestrator] = None,
        llm_service: Optional["LLMOrchestrator"] = None,  # Forward reference
        tool_service: Optional["ToolOrchestrator"] = None,  # Forward reference
    ):
        """
        Initialize base agent.

        Args:
            role: Agent's role in the workflow
            memory_service: Memory orchestrator for context retrieval
            llm_service: LLM orchestrator for model access
            tool_service: Tool orchestrator for external operations
        """
        self.role = role
        self.memory_service = memory_service
        self.llm_service = llm_service
        self.tool_service = tool_service
        self.logger = logging.getLogger(f"{__name__}.{role.value}")

    @abstractmethod
    async def execute(self, state: AgentState) -> AgentResponse:
        """
        Execute agent's primary function.

        CRITICAL: Must be async and return AgentResponse
        CRITICAL: Do not mutate state directly, return state_updates dict

        Args:
            state: Current workflow state (AgentState TypedDict)

        Returns:
            AgentResponse with results and state updates
        """
        pass

    async def retrieve_context(
        self,
        query: str,
        state: AgentState,
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context from memory.

        Args:
            query: Search query
            state: Current state
            k: Number of results

        Returns:
            Dictionary with retrieved context
        """
        if not self.memory_service:
            self.logger.warning("No memory service configured")
            return {"memories": [], "context": {}}

        try:
            # Retrieve from project memory
            results = await self.memory_service.retrieve_relevant_memories(
                query=query,
                k=k,
                filters={
                    "project_id": state.get("project_id"),
                    "agent_id": self.role.value,
                },
            )

            return {
                "memories": [
                    {
                        "content": r.memory.content,
                        "score": r.score,
                        "metadata": r.memory.metadata,
                    }
                    for r in results
                ],
                "count": len(results),
            }

        except Exception as e:
            self.logger.error(f"Error retrieving context: {e}")
            return {"memories": [], "error": str(e)}

    async def store_result(
        self,
        content: str,
        state: AgentState,
        importance: float = 0.7,
        tags: Optional[list] = None,
    ) -> str:
        """
        Store agent result in memory.

        Args:
            content: Content to store
            state: Current state
            importance: Memory importance (0-1)
            tags: Optional tags

        Returns:
            Memory ID
        """
        if not self.memory_service:
            self.logger.warning("No memory service configured")
            return ""

        try:
            from ..models.memory_models import MemoryType

            memory_id = await self.memory_service.store_memory(
                content=content,
                agent_id=self.role.value,
                memory_type=MemoryType.PROJECT,
                project_id=state.get("project_id"),
                session_id=state.get("session_id"),
                importance=importance,
                tags=tags or [self.role.value],
                metadata={
                    "workflow_id": state.get("workflow_id"),
                    "checkpoint_id": state.get("checkpoint_id"),
                    "state_version": state.get("state_version"),
                },
            )

            self.logger.info(f"Stored result in memory: {memory_id}")
            return memory_id

        except Exception as e:
            # Don't log as error if it's just missing embedding function (expected in demo)
            if "embedding" in str(e).lower():
                self.logger.debug(f"Skipping memory storage (no embedding function): {e}")
            else:
                self.logger.error(f"Error storing result: {e}")
            return ""

    def create_message(
        self,
        content: str,
        message_type: str = "info",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a message for state updates.

        Args:
            content: Message content
            message_type: Type (info, error, warning, success)
            metadata: Optional metadata

        Returns:
            Message dictionary
        """
        return {
            "role": self.role.value,
            "content": content,
            "type": message_type,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

    def _create_success_response(
        self,
        result: Dict[str, Any],
        next_agent: Optional[AgentRole] = None,
        state_updates: Optional[Dict[str, Any]] = None,
        messages: Optional[list] = None,
        requires_approval: bool = False,
    ) -> AgentResponse:
        """
        Create a success response.

        Args:
            result: Execution result
            next_agent: Suggested next agent
            state_updates: State updates to apply
            messages: New messages
            requires_approval: Whether human approval is needed

        Returns:
            AgentResponse
        """
        return AgentResponse(
            success=True,
            result=result,
            next_agent=next_agent,
            messages=messages or [],
            state_updates=state_updates or {},
            requires_human_approval=requires_approval,
        )

    def _create_error_response(
        self,
        error: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """
        Create an error response.

        Args:
            error: Error message
            result: Optional partial result

        Returns:
            AgentResponse
        """
        return AgentResponse(
            success=False,
            result=result or {},
            error=error,
            messages=[self.create_message(error, "error")],
        )

    async def generate_llm_response(
        self,
        messages: List[Dict[str, str]],
        task_description: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate LLM response using the LLM service.

        CRITICAL: Use this method for all LLM interactions
        CRITICAL: Automatically tracks costs and uses routing

        Args:
            messages: Chat messages in OpenAI format
            task_description: Description of what task LLM is performing
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text

        Raises:
            Exception: If LLM service not configured or generation fails
        """
        if not self.llm_service:
            raise Exception(
                f"LLM service not configured for {self.role.value} agent. "
                "Cannot generate LLM response."
            )

        from ..models.llm_models import LLMRequest

        # Create LLM request
        request = LLMRequest(
            messages=messages,
            agent_role=self.role.value,
            task_description=task_description,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=True,
        )

        try:
            # Generate response with routing, caching, and fallback
            response = await self.llm_service.generate_response(request)

            self.logger.info(
                f"LLM response generated: {response.model_used} "
                f"(cost: ${response.total_cost:.4f}, "
                f"tokens: {response.input_tokens + response.output_tokens})"
            )

            return response.content

        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        workflow_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool using the tool service.

        CRITICAL: Use this method for all tool interactions
        CRITICAL: Automatically tracks usage and handles errors

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            workflow_id: Optional workflow ID

        Returns:
            Tool result as dictionary

        Raises:
            Exception: If tool service not configured or execution fails
        """
        if not self.tool_service:
            raise Exception(
                f"Tool service not configured for {self.role.value} agent. "
                "Cannot execute tools."
            )

        from ..models.tool_models import ToolStatus

        # Execute tool
        result = await self.tool_service.execute_tool(
            tool_name=tool_name,
            parameters=parameters,
            agent_id=self.role.value,
            workflow_id=workflow_id or "unknown",
        )

        # Check result status
        if result.status != ToolStatus.SUCCESS:
            error_msg = result.error or "Unknown tool error"
            self.logger.error(
                f"Tool '{tool_name}' failed: {error_msg}"
            )
            raise Exception(f"Tool execution failed: {error_msg}")

        self.logger.info(
            f"Tool '{tool_name}' executed successfully "
            f"in {result.execution_time_ms}ms"
        )

        # Return result
        return result.result or {}
