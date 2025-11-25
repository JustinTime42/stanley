"""Architect agent for system design and architecture."""

import logging
from typing import Optional

from .base import BaseAgent
from ..models.state_models import AgentState, AgentRole, WorkflowStatus
from ..models.agent_models import AgentResponse
from ..services.memory_service import MemoryOrchestrator

logger = logging.getLogger(__name__)


class ArchitectAgent(BaseAgent):
    """
    Architect agent responsible for system design and architecture.

    Responsibilities:
    - Design system architecture
    - Define components and interfaces
    - Create technical specifications
    - Ensure architectural best practices
    """

    def __init__(
        self,
        memory_service: Optional[MemoryOrchestrator] = None,
        llm_service: Optional["LLMOrchestrator"] = None,  # Forward reference
        architecture_service: Optional["ArchitectureOrchestrator"] = None,  # Forward reference
        use_advanced_architecture: bool = False,
    ):
        """Initialize architect agent.

        Args:
            memory_service: Memory service for context retrieval
            llm_service: LLM service for architecture design
            architecture_service: Architecture service for pattern-based design (PRP-08 enhancement)
            use_advanced_architecture: Whether to use advanced architecture features (PRP-08 enhancement)
        """
        super().__init__(
            role=AgentRole.ARCHITECT,
            memory_service=memory_service,
            llm_service=llm_service,
        )
        self.architecture_service = architecture_service
        self.use_advanced_architecture = use_advanced_architecture

    async def execute(self, state: AgentState) -> AgentResponse:
        """
        Execute architecture design logic.

        Args:
            state: Current workflow state

        Returns:
            AgentResponse with architecture design
        """
        try:
            self.logger.info(f"Architect designing system {state.get('workflow_id')}")

            plan = state.get("plan", {})
            task = state.get("task", {})

            # Retrieve architectural patterns from memory
            context = await self.retrieve_context(
                query=f"architecture patterns: {task.get('description', '')}",
                state=state,
                k=5,
            )

            # Create architecture using LLM
            architecture = await self._design_architecture(plan, task, context)

            # Store architecture
            await self.store_result(
                content=f"Designed architecture with {len(architecture['components'])} components",
                state=state,
                importance=0.9,
                tags=["architecture", "design"],
            )

            # Create message
            arch_message = self.create_message(
                content=f"Architecture designed with {len(architecture['components'])} components",
                message_type="success",
                metadata={"architecture_id": architecture["architecture_id"]},
            )

            # State updates
            state_updates = {
                "architecture": architecture,
                "status": WorkflowStatus.IMPLEMENTING.value,
            }

            return self._create_success_response(
                result=architecture,
                next_agent=AgentRole.IMPLEMENTER,
                state_updates=state_updates,
                messages=[arch_message],
                requires_approval=True,  # Architect requires approval
            )

        except Exception as e:
            self.logger.error(f"Architect execution failed: {e}")
            return self._create_error_response(
                error=f"Architecture design failed: {str(e)}",
            )

    async def _design_architecture(self, plan: dict, task: dict, context: dict) -> dict:
        """
        Design system architecture using LLM.

        Args:
            plan: Execution plan
            task: Task specification
            context: Retrieved context

        Returns:
            Architecture specification
        """
        import uuid
        import json

        task_description = task.get("description", "Unknown task")
        subtasks = plan.get("subtasks", [])
        subtask_descriptions = "\n".join([
            f"- {st.get('description', st.get('id', 'unknown'))}"
            for st in subtasks
        ])

        # If no LLM service, return basic architecture based on task
        if not self.llm_service:
            self.logger.warning("No LLM service, using basic architecture")
            return self._create_basic_architecture(plan, task)

        # Use LLM to design architecture
        system_prompt = """You are an expert software architect. Design a clean, simple architecture for the given task.

Output ONLY valid JSON in this format:
{
  "components": [
    {
      "name": "descriptive_component_name",
      "type": "module|service|class",
      "responsibilities": ["what this component does"],
      "interfaces": ["how it interacts with other components"]
    }
  ],
  "patterns": ["list of design patterns used"],
  "technologies": ["Python", "any relevant libraries"],
  "dependencies": ["external dependencies if any"]
}

Keep the architecture simple and focused on the task requirements."""

        user_prompt = f"""Design architecture for:
Task: {task_description}

Planned subtasks:
{subtask_descriptions}

Create a minimal, clean architecture that fulfills this task."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.generate_llm_response(
                messages=messages,
                task_description=f"Architecture design: {task_description}",
                temperature=0.5,
                max_tokens=1500,
            )

            # Parse JSON response
            arch_data = self._extract_json_from_response(response)

            return {
                "architecture_id": str(uuid.uuid4()),
                "plan_id": plan.get("plan_id"),
                "components": arch_data.get("components", []),
                "patterns": arch_data.get("patterns", []),
                "technologies": arch_data.get("technologies", ["Python"]),
                "dependencies": arch_data.get("dependencies", []),
                "context_used": len(context.get("memories", [])),
                "generated_by": "llm",
            }

        except Exception as e:
            self.logger.error(f"LLM architecture design failed: {e}")
            return self._create_basic_architecture(plan, task)

    def _extract_json_from_response(self, response: str) -> dict:
        """Extract JSON from LLM response."""
        import json
        import re

        # Try direct parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON in text
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not extract JSON from response")

    def _create_basic_architecture(self, plan: dict, task: dict) -> dict:
        """Create basic architecture without LLM."""
        import uuid

        task_description = task.get("description", "").lower()

        # Infer components from task
        components = []
        if "calculator" in task_description:
            components = [
                {
                    "name": "Calculator",
                    "type": "class",
                    "responsibilities": ["Perform arithmetic operations", "Handle errors"],
                    "interfaces": ["add", "subtract", "multiply", "divide"],
                },
                {
                    "name": "CLI",
                    "type": "module",
                    "responsibilities": ["Parse user input", "Display results"],
                    "interfaces": ["main"],
                },
            ]
        else:
            # Generic architecture
            components = [
                {
                    "name": "Main",
                    "type": "module",
                    "responsibilities": ["Core functionality"],
                    "interfaces": ["main"],
                },
            ]

        return {
            "architecture_id": str(uuid.uuid4()),
            "plan_id": plan.get("plan_id"),
            "components": components,
            "patterns": ["Modular"],
            "technologies": ["Python"],
            "dependencies": [],
            "context_used": 0,
            "generated_by": "fallback",
        }
