"""Implementer agent for code generation and implementation."""

import logging
from typing import Optional

from .base import BaseAgent
from ..models.state_models import AgentState, AgentRole, WorkflowStatus
from ..models.agent_models import AgentResponse
from ..services.memory_service import MemoryOrchestrator

logger = logging.getLogger(__name__)


class ImplementerAgent(BaseAgent):
    """
    Implementer agent responsible for code generation and implementation.

    Responsibilities:
    - Generate code based on architecture
    - Implement features and functionality
    - Follow coding standards and best practices
    - Create implementation artifacts
    """

    def __init__(
        self,
        memory_service: Optional[MemoryOrchestrator] = None,
        llm_service: Optional["LLMOrchestrator"] = None,
        tool_service: Optional["ToolOrchestrator"] = None,
    ):
        """Initialize implementer agent."""
        super().__init__(
            role=AgentRole.IMPLEMENTER,
            memory_service=memory_service,
            llm_service=llm_service,
            tool_service=tool_service,
        )

    async def execute(self, state: AgentState) -> AgentResponse:
        """
        Execute implementation logic.

        Args:
            state: Current workflow state

        Returns:
            AgentResponse with implementation results
        """
        try:
            self.logger.info(f"Implementer working on {state.get('workflow_id')}")

            architecture = state.get("architecture", {})
            subtasks = state.get("subtasks", [])

            # Retrieve implementation examples
            context = await self.retrieve_context(
                query="code implementation examples",
                state=state,
                k=5,
            )

            # Create implementation using real tools
            implementation = await self._implement_code(architecture, subtasks, context, state)

            # Store implementation
            await self.store_result(
                content=f"Implemented {len(implementation['files'])} files",
                state=state,
                importance=0.8,
                tags=["implementation", "code"],
            )

            # Create message
            impl_message = self.create_message(
                content=f"Implementation complete: {len(implementation['files'])} files created",
                message_type="success",
                metadata={"implementation_id": implementation["implementation_id"]},
            )

            # State updates
            state_updates = {
                "implementation": implementation,
                "status": WorkflowStatus.TESTING.value,
            }

            return self._create_success_response(
                result=implementation,
                next_agent=AgentRole.TESTER,
                state_updates=state_updates,
                messages=[impl_message],
            )

        except Exception as e:
            self.logger.error(f"Implementer execution failed: {e}")
            return self._create_error_response(
                error=f"Implementation failed: {str(e)}",
            )

    async def _implement_code(
        self,
        architecture: dict,
        subtasks: list,
        context: dict,
        state: dict,
    ) -> dict:
        """
        Implement code based on architecture using real tools.

        Args:
            architecture: Architecture specification
            subtasks: Subtasks to implement
            context: Retrieved context
            state: Current workflow state

        Returns:
            Implementation details
        """
        import uuid

        implementation_id = str(uuid.uuid4())
        files = []
        workflow_id = state.get("workflow_id", "unknown")
        task = state.get("task", {})

        # Build full task context for better code generation
        task_description = task.get("description", "Unknown task")
        task_requirements = task.get("requirements", [])
        requirements_text = "\n".join([f"- {req}" for req in task_requirements])

        # Check if we have tools available
        if not self.tool_service:
            self.logger.error("No tool service available - cannot generate code")
            raise RuntimeError(
                "Implementer requires tool service to generate code. "
                "Please ensure ToolOrchestrator is properly configured."
            )

        # Generate a single comprehensive implementation
        # Instead of splitting by subtasks, generate complete solution

        # Extract component names (handle both string and dict formats)
        components = architecture.get('components', [])
        if components and isinstance(components[0], dict):
            component_names = [c.get('name', 'unknown') for c in components]
        else:
            component_names = components if components else ['main module']

        # Extract dependencies (handle both string and dict formats)
        dependencies = architecture.get('dependencies', [])
        if dependencies and isinstance(dependencies[0], dict):
            dep_names = [d.get('name', str(d)) for d in dependencies]
        else:
            dep_names = dependencies if dependencies else ['none']

        full_description = f"""
Task: {task_description}

Requirements:
{requirements_text}

Architecture:
- Components: {', '.join(component_names)}
- Dependencies: {', '.join(dep_names)}

Please implement a complete, working solution that fulfills all requirements.
Include proper error handling, docstrings, and follow best practices.
"""

        try:
            # Generate main implementation
            # Extract subtask names/descriptions for context
            subtask_names = [
                s.get('name') or s.get('description', 'unknown')[:50]
                for s in subtasks
            ]

            code_result = await self.execute_tool(
                tool_name="generate_code",
                parameters={
                    "description": full_description,
                    "language": "python",
                    "context": f"Subtasks: {subtask_names}",
                },
                workflow_id=workflow_id,
            )

            generated_code = code_result.get("code", "")
            if not generated_code or generated_code.strip() == "":
                raise RuntimeError("Code generation returned empty result")

            # Determine appropriate file path
            file_path = "src/calculator.py"  # Default for calculator task
            if "calculator" not in task_description.lower():
                # Use first subtask name if not calculator
                if subtasks:
                    name = subtasks[0].get("name") or subtasks[0].get("id", "main")
                    name = name.lower().replace(" ", "_")
                    file_path = f"src/{name}.py"
                else:
                    file_path = "src/main.py"

            # Write file
            write_result = await self.execute_tool(
                tool_name="write_file",
                parameters={
                    "path": file_path,
                    "content": generated_code,
                    "create_dirs": True,
                },
                workflow_id=workflow_id,
            )

            files.append({
                "path": file_path,
                "content": generated_code,
                "language": "python",
            })

            self.logger.info(f"Generated {len(files)} implementation file(s)")

        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            raise RuntimeError(f"Failed to generate code: {e}")

        return {
            "implementation_id": implementation_id,
            "architecture_id": architecture.get("architecture_id"),
            "files": files,
            "lines_of_code": sum(len(f["content"].split("\n")) for f in files),
            "context_used": len(context.get("memories", [])),
            "tool_based": True,
            "task_description": task_description,
        }
