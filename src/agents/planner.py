"""Planner agent for task decomposition and planning."""

import logging
from typing import Optional

from .base import BaseAgent
from ..models.state_models import AgentState, AgentRole, WorkflowStatus
from ..models.agent_models import AgentResponse
from ..services.memory_service import MemoryOrchestrator

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """
    Planner agent responsible for task decomposition and planning.

    Responsibilities:
    - Break down high-level tasks into subtasks
    - Create execution plans with dependencies
    - Estimate effort and resources
    - Define success criteria
    """

    def __init__(
        self,
        memory_service: Optional[MemoryOrchestrator] = None,
        llm_service: Optional["LLMOrchestrator"] = None,  # Forward reference
        decomposition_service: Optional["DecompositionOrchestrator"] = None,  # Forward reference
        planning_service: Optional["PlanningOrchestrator"] = None,  # Forward reference
        use_decomposition: bool = True,
        use_solution_exploration: bool = False,
    ):
        """Initialize planner agent.

        Args:
            memory_service: Memory service for context retrieval
            llm_service: LLM service for plan generation
            decomposition_service: Decomposition service for task decomposition
            planning_service: Planning service for solution exploration (PRP-08 enhancement)
            use_decomposition: Whether to use decomposition service if available
            use_solution_exploration: Whether to use solution exploration (PRP-08 enhancement)
        """
        super().__init__(
            role=AgentRole.PLANNER,
            memory_service=memory_service,
            llm_service=llm_service,
        )
        self.decomposition_service = decomposition_service
        self.planning_service = planning_service
        self.use_decomposition = use_decomposition
        self.use_solution_exploration = use_solution_exploration

    async def execute(self, state: AgentState) -> AgentResponse:
        """
        Execute planning logic to decompose tasks.

        Args:
            state: Current workflow state

        Returns:
            AgentResponse with planning results
        """
        try:
            self.logger.info(f"Planner analyzing task {state.get('workflow_id')}")

            task = state.get("task", {})
            task_description = task.get("description", "")

            # Retrieve relevant planning context
            context = await self.retrieve_context(
                query=f"planning: {task_description}",
                state=state,
                k=5,
            )

            # Try decomposition service first if available
            if self.use_decomposition and self.decomposition_service:
                try:
                    plan = await self._create_plan_with_decomposition(task, context, state)
                except Exception as e:
                    self.logger.error(f"Decomposition failed: {e}, falling back to LLM")
                    if self.llm_service:
                        plan = await self._create_plan_with_llm(task, context, state)
                    else:
                        plan = self._create_plan_fallback(task, context)
            # Otherwise use LLM if available
            elif self.llm_service:
                plan = await self._create_plan_with_llm(task, context, state)
            # Fall back to basic planning
            else:
                self.logger.warning("LLM service not available, using fallback planning")
                plan = self._create_plan_fallback(task, context)

            # Store planning results
            await self.store_result(
                content=f"Created plan with {len(plan['subtasks'])} subtasks",
                state=state,
                importance=0.9,
                tags=["planning", "decomposition"],
            )

            # Create planning message
            plan_message = self.create_message(
                content=f"Created plan with {len(plan['subtasks'])} subtasks",
                message_type="success",
                metadata={"plan_id": plan["plan_id"]},
            )

            # State updates
            state_updates = {
                "plan": plan,
                "subtasks": plan["subtasks"],
                "completed_subtasks": [],
                "status": WorkflowStatus.DESIGNING.value,
            }

            return self._create_success_response(
                result=plan,
                next_agent=AgentRole.ARCHITECT,
                state_updates=state_updates,
                messages=[plan_message],
                requires_approval=True,  # Planner requires human approval
            )

        except Exception as e:
            self.logger.error(f"Planner execution failed: {e}")
            return self._create_error_response(
                error=f"Planning failed: {str(e)}",
            )

    async def _create_plan_with_decomposition(
        self,
        task: dict,
        context: dict,
        state: AgentState,
    ) -> dict:
        """
        Create execution plan using decomposition service.

        PATTERN: Use fractal decomposition for intelligent task breakdown
        CRITICAL: Convert decomposition tree to plan format

        Args:
            task: Task specification
            context: Retrieved context
            state: Current state

        Returns:
            Plan dictionary
        """
        from ..models.decomposition_models import DecompositionRequest
        import uuid

        task_description = task.get("description", "Unknown task")

        # Create decomposition request
        decomp_request = DecompositionRequest(
            task_description=task_description,
            max_depth=5,
            max_subtasks_per_level=5,
            complexity_threshold=0.2,
            include_dependencies=True,
            estimate_costs=True,
            target_model_routing=True,
        )

        # Decompose task
        decomp_result = await self.decomposition_service.decompose_task(decomp_request)

        # Convert tree to plan format (compatible with existing code)
        tree = decomp_result.tree
        subtasks = []

        # Use leaf tasks as subtasks
        for task_id in tree.leaf_tasks:
            decomp_task = tree.tasks[task_id]

            subtask = {
                "id": decomp_task.id,
                "description": decomp_task.description,
                "status": decomp_task.status.value,
                "dependencies": list(decomp_task.dependencies),
                "estimated_complexity": decomp_task.estimated_complexity,
                "estimated_cost": decomp_task.estimated_cost,
                "assigned_model": decomp_task.assigned_model,
            }
            subtasks.append(subtask)

        # Extract success criteria from tree
        success_criteria = [
            f"Complete all {len(tree.leaf_tasks)} subtasks",
            f"Maintain cost under ${tree.estimated_total_cost:.2f}",
            "All dependencies resolved",
            "Tests passing",
        ]

        plan = {
            "plan_id": str(uuid.uuid4()),
            "task_id": task.get("id", "unknown"),
            "subtasks": subtasks,
            "decomposition_tree_id": tree.root_task_id,
            "estimated_effort": "medium",
            "estimated_cost": tree.estimated_total_cost,
            "success_criteria": success_criteria,
            "context_used": len(context.get("memories", [])),
            "generated_by": "decomposition_service",
            "execution_plan": decomp_result.execution_plan,
            "warnings": decomp_result.warnings,
        }

        self.logger.info(
            f"Generated plan with {len(subtasks)} subtasks using decomposition service "
            f"(cost: ${tree.estimated_total_cost:.2f})"
        )

        # Store tree in state for later reference
        state["decomposition_tree_id"] = tree.root_task_id

        return plan

    async def _create_plan_with_llm(
        self,
        task: dict,
        context: dict,
        state: AgentState,
    ) -> dict:
        """
        Create execution plan using LLM.

        PATTERN: Use LLM for intelligent task decomposition
        CRITICAL: Provide clear instructions and expected format

        Args:
            task: Task specification
            context: Retrieved context
            state: Current state

        Returns:
            Plan dictionary
        """
        import uuid
        import json

        task_description = task.get("description", "Unknown task")

        # Build context string from retrieved memories
        context_str = ""
        if context.get("memories"):
            context_str = "\n\nRelevant context from memory:\n"
            for mem in context["memories"][:3]:  # Use top 3
                context_str += f"- {mem['content']}\n"

        # Create planning prompt
        system_prompt = """You are an expert planning agent for software development workflows.
Your role is to decompose high-level tasks into clear, actionable subtasks.

For each task, you must:
1. Break it down into 3-7 logical subtasks
2. Order subtasks by dependencies
3. Define success criteria
4. Estimate effort level (low, medium, high)

Output ONLY valid JSON in this format:
{
  "subtasks": [
    {
      "id": "subtask_1",
      "description": "Clear description of what to do",
      "status": "pending",
      "dependencies": []
    }
  ],
  "estimated_effort": "medium",
  "success_criteria": [
    "Criterion 1",
    "Criterion 2"
  ]
}"""

        user_prompt = f"""Task to plan: {task_description}{context_str}

Please create a detailed execution plan for this task. Break it into clear subtasks with dependencies."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # Generate plan using LLM
            response = await self.generate_llm_response(
                messages=messages,
                task_description=f"Planning task: {task_description}",
                temperature=0.7,  # Some creativity for planning
                max_tokens=2000,
            )

            # Parse JSON response (handle markdown code blocks)
            plan_data = self._extract_json_from_response(response)

            # Add metadata
            plan = {
                "plan_id": str(uuid.uuid4()),
                "task_id": task.get("id", "unknown"),
                "subtasks": plan_data.get("subtasks", []),
                "estimated_effort": plan_data.get("estimated_effort", "medium"),
                "success_criteria": plan_data.get("success_criteria", []),
                "context_used": len(context.get("memories", [])),
                "generated_by": "llm",
            }

            self.logger.info(
                f"Generated plan with {len(plan['subtasks'])} subtasks using LLM"
            )

            return plan

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            self.logger.debug(f"LLM response: {response}")
            # Fall back to basic planning
            return self._create_plan_fallback(task, context)

        except Exception as e:
            self.logger.error(f"LLM planning failed: {e}")
            # Fall back to basic planning
            return self._create_plan_fallback(task, context)

    def _extract_json_from_response(self, response: str) -> dict:
        """
        Extract JSON from LLM response, handling markdown and extra text.

        Args:
            response: LLM response text

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON cannot be extracted
        """
        import json
        import re

        # Try direct parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in text (without markdown)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: raise error with helpful message
        self.logger.error(f"Could not extract valid JSON from LLM response: {response[:500]}")
        raise ValueError(f"Could not extract valid JSON from LLM response. Response preview: {response[:200]}...")

    def _create_plan_fallback(
        self,
        task: dict,
        context: dict,
    ) -> dict:
        """
        Create basic plan without LLM (fallback).

        Args:
            task: Task specification
            context: Retrieved context

        Returns:
            Plan dictionary
        """
        import uuid

        # Simplified planning logic
        subtasks = [
            {
                "id": f"subtask_{i}",
                "description": f"Step {i}: {task.get('description', 'Unknown')}",
                "status": "pending",
                "dependencies": [],
            }
            for i in range(1, 4)  # Create 3 sample subtasks
        ]

        return {
            "plan_id": str(uuid.uuid4()),
            "task_id": task.get("id", "unknown"),
            "subtasks": subtasks,
            "estimated_effort": "medium",
            "success_criteria": [
                "All subtasks completed",
                "Tests passing",
                "Documentation updated",
            ],
            "context_used": len(context.get("memories", [])),
            "generated_by": "fallback",
        }
