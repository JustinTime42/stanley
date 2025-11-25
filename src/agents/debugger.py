"""Debugger agent for error resolution and debugging."""

import logging
import uuid
from typing import Optional, List, Dict, Any

from .base import BaseAgent
from ..models.state_models import AgentState, AgentRole, WorkflowStatus
from ..models.agent_models import AgentResponse
from ..services.memory_service import MemoryOrchestrator

logger = logging.getLogger(__name__)


class DebuggerAgent(BaseAgent):
    """
    Debugger agent responsible for error resolution and debugging.

    This agent actually analyzes failures and uses the LLM to generate
    and apply fixes to the implementation code.

    Responsibilities:
    - Analyze test failures and validation errors
    - Read the failing implementation code
    - Use LLM to generate fixes
    - Write fixes back to files
    - Report actual success/failure
    """

    def __init__(
        self,
        memory_service: Optional[MemoryOrchestrator] = None,
        llm_service: Optional["LLMOrchestrator"] = None,
        tool_service: Optional["ToolOrchestrator"] = None,
    ):
        """Initialize debugger agent with all required services."""
        super().__init__(
            role=AgentRole.DEBUGGER,
            memory_service=memory_service,
            llm_service=llm_service,
            tool_service=tool_service,
        )

    async def execute(self, state: AgentState) -> AgentResponse:
        """
        Execute debugging logic - actually fix code issues.

        Args:
            state: Current workflow state

        Returns:
            AgentResponse with debugging results
        """
        try:
            self.logger.info(f"Debugger analyzing {state.get('workflow_id')}")

            # Get relevant state
            test_results = state.get("test_results") or {}
            validation_results = state.get("validation_results") or {}
            implementation = state.get("implementation") or {}
            task = state.get("task") or {}
            retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", 3)

            # Check if we have LLM service for actual debugging
            if not self.llm_service:
                self.logger.error("No LLM service available - cannot debug")
                return self._create_error_response(
                    error="Debugger requires LLM service to analyze and fix issues"
                )

            # Analyze issues
            issues = self._identify_issues(test_results, validation_results)

            if not issues:
                self.logger.info("No issues to debug - workflow should not have reached debugger")
                return self._create_success_response(
                    result={"fixed": True, "issues": [], "message": "No issues found"},
                    next_agent=AgentRole.VALIDATOR,
                    state_updates={
                        "debug_info": {"fixed": True, "issues": []},
                        "status": WorkflowStatus.VALIDATING.value,
                    },
                )

            self.logger.info(f"Found {len(issues)} issues to debug")

            # Attempt to fix the issues using LLM
            debug_info = await self._fix_issues(
                issues=issues,
                implementation=implementation,
                task=task,
                test_results=test_results,
                validation_results=validation_results,
            )

            # Store debug info in memory
            await self.store_result(
                content=f"Debug session: {len(debug_info['fixes'])} fixes attempted, "
                        f"{debug_info['successful_fixes']} successful",
                state=state,
                importance=0.8,
                tags=["debugging", "fixes"],
            )

            # Create message
            if debug_info["fixed"]:
                message_content = (
                    f"Debugging complete: Applied {debug_info['successful_fixes']} fixes. "
                    f"Retrying tests (attempt {retry_count + 1}/{max_retries})."
                )
                message_type = "success"
            else:
                message_content = (
                    f"Debugging failed: Could not fix {len(issues)} issues. "
                    f"Attempted {len(debug_info['fixes'])} fixes."
                )
                message_type = "error"

            debug_message = self.create_message(
                content=message_content,
                message_type=message_type,
                metadata={"debug_session_id": debug_info["debug_session_id"]},
            )

            # Determine next step
            if debug_info["fixed"]:
                next_agent = AgentRole.TESTER
                next_status = WorkflowStatus.TESTING.value
                new_retry_count = retry_count + 1
            else:
                # Could not fix - fail the workflow
                next_agent = None
                next_status = WorkflowStatus.FAILED.value
                new_retry_count = retry_count

            # State updates - include updated implementation if fixes were applied
            state_updates = {
                "debug_info": debug_info,
                "status": next_status,
                "retry_count": new_retry_count,
            }

            # If we modified implementation, update it in state
            if debug_info.get("updated_implementation"):
                state_updates["implementation"] = debug_info["updated_implementation"]

            return self._create_success_response(
                result=debug_info,
                next_agent=next_agent,
                state_updates=state_updates,
                messages=[debug_message],
            )

        except Exception as e:
            self.logger.error(f"Debugger execution failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._create_error_response(
                error=f"Debugging failed: {str(e)}",
            )

    def _identify_issues(
        self,
        test_results: dict,
        validation_results: dict,
    ) -> List[Dict[str, Any]]:
        """
        Identify specific issues from test and validation results.

        Args:
            test_results: Test execution results
            validation_results: Validation check results

        Returns:
            List of identified issues with details
        """
        issues = []

        # Check test failures
        if not test_results.get("all_passed", True):
            failed_count = test_results.get("failed", 0)
            error_msg = test_results.get("error", "")

            issues.append({
                "type": "test_failure",
                "description": f"{failed_count} tests failed" if failed_count else "Tests did not pass",
                "severity": "high",
                "details": {
                    "total": test_results.get("total", 0),
                    "passed": test_results.get("passed", 0),
                    "failed": failed_count,
                    "error": error_msg,
                    "coverage": test_results.get("coverage", 0),
                },
            })

        # Check validation failures
        if validation_results and not validation_results.get("approved", True):
            failed_checks = [
                check for check in validation_results.get("checks", [])
                if not check.get("passed", True)
            ]

            issues.append({
                "type": "validation_failure",
                "description": f"Validation failed: {len(failed_checks)} checks did not pass",
                "severity": "medium",
                "details": {
                    "failed_checks": failed_checks,
                    "status": validation_results.get("status", "unknown"),
                },
            })

        return issues

    async def _fix_issues(
        self,
        issues: List[Dict[str, Any]],
        implementation: dict,
        task: dict,
        test_results: dict,
        validation_results: dict,
    ) -> dict:
        """
        Attempt to fix identified issues using LLM.

        Args:
            issues: List of identified issues
            implementation: Current implementation details
            task: Original task specification
            test_results: Test results with failure details
            validation_results: Validation results

        Returns:
            Debug info with fix results
        """
        debug_session_id = str(uuid.uuid4())
        fixes = []
        successful_fixes = 0
        updated_files = []

        # Get implementation files
        impl_files = implementation.get("files", [])

        if not impl_files:
            self.logger.warning("No implementation files to fix")
            return {
                "debug_session_id": debug_session_id,
                "issues": issues,
                "fixes": [],
                "successful_fixes": 0,
                "fixed": False,
                "error": "No implementation files available to fix",
            }

        # Build context for the LLM
        issue_descriptions = "\n".join([
            f"- {issue['type']}: {issue['description']}"
            for issue in issues
        ])

        task_description = task.get("description", "Unknown task")
        task_requirements = task.get("requirements", [])
        requirements_text = "\n".join([f"- {req}" for req in task_requirements])

        # Process each file
        for file_info in impl_files:
            file_path = file_info.get("path", "")
            original_content = file_info.get("content", "")

            if not file_path or not original_content:
                continue

            # Generate fix using LLM
            try:
                fixed_content = await self._generate_fix_for_file(
                    file_path=file_path,
                    original_content=original_content,
                    issues=issue_descriptions,
                    task_description=task_description,
                    requirements=requirements_text,
                    test_results=test_results,
                )

                if fixed_content and fixed_content != original_content:
                    # Write the fixed content
                    write_success = await self._write_fixed_file(
                        file_path=file_path,
                        content=fixed_content,
                    )

                    if write_success:
                        successful_fixes += 1
                        updated_files.append({
                            "path": file_path,
                            "content": fixed_content,
                            "language": file_info.get("language", "python"),
                        })
                        fixes.append({
                            "file": file_path,
                            "action": "Modified file with LLM-generated fix",
                            "success": True,
                        })
                        self.logger.info(f"Successfully fixed {file_path}")
                    else:
                        fixes.append({
                            "file": file_path,
                            "action": "Generated fix but failed to write",
                            "success": False,
                        })
                else:
                    fixes.append({
                        "file": file_path,
                        "action": "LLM determined no changes needed or couldn't generate fix",
                        "success": False,
                    })

            except Exception as e:
                self.logger.error(f"Failed to fix {file_path}: {e}")
                fixes.append({
                    "file": file_path,
                    "action": f"Error: {str(e)}",
                    "success": False,
                })

        # Build updated implementation if we made changes
        updated_implementation = None
        if updated_files:
            updated_implementation = {
                **implementation,
                "files": updated_files,
                "debug_session_id": debug_session_id,
            }

        return {
            "debug_session_id": debug_session_id,
            "issues": issues,
            "fixes": fixes,
            "successful_fixes": successful_fixes,
            "fixed": successful_fixes > 0,
            "updated_implementation": updated_implementation,
        }

    async def _generate_fix_for_file(
        self,
        file_path: str,
        original_content: str,
        issues: str,
        task_description: str,
        requirements: str,
        test_results: dict,
    ) -> Optional[str]:
        """
        Use LLM to generate a fix for a file.

        Args:
            file_path: Path to the file
            original_content: Current file content
            issues: Description of issues to fix
            task_description: Original task description
            requirements: Task requirements
            test_results: Test results for context

        Returns:
            Fixed file content or None if no fix generated
        """
        # Build detailed prompt for the LLM
        error_details = ""
        if test_results.get("error"):
            error_details = f"\nError message: {test_results['error']}"

        prompt = f"""You are a debugging expert. Fix the following code to resolve the issues.

TASK: {task_description}

REQUIREMENTS:
{requirements}

CURRENT ISSUES:
{issues}
{error_details}

CURRENT CODE ({file_path}):
```
{original_content}
```

INSTRUCTIONS:
1. Analyze the issues and the current code
2. Fix all identified problems
3. Ensure the code meets all requirements
4. Make sure error handling is proper (e.g., division by zero)
5. Return ONLY the fixed code, no explanations

Return the complete fixed code:"""

        messages = [{"role": "user", "content": prompt}]

        try:
            # Use the base agent's LLM method
            response = await self.generate_llm_response(
                messages=messages,
                task_description=f"Fix code in {file_path}",
                temperature=0.3,
                max_tokens=4000,
            )

            # Clean up the response
            fixed_code = response.strip()

            # Remove markdown code blocks if present
            if fixed_code.startswith("```"):
                lines = fixed_code.split("\n")
                # Remove first line (```python or similar) and last line (```)
                if len(lines) > 2:
                    fixed_code = "\n".join(lines[1:-1])
                fixed_code = fixed_code.strip()

            return fixed_code if fixed_code else None

        except Exception as e:
            self.logger.error(f"LLM fix generation failed: {e}")
            return None

    async def _write_fixed_file(self, file_path: str, content: str) -> bool:
        """
        Write fixed content to file using tool service.

        Args:
            file_path: Path to write to
            content: Fixed content

        Returns:
            True if write succeeded
        """
        if not self.tool_service:
            self.logger.warning("No tool service - cannot write fixes")
            return False

        try:
            result = await self.execute_tool(
                tool_name="write_file",
                parameters={
                    "path": file_path,
                    "content": content,
                    "create_dirs": True,
                },
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to write fixed file {file_path}: {e}")
            return False
