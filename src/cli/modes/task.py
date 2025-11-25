"""Task mode for multi-agent autonomous workflows."""

import logging
from datetime import datetime

from .base import BaseMode
from ...models.workflow_models import WorkflowConfig

logger = logging.getLogger(__name__)


class TaskMode(BaseMode):
    """
    Multi-agent autonomous workflow mode.

    Integrates with WorkflowOrchestrator for complex tasks.
    """

    name = "task"
    description = "Multi-agent autonomous workflow mode"
    prompt_prefix = "Task: "

    async def process(self, message: str) -> None:
        """
        Process a task request.

        PATTERN: Create workflow config -> Start workflow -> Display progress

        Args:
            message: Task description
        """
        self.console.print("\n[bold yellow]Starting autonomous task...[/bold yellow]")

        if not self.workflow:
            self.console.print("[red]Workflow service not available[/red]")
            self.console.print("[dim]Task mode requires the WorkflowOrchestrator[/dim]")
            return

        # Create task specification
        task = {
            "id": f"cli_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "description": message,
            "requirements": [],  # Could parse from message
            "context": {
                "working_directory": self.session.working_directory,
                "session_id": self.session.session_id,
            },
        }

        # Create workflow configuration
        config = WorkflowConfig(
            project_id=f"cli_{self.session.session_id}",
            enable_human_approval=True,  # Ask before executing
            max_retries=3,
        )

        try:
            # Update session state
            self.session.active_workflow_id = config.workflow_id
            self.session.workflow_status = "starting"

            # Show starting message
            self.console.print(f"[dim]Workflow ID: {config.workflow_id}[/dim]")
            self.console.print("[dim]Human approval: enabled[/dim]")
            self.console.print()

            # Start workflow
            execution = await self.workflow.start_workflow(
                task=task,
                config=config,
            )

            # Update session state
            self.session.workflow_status = execution.status.value if hasattr(execution.status, 'value') else str(execution.status)

            # Display results
            self._display_results(execution)

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Task interrupted[/yellow]")
            self.session.workflow_status = "interrupted"

            # Try to cancel the workflow
            if self.workflow and self.session.active_workflow_id:
                try:
                    await self.workflow.cancel_workflow(self.session.active_workflow_id)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Task failed: {e}")
            self.console.print(f"\n[bold red]✗ Task failed: {e}[/bold red]")
            self.session.workflow_status = "failed"

        self.console.print()

    def _display_results(self, execution) -> None:
        """
        Display workflow execution results.

        Args:
            execution: WorkflowExecution result
        """
        status = execution.status.value if hasattr(execution.status, 'value') else str(execution.status)

        if status in ("complete", "completed"):
            self.console.print("\n[bold green]✓ Task complete![/bold green]")
        elif status == "failed":
            self.console.print("\n[bold red]✗ Task failed[/bold red]")
            if execution.error:
                self.console.print(f"[red]Error: {execution.error}[/red]")
        elif status == "human_review":
            self.console.print("\n[bold yellow]⏸ Waiting for approval[/bold yellow]")
        else:
            self.console.print(f"\n[bold blue]Status: {status}[/bold blue]")

        # Show statistics
        self.console.print(f"[dim]Duration: {execution.elapsed_time_seconds:.1f}s[/dim]")

        if hasattr(execution, 'total_cost_usd') and execution.total_cost_usd > 0:
            self.console.print(f"[dim]Cost: ${execution.total_cost_usd:.4f}[/dim]")

        if hasattr(execution, 'total_tokens') and execution.total_tokens > 0:
            self.console.print(f"[dim]Tokens: {execution.total_tokens:,}[/dim]")

    async def resume_workflow(self) -> None:
        """Resume a paused workflow."""
        if not self.session.active_workflow_id:
            self.console.print("[yellow]No active workflow to resume[/yellow]")
            return

        if not self.workflow:
            self.console.print("[red]Workflow service not available[/red]")
            return

        try:
            self.console.print(f"[dim]Resuming workflow {self.session.active_workflow_id}...[/dim]")

            execution = await self.workflow.resume_workflow(
                workflow_id=self.session.active_workflow_id,
            )

            self.session.workflow_status = execution.status.value if hasattr(execution.status, 'value') else str(execution.status)
            self._display_results(execution)

        except Exception as e:
            logger.error(f"Failed to resume workflow: {e}")
            self.console.print(f"[red]Failed to resume: {e}[/red]")

    async def get_workflow_status(self) -> None:
        """Display current workflow status."""
        if not self.session.active_workflow_id:
            self.console.print("[dim]No active workflow[/dim]")
            return

        if not self.workflow:
            self.console.print("[red]Workflow service not available[/red]")
            return

        try:
            execution = await self.workflow.get_workflow_status(
                self.session.active_workflow_id
            )

            if execution:
                self._display_results(execution)
            else:
                self.console.print("[dim]Workflow not found[/dim]")

        except Exception as e:
            self.console.print(f"[red]Status check failed: {e}[/red]")

    def get_prompt(self) -> str:
        """
        Get the prompt string for task mode.

        Returns:
            Prompt with emoji indicator
        """
        return "🤖 Task: "

    async def on_enter(self) -> None:
        """Called when entering task mode."""
        self.renderer.render_info("Entered task mode (autonomous multi-agent)")
        self.renderer.render_dim("Tasks will be executed by the agent swarm")

    async def on_exit(self) -> None:
        """Called when exiting task mode."""
        # Check for active workflow
        if self.session.active_workflow_id and self.session.workflow_status in ("running", "human_review"):
            self.renderer.render_warning(
                f"Note: Workflow {self.session.active_workflow_id} may still be running"
            )
