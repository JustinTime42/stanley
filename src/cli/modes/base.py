"""Base mode class for CLI modes."""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from ..session.state import SessionState
    from ..config.cli_config import CLIConfig
    from ..output.renderer import OutputRenderer
    from ...services.llm_service import LLMOrchestrator
    from ...services.tool_service import ToolOrchestrator
    from ...services.memory_service import MemoryOrchestrator
    from ...services.workflow_service import WorkflowOrchestrator


class BaseMode(ABC):
    """
    Base class for CLI operating modes.

    Modes define how user input is processed:
    - Chat mode: Direct conversation with LLM
    - Task mode: Multi-agent autonomous workflow
    """

    name: str = "base"
    description: str = "Base mode"
    prompt_prefix: str = "> "

    def __init__(
        self,
        session: "SessionState",
        config: "CLIConfig",
        renderer: "OutputRenderer",
        console: Console,
        llm_service: Optional["LLMOrchestrator"] = None,
        tool_service: Optional["ToolOrchestrator"] = None,
        memory_service: Optional["MemoryOrchestrator"] = None,
        workflow_service: Optional["WorkflowOrchestrator"] = None,
    ):
        """
        Initialize base mode.

        Args:
            session: Current session state
            config: CLI configuration
            renderer: Output renderer
            console: Rich console
            llm_service: LLM orchestrator
            tool_service: Tool orchestrator
            memory_service: Memory orchestrator
            workflow_service: Workflow orchestrator
        """
        self.session = session
        self.config = config
        self.renderer = renderer
        self.console = console
        self.llm = llm_service
        self.tools = tool_service
        self.memory = memory_service
        self.workflow = workflow_service

    @abstractmethod
    async def process(self, message: str) -> None:
        """
        Process user input in this mode.

        Args:
            message: User input message
        """
        pass

    def get_prompt(self) -> str:
        """
        Get the prompt string for this mode.

        Returns:
            Prompt string
        """
        return self.prompt_prefix

    def get_system_prompt(self) -> Optional[str]:
        """
        Get the system prompt for this mode.

        Returns:
            System prompt or None
        """
        return None

    async def on_enter(self) -> None:
        """Called when entering this mode."""
        pass

    async def on_exit(self) -> None:
        """Called when exiting this mode."""
        pass
