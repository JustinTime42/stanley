"""Rich output rendering for CLI."""

import logging
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.traceback import Traceback

from .themes import get_theme
from ..config.cli_config import CLIConfig

logger = logging.getLogger(__name__)


class OutputRenderer:
    """
    Rich output renderer for CLI.

    Handles markdown, code blocks, tables, and errors.
    """

    def __init__(
        self,
        config: Optional[CLIConfig] = None,
        console: Optional[Console] = None,
    ):
        """
        Initialize output renderer.

        Args:
            config: CLI configuration
            console: Rich console (creates new if not provided)
        """
        self.config = config or CLIConfig()
        self.console = console or Console()
        self.theme = get_theme(self.config.theme)

    def render(self, content: str, style: Optional[str] = None) -> None:
        """
        Render content to console.

        Args:
            content: Content to render
            style: Optional style override
        """
        if style:
            self.console.print(content, style=style)
        else:
            self.console.print(content)

    def render_markdown(self, content: str) -> None:
        """
        Render markdown content.

        Args:
            content: Markdown content
        """
        md = Markdown(content, code_theme=self.theme.code_style)
        self.console.print(md)

    def render_message(
        self,
        content: str,
        role: str = "assistant",
        stream: bool = False,
    ) -> None:
        """
        Render a chat message.

        Args:
            content: Message content
            role: Message role (user, assistant, system)
            stream: If True, render for streaming (no newline)
        """
        # Get role color
        if role == "user":
            color = self.theme.user_color
            prefix = "You"
        elif role == "assistant":
            color = self.theme.assistant_color
            prefix = "Assistant"
        elif role == "system":
            color = self.theme.system_color
            prefix = "System"
        else:
            color = self.theme.dim_color
            prefix = role.capitalize()

        if stream:
            # For streaming, just print content without header
            self.console.print(content, end="")
        else:
            # Full message with header
            self.console.print(f"\n[bold {color}]{prefix}:[/bold {color}]")
            self.render_markdown(content)

    def render_code(
        self,
        code: str,
        language: str = "python",
        line_numbers: bool = True,
    ) -> None:
        """
        Render code with syntax highlighting.

        Args:
            code: Code content
            language: Programming language
            line_numbers: Show line numbers
        """
        syntax = Syntax(
            code,
            language,
            theme=self.theme.code_style,
            line_numbers=line_numbers,
        )
        self.console.print(syntax)

    def render_panel(
        self,
        content: str,
        title: Optional[str] = None,
        style: str = "blue",
    ) -> None:
        """
        Render content in a panel.

        Args:
            content: Panel content
            title: Panel title
            style: Border style
        """
        panel = Panel(content, title=title, border_style=style)
        self.console.print(panel)

    def render_table(
        self,
        data: list[dict],
        title: Optional[str] = None,
        columns: Optional[list[str]] = None,
    ) -> None:
        """
        Render data as a table.

        Args:
            data: List of row dictionaries
            title: Table title
            columns: Column names (auto-detect if not provided)
        """
        if not data:
            self.console.print("[dim]No data[/dim]")
            return

        # Auto-detect columns
        if not columns:
            columns = list(data[0].keys())

        table = Table(title=title, show_header=True)

        for col in columns:
            table.add_column(col.replace("_", " ").title())

        for row in data:
            table.add_row(*[str(row.get(col, "")) for col in columns])

        self.console.print(table)

    def render_error(self, error: Exception | str) -> None:
        """
        Render an error message.

        Args:
            error: Exception or error message
        """
        if isinstance(error, Exception):
            # Show rich traceback for exceptions
            self.console.print(f"[{self.theme.error_color}]Error: {error}[/{self.theme.error_color}]")

            # Show traceback in debug mode
            if logger.isEnabledFor(logging.DEBUG):
                self.console.print(Traceback())
        else:
            self.console.print(f"[{self.theme.error_color}]{error}[/{self.theme.error_color}]")

    def render_warning(self, message: str) -> None:
        """
        Render a warning message.

        Args:
            message: Warning message
        """
        self.console.print(f"[{self.theme.warning_color}]{message}[/{self.theme.warning_color}]")

    def render_success(self, message: str) -> None:
        """
        Render a success message.

        Args:
            message: Success message
        """
        self.console.print(f"[{self.theme.success_color}]{message}[/{self.theme.success_color}]")

    def render_info(self, message: str) -> None:
        """
        Render an info message.

        Args:
            message: Info message
        """
        self.console.print(f"[{self.theme.info_color}]{message}[/{self.theme.info_color}]")

    def render_dim(self, message: str) -> None:
        """
        Render a dimmed message.

        Args:
            message: Message to dim
        """
        self.console.print(f"[{self.theme.dim_color}]{message}[/{self.theme.dim_color}]")

    def render_status(
        self,
        tokens: int = 0,
        cost: float = 0.0,
        model: Optional[str] = None,
    ) -> None:
        """
        Render status line with token/cost info.

        Args:
            tokens: Token count
            cost: Cost in USD
            model: Model name
        """
        parts = []

        if self.config.show_tokens and tokens > 0:
            parts.append(f"tokens: {tokens:,}")

        if self.config.show_cost and cost > 0:
            parts.append(f"cost: ${cost:.4f}")

        if model:
            parts.append(f"model: {model}")

        if parts:
            status = " | ".join(parts)
            self.console.print(f"[{self.theme.dim_color}]{status}[/{self.theme.dim_color}]")

    def clear(self) -> None:
        """Clear the console."""
        self.console.clear()

    def newline(self) -> None:
        """Print a newline."""
        self.console.print()
