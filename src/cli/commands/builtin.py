"""Built-in slash commands."""

from typing import Optional

from rich.table import Table
from rich.panel import Panel

from .base import BaseCommand, CommandContext, CommandRegistry
from ..session.state import CLIMode


class HelpCommand(BaseCommand):
    """Show help information."""

    name = "help"
    description = "Show help for commands"
    arguments = "[command]"

    def __init__(self, registry: CommandRegistry):
        """Initialize with command registry reference."""
        self.registry = registry

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """Execute help command."""
        if args.strip():
            # Help for specific command
            cmd_name = args.strip().lstrip("/")
            command = self.registry.get(cmd_name)
            if command:
                context.console.print(Panel(
                    command.get_help(),
                    title=f"/{cmd_name}",
                    border_style="blue",
                ))
            else:
                context.console.print(f"[red]Unknown command: /{cmd_name}[/red]")
        else:
            # General help
            table = Table(title="Available Commands", show_header=True)
            table.add_column("Command", style="cyan")
            table.add_column("Description")

            for cmd in sorted(self.registry.list_commands(), key=lambda c: c.name):
                table.add_row(f"/{cmd.name}", cmd.description)

            context.console.print(table)
            context.console.print("\nType /help <command> for more details")

        return None


class ClearCommand(BaseCommand):
    """Clear conversation history."""

    name = "clear"
    description = "Clear conversation history"
    aliases = ["reset"]

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """Execute clear command."""
        context.session.clear_messages()
        context.console.print("[dim]Conversation cleared[/dim]")
        return None


class CostCommand(BaseCommand):
    """Show token usage and cost statistics."""

    name = "cost"
    description = "Show token usage and cost"
    aliases = ["tokens", "usage"]

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """Execute cost command."""
        session = context.session

        table = Table(title="Session Statistics", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Tokens", f"{session.total_tokens:,}")
        table.add_row("Total Cost", f"${session.total_cost:.4f}")
        table.add_row("Conversation Turns", str(session.turn_count))
        table.add_row("Messages", str(len(session.messages)))

        if context.llm:
            cache_stats = context.llm.get_cache_stats()
            if cache_stats.get("enabled"):
                table.add_row("Cache Hits", str(cache_stats.get("hits", 0)))
                table.add_row("Cache Savings", f"${cache_stats.get('cost_savings', 0):.4f}")

        context.console.print(table)
        return None


class ModelCommand(BaseCommand):
    """Show or change model."""

    name = "model"
    description = "Show or change the AI model"
    arguments = "[model-name]"

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """Execute model command."""
        if args.strip():
            # Change model
            model_name = args.strip()
            context.session.model = model_name
            context.console.print(f"[green]Model set to: {model_name}[/green]")
        else:
            # Show current model
            current = context.session.model or "auto (intelligent routing)"
            context.console.print(f"[cyan]Current model:[/cyan] {current}")

            if context.llm:
                available = context.llm.get_available_models()
                if available:
                    context.console.print("\n[cyan]Available models:[/cyan]")
                    for model in available:
                        context.console.print(f"  - {model}")

        return None


class ModeCommand(BaseCommand):
    """Switch between chat and task modes."""

    name = "mode"
    description = "Switch between chat and task modes"
    arguments = "[chat|task]"
    aliases = ["chat", "task"]

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """Execute mode command."""
        mode_arg = args.strip().lower()

        if mode_arg in ("chat", "task"):
            context.session.mode = CLIMode(mode_arg)
            mode_desc = "conversational" if mode_arg == "chat" else "autonomous multi-agent"
            context.console.print(f"[green]Switched to {mode_arg} mode ({mode_desc})[/green]")
        else:
            # Show current mode
            current = context.session.mode
            context.console.print(f"[cyan]Current mode:[/cyan] {current}")
            context.console.print("\nAvailable modes:")
            context.console.print("  - chat: Single-agent conversational mode")
            context.console.print("  - task: Multi-agent autonomous workflow mode")

        return None


class QuitCommand(BaseCommand):
    """Exit the CLI."""

    name = "quit"
    description = "Exit the CLI"
    aliases = ["exit", "q"]

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """Execute quit command."""
        # This is handled specially in the REPL
        raise SystemExit(0)


class SessionsCommand(BaseCommand):
    """List saved sessions."""

    name = "sessions"
    description = "List saved sessions"
    arguments = "[--all]"

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """Execute sessions command."""
        # This needs access to SessionManager which isn't in context
        # Will be implemented in REPL
        context.console.print("[yellow]Use --resume <id> to resume a session[/yellow]")
        return None


class SaveCommand(BaseCommand):
    """Save current session with a name."""

    name = "save"
    description = "Save current session with a name"
    arguments = "[name]"

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """Execute save command."""
        name = args.strip() if args.strip() else None
        if name:
            context.session.name = name
        context.console.print(
            f"[green]Session saved: {context.session.session_id}[/green]"
        )
        if name:
            context.console.print(f"[dim]Name: {name}[/dim]")
        return None


class StatusCommand(BaseCommand):
    """Show current status."""

    name = "status"
    description = "Show current session status"

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """Execute status command."""
        session = context.session

        table = Table(title="Session Status", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Session ID", session.session_id)
        table.add_row("Mode", session.mode)
        table.add_row("Model", session.model or "auto")
        table.add_row("Working Directory", session.working_directory)
        table.add_row("Created", session.created_at.strftime("%Y-%m-%d %H:%M"))
        table.add_row("Messages", str(len(session.messages)))
        table.add_row("Turns", str(session.turn_count))

        if session.active_workflow_id:
            table.add_row("Active Workflow", session.active_workflow_id)
            table.add_row("Workflow Status", session.workflow_status or "unknown")

        context.console.print(table)
        return None


class VimCommand(BaseCommand):
    """Toggle vim mode."""

    name = "vim"
    description = "Toggle vim keybindings"

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """Execute vim command."""
        if context.config:
            context.config.vim_mode = not context.config.vim_mode
            status = "enabled" if context.config.vim_mode else "disabled"
            context.console.print(f"[green]Vim mode {status}[/green]")
            context.console.print("[dim]Restart REPL to apply changes[/dim]")
        else:
            context.console.print("[red]Configuration not available[/red]")
        return None


class CompactCommand(BaseCommand):
    """Compact conversation history."""

    name = "compact"
    description = "Compact conversation history"
    arguments = "[instructions]"

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """Execute compact command."""
        # This would summarize the conversation
        # For now, just inform the user
        msg_count = len(context.session.messages)
        context.console.print(
            f"[yellow]Conversation has {msg_count} messages.[/yellow]"
        )
        context.console.print(
            "[dim]Compaction not yet implemented - use /clear to start fresh[/dim]"
        )
        return None


def load_builtin_commands(registry: CommandRegistry) -> None:
    """
    Load all built-in commands into registry.

    Args:
        registry: Command registry to populate
    """
    # Help needs registry reference
    registry.register(HelpCommand(registry))

    # Register other commands
    registry.register(ClearCommand())
    registry.register(CostCommand())
    registry.register(ModelCommand())
    registry.register(ModeCommand())
    registry.register(QuitCommand())
    registry.register(SessionsCommand())
    registry.register(SaveCommand())
    registry.register(StatusCommand())
    registry.register(VimCommand())
    registry.register(CompactCommand())
