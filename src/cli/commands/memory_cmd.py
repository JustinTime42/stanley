"""Memory command implementation."""

from typing import Optional

from rich.table import Table
from rich.panel import Panel

from .base import BaseCommand, CommandContext


class MemoryCommand(BaseCommand):
    """/memory command for context management."""

    name = "memory"
    description = "Manage conversation memory and context"
    arguments = "[search <query> | add <content> | clear | stats]"

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """Execute memory command."""
        args_parts = args.strip().split(maxsplit=1)

        if not args_parts or not args_parts[0]:
            # Show memory stats
            return await self._show_stats(context)

        subcommand = args_parts[0].lower()
        sub_args = args_parts[1] if len(args_parts) > 1 else ""

        if subcommand == "search":
            return await self._search_memory(context, sub_args)
        elif subcommand == "add":
            return await self._add_memory(context, sub_args)
        elif subcommand == "clear":
            return await self._clear_memory(context)
        elif subcommand == "stats":
            return await self._show_stats(context)
        else:
            context.console.print(
                "[yellow]Usage: /memory [search <query> | add <content> | clear | stats][/yellow]"
            )
            return None

    async def _show_stats(self, context: CommandContext) -> Optional[str]:
        """Show memory statistics."""
        table = Table(title="Memory Statistics", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value")

        # Session message stats
        session = context.session
        table.add_row("Session Messages", str(len(session.messages)))
        table.add_row("Session Tokens", f"{session.total_tokens:,}")

        # Memory service stats if available
        if context.memory:
            try:
                stats = context.memory.get_memory_stats()
                if "cache" in stats:
                    cache_stats = stats["cache"]
                    table.add_row("Cache Entries", str(cache_stats.get("size", 0)))
                    table.add_row("Cache Hits", str(cache_stats.get("hits", 0)))
                    table.add_row("Cache Misses", str(cache_stats.get("misses", 0)))
            except Exception:
                table.add_row("Memory Service", "[dim]unavailable[/dim]")
        else:
            table.add_row("Memory Service", "[dim]not connected[/dim]")

        context.console.print(table)
        return None

    async def _search_memory(
        self,
        context: CommandContext,
        query: str,
    ) -> Optional[str]:
        """Search memories."""
        if not query:
            context.console.print("[red]Please provide a search query[/red]")
            return None

        if not context.memory:
            context.console.print("[red]Memory service not available[/red]")
            return None

        try:
            results = await context.memory.retrieve_relevant_memories(
                query=query,
                k=5,
            )

            if not results:
                context.console.print("[dim]No matching memories found[/dim]")
                return None

            context.console.print(f"\n[bold]Found {len(results)} memories:[/bold]\n")

            for i, result in enumerate(results, 1):
                context.console.print(
                    Panel(
                        result.content[:500] + ("..." if len(result.content) > 500 else ""),
                        title=f"Memory {i} (score: {result.score:.3f})",
                        border_style="blue",
                    )
                )

        except Exception as e:
            context.console.print(f"[red]Search failed: {e}[/red]")

        return None

    async def _add_memory(
        self,
        context: CommandContext,
        content: str,
    ) -> Optional[str]:
        """Add content to project memory."""
        if not content:
            context.console.print("[red]Please provide content to add[/red]")
            return None

        if not context.memory:
            context.console.print("[red]Memory service not available[/red]")
            return None

        try:
            from ...models.memory_models import MemoryType

            memory_id = await context.memory.store_memory(
                content=content,
                agent_id="cli_user",
                memory_type=MemoryType.PROJECT,
                session_id=context.session.session_id,
                tags=["manual", "cli"],
            )

            context.console.print(f"[green]Memory added: {memory_id}[/green]")

        except Exception as e:
            context.console.print(f"[red]Failed to add memory: {e}[/red]")

        return None

    async def _clear_memory(self, context: CommandContext) -> Optional[str]:
        """Clear working memory."""
        if not context.memory:
            context.console.print("[red]Memory service not available[/red]")
            return None

        try:
            await context.memory.cleanup()
            context.console.print("[green]Working memory cleared[/green]")
        except Exception as e:
            context.console.print(f"[red]Failed to clear memory: {e}[/red]")

        return None
