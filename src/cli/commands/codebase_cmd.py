"""/codebase command for codebase understanding."""

import asyncio
from typing import Optional
from pathlib import Path

from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.tree import Tree
from rich.syntax import Syntax

from .base import BaseCommand, CommandContext


class CodebaseCommand(BaseCommand):
    """/codebase command family for codebase understanding."""

    name = "codebase"
    description = "Analyze and query codebase understanding"
    arguments = "[analyze|find|deps|symbols|gaps|watch|status] [args...]"
    aliases = ["cb", "code"]

    def __init__(self):
        """Initialize codebase command."""
        super().__init__()
        self._service = None

    async def _get_service(self, context: CommandContext):
        """Get or create understanding service."""
        if self._service is None:
            try:
                from ...services.understanding_service import UnderstandingService

                # Use current working directory or session workspace
                root_path = context.session.working_directory if hasattr(context.session, "working_directory") else "."
                root_path = Path(root_path).resolve()

                self._service = UnderstandingService(
                    root_path=str(root_path),
                    memory_service=context.memory,
                )

                # Try to load existing understanding
                await self._service.load_existing()

            except ImportError as e:
                context.console.print(f"[red]Understanding service not available: {e}[/red]")
                return None

        return self._service

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """Execute codebase command."""
        args_parts = args.strip().split(maxsplit=1)

        if not args_parts or not args_parts[0]:
            # Show status
            return await self._show_status(context)

        subcommand = args_parts[0].lower()
        sub_args = args_parts[1] if len(args_parts) > 1 else ""

        if subcommand == "analyze":
            return await self._analyze(context, sub_args)
        elif subcommand == "find":
            return await self._find(context, sub_args)
        elif subcommand == "deps":
            return await self._show_deps(context, sub_args)
        elif subcommand == "symbols":
            return await self._show_symbols(context, sub_args)
        elif subcommand == "gaps":
            return await self._show_gaps(context)
        elif subcommand == "watch":
            return await self._manage_watcher(context, sub_args)
        elif subcommand == "status":
            return await self._show_status(context)
        elif subcommand == "refresh":
            return await self._refresh(context)
        elif subcommand == "verify":
            return await self._verify(context, sub_args)
        else:
            context.console.print(
                "[yellow]Usage: /codebase [analyze|find|deps|symbols|gaps|watch|status|refresh|verify][/yellow]"
            )
            return None

    async def _show_status(self, context: CommandContext) -> Optional[str]:
        """Show understanding status."""
        service = await self._get_service(context)
        if not service:
            return None

        stats = service.get_statistics()

        table = Table(title="Codebase Understanding", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value")

        if stats.get("analyzed"):
            table.add_row("Project", stats.get("project_name", "Unknown"))
            table.add_row("Type", stats.get("detected_type", "Unknown"))
            if stats.get("framework"):
                table.add_row("Framework", stats.get("framework"))
            table.add_row("Files", str(stats.get("total_files", 0)))
            table.add_row("Symbols", str(stats.get("total_symbols", 0)))
            table.add_row("Lines", f"{stats.get('total_lines', 0):,}")
            table.add_row("Analysis Mode", stats.get("analysis_mode", "Unknown"))
            table.add_row("Analysis Time", f"{stats.get('analysis_time_seconds', 0):.1f}s")
            table.add_row("Knowledge Gaps", str(stats.get("knowledge_gaps", 0)))
            table.add_row("Watcher", "Active" if stats.get("watcher_active") else "Inactive")

            # Confidence breakdown
            if "confidence" in stats:
                conf = stats["confidence"]
                table.add_row("---", "---")
                table.add_row("Verified", f"{conf.get('verified', 0)} ({conf.get('verified_pct', 0):.1f}%)")
                table.add_row("High Confidence", f"{conf.get('high_confidence_pct', 0):.1f}%")
        else:
            table.add_row("Status", "[yellow]Not analyzed[/yellow]")
            table.add_row("", "Run /codebase analyze to analyze")

        context.console.print(table)
        return None

    async def _analyze(self, context: CommandContext, args: str) -> Optional[str]:
        """Analyze codebase."""
        service = await self._get_service(context)
        if not service:
            return None

        # Parse mode
        mode = "deep"
        if "quick" in args.lower():
            mode = "quick"
        elif "deep" in args.lower():
            mode = "deep"

        context.console.print(f"[blue]Starting {mode} analysis...[/blue]")

        # Progress callback
        def progress_callback(progress):
            # This is called from analyzer - just log
            pass

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=context.console,
        ) as progress:
            task = progress.add_task(f"Analyzing ({mode})...", total=100)

            try:
                understanding = await service.analyze(mode=mode)
                progress.update(task, completed=100)

                context.console.print(f"\n[green]Analysis complete![/green]")
                context.console.print(f"- Files: {len(understanding.files)}")
                context.console.print(f"- Symbols: {len(understanding.symbols)}")
                context.console.print(f"- Time: {understanding.total_analysis_time_seconds:.1f}s")

            except Exception as e:
                context.console.print(f"[red]Analysis failed: {e}[/red]")

        return None

    async def _find(self, context: CommandContext, query: str) -> Optional[str]:
        """Find code by query."""
        if not query:
            context.console.print("[red]Please provide a search query[/red]")
            return None

        service = await self._get_service(context)
        if not service:
            return None

        if not service.is_analyzed:
            context.console.print("[yellow]Codebase not analyzed. Run /codebase analyze first.[/yellow]")
            return None

        context.console.print(f"[blue]Searching for: {query}[/blue]\n")

        try:
            response = await service.query(query)

            context.console.print(Panel(
                response.answer,
                title=f"Results (confidence: {response.confidence.value})",
                border_style="blue",
            ))

            if response.sources:
                context.console.print("\n[dim]Sources:[/dim]")
                for source in response.sources[:5]:
                    context.console.print(f"  - {source}")

            if response.knowledge_gaps:
                context.console.print("\n[yellow]Knowledge gaps:[/yellow]")
                for gap in response.knowledge_gaps[:3]:
                    context.console.print(f"  - {gap}")

        except Exception as e:
            context.console.print(f"[red]Search failed: {e}[/red]")

        return None

    async def _show_deps(self, context: CommandContext, module: str) -> Optional[str]:
        """Show dependencies."""
        service = await self._get_service(context)
        if not service:
            return None

        if not service.understanding:
            context.console.print("[yellow]Codebase not analyzed.[/yellow]")
            return None

        graph = service.understanding.dependency_graph

        if module:
            # Show specific module dependencies
            deps = graph.get_dependencies(module)
            dependents = graph.get_dependents(module)

            context.console.print(Panel(
                f"[cyan]Module:[/cyan] {module}",
                title="Dependencies",
            ))

            if deps:
                context.console.print("\n[green]Depends on:[/green]")
                for d in deps[:20]:
                    context.console.print(f"  -> {d}")

            if dependents:
                context.console.print("\n[yellow]Depended on by:[/yellow]")
                for d in dependents[:20]:
                    context.console.print(f"  <- {d}")
        else:
            # Show summary
            table = Table(title="Dependency Graph")
            table.add_column("Metric")
            table.add_column("Value")

            table.add_row("Modules", str(len(graph.nodes)))
            table.add_row("Edges", str(len(graph.edges)))
            table.add_row("Entry Points", str(len(graph.entry_points)))
            table.add_row("Leaf Modules", str(len(graph.leaf_modules)))
            table.add_row("Cycles", str(len(graph.cycles)))

            context.console.print(table)

            if graph.entry_points:
                context.console.print("\n[green]Entry points:[/green]")
                for ep in graph.entry_points[:10]:
                    context.console.print(f"  - {ep}")

            if graph.cycles:
                context.console.print("\n[red]Circular dependencies:[/red]")
                for cycle in graph.cycles[:5]:
                    context.console.print(f"  - {' -> '.join(cycle)}")

        return None

    async def _show_symbols(self, context: CommandContext, file_filter: str) -> Optional[str]:
        """Show symbols."""
        service = await self._get_service(context)
        if not service:
            return None

        if not service.understanding:
            context.console.print("[yellow]Codebase not analyzed.[/yellow]")
            return None

        symbols = list(service.knowledge_store.get_all_symbols().values())

        if file_filter:
            symbols = [s for s in symbols if file_filter in s.file_path]

        # Group by kind
        by_kind = {}
        for sym in symbols:
            kind = sym.kind.value
            if kind not in by_kind:
                by_kind[kind] = []
            by_kind[kind].append(sym)

        table = Table(title=f"Symbols ({len(symbols)} total)")
        table.add_column("Kind")
        table.add_column("Count")

        for kind, syms in sorted(by_kind.items(), key=lambda x: -len(x[1])):
            table.add_row(kind, str(len(syms)))

        context.console.print(table)

        # Show sample symbols
        if symbols and not file_filter:
            context.console.print("\n[dim]Sample symbols (use /codebase symbols <file> for specific file):[/dim]")
            for sym in symbols[:10]:
                context.console.print(f"  {sym.kind.value}: {sym.qualified_name}")

        elif file_filter and symbols:
            context.console.print(f"\n[cyan]Symbols in {file_filter}:[/cyan]")
            for sym in symbols[:30]:
                line_info = f":{sym.line_start}" if sym.line_start else ""
                context.console.print(f"  {sym.kind.value}: {sym.name}{line_info}")

        return None

    async def _show_gaps(self, context: CommandContext) -> Optional[str]:
        """Show knowledge gaps."""
        service = await self._get_service(context)
        if not service:
            return None

        if not service.understanding:
            context.console.print("[yellow]Codebase not analyzed.[/yellow]")
            return None

        report = await service.what_dont_i_know()
        context.console.print(Panel(report, title="Knowledge Gaps", border_style="yellow"))

        return None

    async def _manage_watcher(self, context: CommandContext, action: str) -> Optional[str]:
        """Manage file watcher."""
        service = await self._get_service(context)
        if not service:
            return None

        action = action.lower().strip()

        if action == "start":
            if service.start_watcher():
                context.console.print("[green]File watcher started[/green]")
            else:
                context.console.print("[red]Failed to start watcher. Is watchdog installed?[/red]")

        elif action == "stop":
            service.stop_watcher()
            context.console.print("[yellow]File watcher stopped[/yellow]")

        elif action == "status" or not action:
            status = service.get_watcher_status()
            table = Table(title="Watcher Status", show_header=False)
            table.add_column("Property")
            table.add_column("Value")

            table.add_row("Running", "[green]Yes[/green]" if status.get("running") else "[red]No[/red]")
            table.add_row("Root Path", str(status.get("root_path", "N/A")))
            table.add_row("Debounce", f"{status.get('debounce_seconds', 0)}s")
            table.add_row("Pending Changes", str(status.get("pending_changes", 0)))

            context.console.print(table)
        else:
            context.console.print("[yellow]Usage: /codebase watch [start|stop|status][/yellow]")

        return None

    async def _refresh(self, context: CommandContext) -> Optional[str]:
        """Refresh analysis."""
        service = await self._get_service(context)
        if not service:
            return None

        context.console.print("[blue]Refreshing analysis...[/blue]")

        try:
            understanding = await service.refresh()
            context.console.print(f"[green]Refresh complete! {len(understanding.symbols)} symbols[/green]")
        except Exception as e:
            context.console.print(f"[red]Refresh failed: {e}[/red]")

        return None

    async def _verify(self, context: CommandContext, claim: str) -> Optional[str]:
        """Verify a claim."""
        if not claim:
            context.console.print("[red]Please provide a claim to verify[/red]")
            return None

        service = await self._get_service(context)
        if not service:
            return None

        if not service.is_analyzed:
            context.console.print("[yellow]Codebase not analyzed.[/yellow]")
            return None

        result = await service.verify_claim(claim)

        # Format result
        if result.verified:
            status = "[green]Verified[/green]"
        else:
            status = "[red]Not Verified[/red]"

        context.console.print(Panel(
            f"**Claim**: {result.claim}\n\n"
            f"**Status**: {status}\n"
            f"**Confidence**: {result.confidence.value}\n\n"
            + (f"**Supporting Evidence**:\n" + "\n".join(f"  - {e}" for e in result.supporting_evidence) if result.supporting_evidence else "")
            + (f"\n\n**Contradicting Evidence**:\n" + "\n".join(f"  - {e}" for e in result.contradicting_evidence) if result.contradicting_evidence else "")
            + (f"\n\n**Correction**: {result.correction}" if result.correction else ""),
            title="Verification Result",
            border_style="blue" if result.verified else "yellow",
        ))

        return None
