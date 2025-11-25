"""Configuration command implementation."""

from typing import Optional

from rich.table import Table

from .base import BaseCommand, CommandContext
from ..config.cli_config import save_config


class ConfigCommand(BaseCommand):
    """/config command for managing settings."""

    name = "config"
    description = "View or modify configuration"
    arguments = "[set <key> <value> | reset]"

    async def execute(
        self,
        args: str,
        context: CommandContext,
    ) -> Optional[str]:
        """Execute config command."""
        args_parts = args.strip().split(maxsplit=2)

        if not args_parts or not args_parts[0]:
            # Show current config
            return await self._show_config(context)

        subcommand = args_parts[0].lower()

        if subcommand == "set" and len(args_parts) >= 3:
            key = args_parts[1]
            value = args_parts[2]
            return await self._set_config(context, key, value)

        elif subcommand == "reset":
            return await self._reset_config(context)

        elif subcommand == "save":
            return await self._save_config(context)

        else:
            context.console.print(
                "[yellow]Usage: /config [set <key> <value> | reset | save][/yellow]"
            )
            return None

    async def _show_config(self, context: CommandContext) -> Optional[str]:
        """Show current configuration."""
        if not context.config:
            context.console.print("[red]Configuration not available[/red]")
            return None

        config = context.config

        table = Table(title="CLI Configuration", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value")
        table.add_column("Description", style="dim")

        # Display settings
        settings = [
            ("theme", config.theme, "Color theme"),
            ("show_tokens", config.show_tokens, "Show token count"),
            ("show_cost", config.show_cost, "Show cost information"),
            ("stream_output", config.stream_output, "Stream responses"),
            ("default_mode", config.default_mode, "Default CLI mode"),
            ("auto_save_sessions", config.auto_save_sessions, "Auto-save sessions"),
            ("max_history_size", config.max_history_size, "Max command history"),
            ("vim_mode", config.vim_mode, "Vim keybindings"),
            ("default_model", config.default_model or "auto", "Default model"),
            ("default_temperature", config.default_temperature, "Default temperature"),
        ]

        for key, value, description in settings:
            table.add_row(key, str(value), description)

        context.console.print(table)

        # Show config file locations
        context.console.print("\n[dim]Configuration files:[/dim]")
        context.console.print("  - ~/.agent-swarm/config.yaml (user)")
        context.console.print("  - ./.agent-swarm/config.yaml (project)")

        return None

    async def _set_config(
        self,
        context: CommandContext,
        key: str,
        value: str,
    ) -> Optional[str]:
        """Set a configuration value."""
        if not context.config:
            context.console.print("[red]Configuration not available[/red]")
            return None

        config = context.config

        # Type conversion based on key
        converted_value: any

        # Boolean settings
        bool_keys = [
            "show_tokens",
            "show_cost",
            "stream_output",
            "auto_save_sessions",
            "vim_mode",
        ]
        # Integer settings
        int_keys = ["max_history_size"]
        # Float settings
        float_keys = ["default_temperature"]

        if key in bool_keys:
            converted_value = value.lower() in ("true", "1", "yes", "on")
        elif key in int_keys:
            try:
                converted_value = int(value)
            except ValueError:
                context.console.print(f"[red]Invalid integer value: {value}[/red]")
                return None
        elif key in float_keys:
            try:
                converted_value = float(value)
            except ValueError:
                context.console.print(f"[red]Invalid float value: {value}[/red]")
                return None
        else:
            converted_value = value

        # Check if key exists
        if not hasattr(config, key):
            context.console.print(f"[red]Unknown setting: {key}[/red]")
            context.console.print("[dim]Use /config to see available settings[/dim]")
            return None

        # Set the value
        setattr(config, key, converted_value)
        context.console.print(f"[green]Set {key} = {converted_value}[/green]")
        context.console.print("[dim]Use /config save to persist changes[/dim]")

        return None

    async def _reset_config(self, context: CommandContext) -> Optional[str]:
        """Reset configuration to defaults."""
        from ..config.cli_config import CLIConfig

        if not context.config:
            context.console.print("[red]Configuration not available[/red]")
            return None

        # Create fresh config with defaults
        fresh_config = CLIConfig()

        # Copy values to current config
        for key, value in fresh_config.model_dump().items():
            if hasattr(context.config, key):
                setattr(context.config, key, value)

        context.console.print("[green]Configuration reset to defaults[/green]")
        context.console.print("[dim]Use /config save to persist changes[/dim]")

        return None

    async def _save_config(self, context: CommandContext) -> Optional[str]:
        """Save configuration to file."""
        if not context.config:
            context.console.print("[red]Configuration not available[/red]")
            return None

        try:
            save_config(context.config)
            context.console.print("[green]Configuration saved[/green]")
        except Exception as e:
            context.console.print(f"[red]Failed to save config: {e}[/red]")

        return None
