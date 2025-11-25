"""
Agent Swarm CLI - Interactive command-line interface.

This package provides a Claude Code-style interactive CLI with:
- REPL with streaming responses
- Chat mode (single-agent conversation)
- Task mode (multi-agent autonomous workflows)
- Session persistence and resume
- Built-in and custom slash commands
- Rich terminal output with syntax highlighting
"""

from .app import main
from .repl import REPL, create_repl
from .config import CLIConfig, CLIMode, load_config
from .session import SessionState, SessionManager
from .commands import (
    BaseCommand,
    CommandContext,
    CommandRegistry,
)
from .output import OutputRenderer
from .modes import ChatMode, TaskMode

__all__ = [
    # Entry points
    "main",
    "REPL",
    "create_repl",
    # Configuration
    "CLIConfig",
    "CLIMode",
    "load_config",
    # Session
    "SessionState",
    "SessionManager",
    # Commands
    "BaseCommand",
    "CommandContext",
    "CommandRegistry",
    # Output
    "OutputRenderer",
    # Modes
    "ChatMode",
    "TaskMode",
]
