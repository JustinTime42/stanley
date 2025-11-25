"""Command history management."""

import logging
from pathlib import Path
from typing import Optional

from prompt_toolkit.history import FileHistory, History

from ..config.cli_config import CLIConfig

logger = logging.getLogger(__name__)


class HistoryManager:
    """
    Command history manager.

    Integrates with prompt_toolkit's FileHistory for persistence.
    """

    def __init__(
        self,
        config: CLIConfig,
        working_directory: Optional[str] = None,
    ):
        """
        Initialize history manager.

        Args:
            config: CLI configuration
            working_directory: Optional specific working directory
        """
        self.config = config
        self.max_size = config.max_history_size

        # Determine history file path
        if working_directory:
            # Per-directory history
            work_dir = Path(working_directory)
            history_dir = work_dir / ".agent-swarm"
        else:
            # Global history
            history_dir = Path.home() / ".agent-swarm"

        history_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = history_dir / "history"

        # Create prompt_toolkit history
        self._history: Optional[FileHistory] = None

    def get_history(self) -> History:
        """
        Get prompt_toolkit History instance.

        Returns:
            FileHistory instance
        """
        if self._history is None:
            self._history = FileHistory(str(self.history_path))
            logger.debug(f"Initialized history from {self.history_path}")

        return self._history

    def trim_history(self) -> int:
        """
        Trim history to max size.

        Returns:
            Number of entries removed
        """
        if not self.history_path.exists():
            return 0

        try:
            # Read all lines
            with open(self.history_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) <= self.max_size:
                return 0

            # Keep only the most recent entries
            removed = len(lines) - self.max_size
            lines = lines[-self.max_size :]

            # Write back
            with open(self.history_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            logger.info(f"Trimmed {removed} history entries")
            return removed

        except Exception as e:
            logger.warning(f"Failed to trim history: {e}")
            return 0

    def clear_history(self) -> None:
        """Clear all history."""
        try:
            if self.history_path.exists():
                self.history_path.unlink()
            # Reset the FileHistory instance
            self._history = None
            logger.info("Cleared history")
        except Exception as e:
            logger.warning(f"Failed to clear history: {e}")

    def get_recent(self, count: int = 10) -> list[str]:
        """
        Get recent history entries.

        Args:
            count: Number of entries to return

        Returns:
            List of recent commands
        """
        try:
            history = self.get_history()
            # FileHistory stores entries with newlines
            entries = list(history.get_strings())
            return entries[-count:] if len(entries) > count else entries
        except Exception as e:
            logger.warning(f"Failed to get recent history: {e}")
            return []

    def search(self, query: str, limit: int = 20) -> list[str]:
        """
        Search history for matching entries.

        Args:
            query: Search query (case-insensitive)
            limit: Maximum results

        Returns:
            List of matching entries
        """
        try:
            history = self.get_history()
            query_lower = query.lower()
            matches = [
                entry
                for entry in history.get_strings()
                if query_lower in entry.lower()
            ]
            return matches[-limit:]
        except Exception as e:
            logger.warning(f"Failed to search history: {e}")
            return []
