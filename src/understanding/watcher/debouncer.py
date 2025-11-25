"""Change debouncing for rapid edits."""

import logging
import threading
import time
import fnmatch
from typing import Callable, Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class ChangeDebouncer:
    """
    Debounce rapid file changes.

    PATTERN: Batch rapid changes with configurable window
    CRITICAL: Deduplicate same-file events
    """

    def __init__(
        self,
        debounce_seconds: float = 1.0,
        callback: Optional[Callable[[List[str]], None]] = None,
    ):
        """
        Initialize debouncer.

        Args:
            debounce_seconds: Time window to batch changes
            callback: Function to call with batched changes
        """
        self.debounce_seconds = debounce_seconds
        self.callback = callback

        self._pending: Dict[str, float] = {}
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def add_change(self, file_path: str) -> None:
        """
        Add a file change to be debounced.

        Args:
            file_path: Path that changed
        """
        with self._lock:
            self._pending[file_path] = time.time()

            # Reset timer
            if self._timer:
                self._timer.cancel()

            self._timer = threading.Timer(
                self.debounce_seconds,
                self._flush,
            )
            self._timer.start()

    def _flush(self) -> None:
        """Flush pending changes to callback."""
        with self._lock:
            if self._pending and self.callback:
                files = list(self._pending.keys())
                self._pending.clear()

                try:
                    self.callback(files)
                except Exception as e:
                    logger.error(f"Error in debounce callback: {e}")

    def cancel(self) -> None:
        """Cancel pending flush."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            self._pending.clear()

    def get_pending_count(self) -> int:
        """Get number of pending changes."""
        with self._lock:
            return len(self._pending)


class DebouncedHandler:
    """
    Watchdog-compatible debounced event handler.

    PATTERN: Wrap file system events with debouncing
    """

    def __init__(
        self,
        callback: Callable[[List[str]], None],
        debounce_seconds: float = 1.0,
        ignore_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize handler.

        Args:
            callback: Function to call with changed files
            debounce_seconds: Debounce window
            ignore_patterns: Patterns to ignore
        """
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.ignore_patterns = ignore_patterns or []

        self._debouncer = ChangeDebouncer(
            debounce_seconds=debounce_seconds,
            callback=callback,
        )

    def on_any_event(self, event) -> None:
        """
        Handle any file system event.

        Compatible with watchdog FileSystemEventHandler.
        """
        # Skip directory events
        if getattr(event, "is_directory", False):
            return

        # Get source path
        src_path = getattr(event, "src_path", None)
        if not src_path:
            return

        # Check ignore patterns
        if self._should_ignore(src_path):
            return

        # Add to debouncer
        self._debouncer.add_change(src_path)

    def _should_ignore(self, path: str) -> bool:
        """Check if path should be ignored."""
        path_obj = Path(path)
        name = path_obj.name

        for pattern in self.ignore_patterns:
            # Match against name
            if fnmatch.fnmatch(name, pattern):
                return True

            # Match against full path
            if fnmatch.fnmatch(path, pattern):
                return True

            # Match against any path component
            for part in path_obj.parts:
                if fnmatch.fnmatch(part, pattern):
                    return True

        return False

    def stop(self) -> None:
        """Stop the handler and cancel pending changes."""
        self._debouncer.cancel()


class PriorityChangeQueue:
    """
    Priority queue for file changes.

    PATTERN: Process important files first
    """

    def __init__(self):
        """Initialize priority queue."""
        self._high: List[str] = []
        self._normal: List[str] = []
        self._low: List[str] = []
        self._lock = threading.Lock()

        # Patterns for priority classification
        self._high_priority_patterns = [
            "*.py",
            "*.ts",
            "*.tsx",
            "*.js",
            "*.jsx",
        ]
        self._low_priority_patterns = [
            "*.md",
            "*.txt",
            "*.json",
            "*.yaml",
            "*.yml",
        ]

    def add(self, file_path: str) -> None:
        """Add file to appropriate queue."""
        name = Path(file_path).name

        with self._lock:
            # Determine priority
            for pattern in self._high_priority_patterns:
                if fnmatch.fnmatch(name, pattern):
                    if file_path not in self._high:
                        self._high.append(file_path)
                    return

            for pattern in self._low_priority_patterns:
                if fnmatch.fnmatch(name, pattern):
                    if file_path not in self._low:
                        self._low.append(file_path)
                    return

            # Default to normal
            if file_path not in self._normal:
                self._normal.append(file_path)

    def get_batch(self, size: int = 10) -> List[str]:
        """Get batch of files in priority order."""
        with self._lock:
            batch = []

            # High priority first
            while self._high and len(batch) < size:
                batch.append(self._high.pop(0))

            # Then normal
            while self._normal and len(batch) < size:
                batch.append(self._normal.pop(0))

            # Then low
            while self._low and len(batch) < size:
                batch.append(self._low.pop(0))

            return batch

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return not (self._high or self._normal or self._low)

    def get_total_count(self) -> int:
        """Get total items in queue."""
        with self._lock:
            return len(self._high) + len(self._normal) + len(self._low)
