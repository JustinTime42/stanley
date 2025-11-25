"""Background file system watcher."""

import logging
import asyncio
import threading
from pathlib import Path
from typing import Optional, List, Callable, TYPE_CHECKING

from .debouncer import DebouncedHandler

if TYPE_CHECKING:
    from ..analyzer import CodebaseAnalyzer

logger = logging.getLogger(__name__)


class CodebaseWatcher:
    """
    Background file system watcher for keeping understanding current.

    PATTERN: Run in background, debounce rapid changes
    CRITICAL: Must not block main thread
    """

    DEFAULT_IGNORE_PATTERNS = [
        "*.pyc",
        "__pycache__",
        ".git",
        "node_modules",
        ".agent-swarm",
        "*.log",
        "*.tmp",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "venv",
        ".venv",
        "*.egg-info",
        "dist",
        "build",
    ]

    def __init__(
        self,
        root_path: str,
        analyzer: Optional["CodebaseAnalyzer"] = None,
        debounce_seconds: float = 1.0,
        ignore_patterns: Optional[List[str]] = None,
        on_change_callback: Optional[Callable[[List[str]], None]] = None,
    ):
        """
        Initialize watcher.

        Args:
            root_path: Directory to watch
            analyzer: Optional CodebaseAnalyzer for updates
            debounce_seconds: Debounce window
            ignore_patterns: Additional patterns to ignore
            on_change_callback: Optional callback for changes
        """
        self.root_path = Path(root_path).resolve()
        self.analyzer = analyzer
        self.debounce_seconds = debounce_seconds

        self.ignore_patterns = self.DEFAULT_IGNORE_PATTERNS.copy()
        if ignore_patterns:
            self.ignore_patterns.extend(ignore_patterns)

        self.on_change_callback = on_change_callback

        self._observer = None
        self._handler: Optional[DebouncedHandler] = None
        self._running = False
        self._change_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    def start(self) -> bool:
        """
        Start watching for file changes.

        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning("Watcher already running")
            return True

        try:
            # Import watchdog here to make it optional
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            # Create handler that forwards to our debounced handler
            self._handler = DebouncedHandler(
                callback=self._on_changes,
                debounce_seconds=self.debounce_seconds,
                ignore_patterns=self.ignore_patterns,
            )

            # Create wrapper for watchdog
            class WatchdogAdapter(FileSystemEventHandler):
                def __init__(self, handler: DebouncedHandler):
                    self.handler = handler

                def on_any_event(self, event):
                    self.handler.on_any_event(event)

            adapter = WatchdogAdapter(self._handler)

            # Create and start observer
            self._observer = Observer()
            self._observer.schedule(
                adapter,
                str(self.root_path),
                recursive=True,
            )
            self._observer.start()
            self._running = True

            # Start async processor if we have an event loop
            try:
                self._loop = asyncio.get_running_loop()
                self._processor_task = asyncio.create_task(self._process_changes())
            except RuntimeError:
                # No running event loop, changes will be processed synchronously
                self._loop = None

            logger.info(f"Started watching: {self.root_path}")
            return True

        except ImportError:
            logger.error("watchdog library not installed. Run: pip install watchdog")
            return False
        except Exception as e:
            logger.error(f"Failed to start watcher: {e}")
            return False

    def stop(self) -> None:
        """Stop watching."""
        if not self._running:
            return

        self._running = False

        # Stop handler
        if self._handler:
            self._handler.stop()
            self._handler = None

        # Stop observer
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

        # Cancel processor
        if self._processor_task:
            self._processor_task.cancel()
            self._processor_task = None

        logger.info("Stopped watching")

    def _on_changes(self, changed_files: List[str]) -> None:
        """Handle debounced changes."""
        logger.debug(f"Detected changes: {len(changed_files)} files")

        # Call user callback if provided
        if self.on_change_callback:
            try:
                self.on_change_callback(changed_files)
            except Exception as e:
                logger.error(f"Error in change callback: {e}")

        # Queue for async processing
        if self._loop and self._running:
            for file_path in changed_files:
                try:
                    self._loop.call_soon_threadsafe(
                        self._change_queue.put_nowait,
                        file_path,
                    )
                except Exception as e:
                    logger.warning(f"Failed to queue change: {e}")

    async def _process_changes(self) -> None:
        """Process file changes asynchronously."""
        batch = []
        batch_deadline = None

        while self._running:
            try:
                # Wait for changes with timeout
                try:
                    file_path = await asyncio.wait_for(
                        self._change_queue.get(),
                        timeout=0.5,
                    )
                    batch.append(file_path)

                    if batch_deadline is None:
                        batch_deadline = asyncio.get_event_loop().time() + 1.0

                except asyncio.TimeoutError:
                    pass

                # Process batch if deadline passed
                now = asyncio.get_event_loop().time()
                if batch and batch_deadline and now >= batch_deadline:
                    unique_files = list(set(batch))
                    logger.info(f"Processing {len(unique_files)} changed files")

                    # Update analyzer if available
                    if self.analyzer:
                        try:
                            await self.analyzer.update_from_changes(unique_files)
                        except Exception as e:
                            logger.error(f"Error updating from changes: {e}")

                    batch = []
                    batch_deadline = None

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in change processor: {e}")

    def get_status(self) -> dict:
        """Get watcher status."""
        return {
            "running": self._running,
            "root_path": str(self.root_path),
            "ignore_patterns": len(self.ignore_patterns),
            "debounce_seconds": self.debounce_seconds,
            "pending_changes": self._change_queue.qsize() if self._loop else 0,
        }


class WatcherManager:
    """
    Manage multiple codebase watchers.

    PATTERN: Singleton for watcher management
    """

    _instance: Optional["WatcherManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "WatcherManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._watchers = {}
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._watchers: dict = {}
            self._initialized = True

    def create_watcher(
        self,
        watcher_id: str,
        root_path: str,
        **kwargs,
    ) -> CodebaseWatcher:
        """Create and register a watcher."""
        if watcher_id in self._watchers:
            return self._watchers[watcher_id]

        watcher = CodebaseWatcher(root_path, **kwargs)
        self._watchers[watcher_id] = watcher
        return watcher

    def get_watcher(self, watcher_id: str) -> Optional[CodebaseWatcher]:
        """Get watcher by ID."""
        return self._watchers.get(watcher_id)

    def stop_all(self) -> None:
        """Stop all watchers."""
        for watcher in self._watchers.values():
            watcher.stop()
        self._watchers.clear()

    def get_all_status(self) -> dict:
        """Get status of all watchers."""
        return {
            wid: w.get_status()
            for wid, w in self._watchers.items()
        }
