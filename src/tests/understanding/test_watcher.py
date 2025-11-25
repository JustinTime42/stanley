"""Tests for file watcher."""

import pytest
import tempfile
from pathlib import Path
import time


class TestDebouncer:
    """Tests for ChangeDebouncer."""

    def test_debouncer_creation(self):
        """Test debouncer can be created."""
        from src.understanding.watcher import ChangeDebouncer

        debouncer = ChangeDebouncer(debounce_seconds=0.5)
        assert debouncer.debounce_seconds == 0.5

    def test_add_change(self):
        """Test adding changes."""
        from src.understanding.watcher import ChangeDebouncer

        received_changes = []

        def callback(files):
            received_changes.extend(files)

        debouncer = ChangeDebouncer(debounce_seconds=0.1, callback=callback)

        # Add changes
        debouncer.add_change("/path/to/file1.py")
        debouncer.add_change("/path/to/file2.py")

        # Wait for debounce
        time.sleep(0.2)

        assert len(received_changes) == 2

    def test_deduplication(self):
        """Test that same file is deduplicated."""
        from src.understanding.watcher import ChangeDebouncer

        received_changes = []

        def callback(files):
            received_changes.extend(files)

        debouncer = ChangeDebouncer(debounce_seconds=0.1, callback=callback)

        # Add same file multiple times
        debouncer.add_change("/path/to/file.py")
        debouncer.add_change("/path/to/file.py")
        debouncer.add_change("/path/to/file.py")

        time.sleep(0.2)

        # Should only have one entry
        assert received_changes.count("/path/to/file.py") == 1

    def test_cancel(self):
        """Test canceling pending changes."""
        from src.understanding.watcher import ChangeDebouncer

        received_changes = []

        def callback(files):
            received_changes.extend(files)

        debouncer = ChangeDebouncer(debounce_seconds=0.5, callback=callback)

        debouncer.add_change("/path/to/file.py")
        debouncer.cancel()

        time.sleep(0.6)

        # Should not have received anything
        assert len(received_changes) == 0


class TestDebouncedHandler:
    """Tests for DebouncedHandler."""

    def test_ignore_patterns(self):
        """Test ignore patterns work."""
        from src.understanding.watcher import DebouncedHandler

        handler = DebouncedHandler(
            callback=lambda x: None,
            debounce_seconds=0.1,
            ignore_patterns=["*.pyc", "__pycache__", ".git"],
        )

        # These should be ignored
        assert handler._should_ignore("/path/__pycache__/file.py") is True
        assert handler._should_ignore("/path/file.pyc") is True
        assert handler._should_ignore("/path/.git/config") is True

        # These should not be ignored
        assert handler._should_ignore("/path/file.py") is False
        assert handler._should_ignore("/path/main.js") is False


class TestCodebaseWatcher:
    """Tests for CodebaseWatcher."""

    @pytest.fixture
    def temp_dir(self):
        """Create temp directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("# Test file")
            yield tmpdir

    def test_watcher_creation(self, temp_dir):
        """Test watcher can be created."""
        from src.understanding.watcher import CodebaseWatcher

        watcher = CodebaseWatcher(temp_dir)
        assert watcher.root_path == Path(temp_dir).resolve()
        assert not watcher.is_running

    def test_watcher_status(self, temp_dir):
        """Test watcher status."""
        from src.understanding.watcher import CodebaseWatcher

        watcher = CodebaseWatcher(temp_dir)
        status = watcher.get_status()

        assert status["running"] is False
        assert status["root_path"] == str(Path(temp_dir).resolve())
