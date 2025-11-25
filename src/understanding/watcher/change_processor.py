"""Incremental change processor."""

import logging
from pathlib import Path
from typing import List, Dict, Optional, TYPE_CHECKING
from datetime import datetime

from ..models import ConfidenceLevel

if TYPE_CHECKING:
    from ..analyzer import CodebaseAnalyzer
    from ..knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)


class ChangeProcessor:
    """
    Process file changes incrementally.

    PATTERN: Efficient incremental update
    CRITICAL: Don't re-analyze unchanged files
    """

    def __init__(
        self,
        knowledge_store: Optional["KnowledgeStore"] = None,
    ):
        """
        Initialize processor.

        Args:
            knowledge_store: Knowledge store to update
        """
        self.store = knowledge_store
        self._pending_changes: Dict[str, datetime] = {}

    async def process_changes(
        self,
        changed_files: List[str],
        analyzer: Optional["CodebaseAnalyzer"] = None,
    ) -> Dict[str, str]:
        """
        Process file changes incrementally.

        Args:
            changed_files: List of changed file paths
            analyzer: Optional analyzer for re-analysis

        Returns:
            Dict mapping file path to result status
        """
        results = {}

        for file_path in changed_files:
            path = Path(file_path)

            if not path.exists():
                # File was deleted
                results[file_path] = await self._handle_deleted(file_path)
            else:
                # File was created or modified
                results[file_path] = await self._handle_modified(file_path, analyzer)

        return results

    async def _handle_deleted(self, file_path: str) -> str:
        """Handle deleted file."""
        if not self.store:
            return "no_store"

        # Remove symbols from this file
        all_symbols = self.store.get_all_symbols()
        deleted_symbols = [
            sid for sid, sym in all_symbols.items()
            if sym.file_path == file_path
        ]

        for sid in deleted_symbols:
            # Mark as deleted (remove from store)
            if sid in self.store._symbols:
                del self.store._symbols[sid]
                logger.debug(f"Removed symbol {sid} from deleted file")

        # Remove file info
        if file_path in self.store._files:
            del self.store._files[file_path]

        logger.info(f"Processed deletion: {file_path} ({len(deleted_symbols)} symbols removed)")
        return "deleted"

    async def _handle_modified(
        self,
        file_path: str,
        analyzer: Optional["CodebaseAnalyzer"],
    ) -> str:
        """Handle modified or created file."""
        if not self.store:
            return "no_store"

        # Mark existing symbols as stale
        all_symbols = self.store.get_all_symbols()
        stale_symbols = [
            sid for sid, sym in all_symbols.items()
            if sym.file_path == file_path
        ]

        for sid in stale_symbols:
            if sid in self.store._symbols:
                self.store._symbols[sid].confidence = ConfidenceLevel.STALE

        # Re-analyze if analyzer available
        if analyzer:
            try:
                from ..scanner import FileScanner

                # Create file info
                scanner = FileScanner(Path(file_path).parent)
                file_infos = list(scanner.scan())

                # Find our file
                file_info = None
                for fi in file_infos:
                    if fi.path == file_path:
                        file_info = fi
                        break

                if file_info:
                    # Re-extract symbols
                    from ..extractors import SymbolExtractor

                    extractor = SymbolExtractor()
                    new_symbols = await extractor.extract_file(file_info)

                    # Update store
                    await self.store.update_symbols(new_symbols, [file_path])

                    logger.info(
                        f"Re-analyzed {file_path}: {len(new_symbols)} symbols"
                    )
                    return "reanalyzed"

            except Exception as e:
                logger.error(f"Failed to re-analyze {file_path}: {e}")
                return "error"

        return "marked_stale"

    async def batch_process(
        self,
        changes: Dict[str, str],
        analyzer: Optional["CodebaseAnalyzer"] = None,
    ) -> Dict[str, str]:
        """
        Process batch of changes.

        Args:
            changes: Dict of {file_path: change_type} where type is
                    'created', 'modified', 'deleted'
            analyzer: Optional analyzer

        Returns:
            Results dict
        """
        results = {}

        # Group by type for efficient processing
        deletions = [p for p, t in changes.items() if t == "deleted"]
        modifications = [p for p, t in changes.items() if t in ("created", "modified")]

        # Process deletions first
        for file_path in deletions:
            results[file_path] = await self._handle_deleted(file_path)

        # Then modifications (may depend on deletions)
        for file_path in modifications:
            results[file_path] = await self._handle_modified(file_path, analyzer)

        return results

    def queue_change(self, file_path: str) -> None:
        """Queue a change for later processing."""
        self._pending_changes[file_path] = datetime.now()

    def get_pending_changes(self) -> List[str]:
        """Get list of pending changes."""
        return list(self._pending_changes.keys())

    def clear_pending(self) -> None:
        """Clear pending changes."""
        self._pending_changes.clear()

    async def process_pending(
        self,
        analyzer: Optional["CodebaseAnalyzer"] = None,
    ) -> Dict[str, str]:
        """Process all pending changes."""
        if not self._pending_changes:
            return {}

        files = list(self._pending_changes.keys())
        self._pending_changes.clear()

        return await self.process_changes(files, analyzer)


class IncrementalUpdater:
    """
    High-level incremental update coordinator.

    PATTERN: Coordinate between watcher and analyzer
    """

    def __init__(
        self,
        analyzer: "CodebaseAnalyzer",
        processor: Optional[ChangeProcessor] = None,
    ):
        """
        Initialize updater.

        Args:
            analyzer: Codebase analyzer
            processor: Optional change processor
        """
        self.analyzer = analyzer
        self.processor = processor or ChangeProcessor(
            knowledge_store=analyzer.knowledge_store if hasattr(analyzer, "knowledge_store") else None
        )

    async def on_files_changed(self, changed_files: List[str]) -> None:
        """
        Handle file change notification.

        Args:
            changed_files: List of changed file paths
        """
        logger.info(f"Processing {len(changed_files)} file changes")

        # Process changes
        results = await self.processor.process_changes(
            changed_files,
            self.analyzer,
        )

        # Log results
        stats = {
            "deleted": 0,
            "reanalyzed": 0,
            "marked_stale": 0,
            "error": 0,
        }

        for result in results.values():
            if result in stats:
                stats[result] += 1

        logger.info(f"Change processing complete: {stats}")

    async def refresh_stale(self) -> int:
        """
        Re-analyze all stale symbols.

        Returns:
            Number of symbols refreshed
        """
        if not self.analyzer.knowledge_store:
            return 0

        # Find stale files
        stale_files = set()
        for symbol in self.analyzer.knowledge_store.get_all_symbols().values():
            if symbol.confidence == ConfidenceLevel.STALE:
                stale_files.add(symbol.file_path)

        if not stale_files:
            return 0

        # Re-analyze
        results = await self.processor.process_changes(
            list(stale_files),
            self.analyzer,
        )

        return sum(1 for r in results.values() if r == "reanalyzed")
