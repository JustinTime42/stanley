"""Main codebase analyzer orchestrator."""

import gc
import logging
import hashlib
from pathlib import Path
from typing import Optional, List, Callable, Any
from datetime import datetime

from .scanner import FileScanner
from .extractors import (
    StructureExtractor,
    SymbolExtractor,
    DependencyExtractor,
    ConventionExtractor,
    DocumentationExtractor,
)
from .knowledge import KnowledgeStore
from .models import (
    CodebaseUnderstanding,
    ProjectStructure,
    DependencyGraph,
    AnalysisProgress,
    AnalysisMode,
    FileInfo,
    KnowledgeGap,
)

logger = logging.getLogger(__name__)


class CodebaseAnalyzer:
    """
    Main codebase analysis engine.

    PATTERN: Orchestrator for all analysis operations
    CRITICAL: Support both quick and deep analysis modes
    """

    def __init__(
        self,
        root_path: str,
        knowledge_store: Optional[KnowledgeStore] = None,
        memory_service: Optional[Any] = None,
        persistence_path: Optional[str] = None,
    ):
        """
        Initialize analyzer.

        Args:
            root_path: Project root directory
            knowledge_store: Optional existing knowledge store
            memory_service: Optional memory service for embeddings
            persistence_path: Path for persisting understanding
        """
        self.root_path = Path(root_path).resolve()
        self.project_id = self._generate_project_id()

        # Set up persistence path
        if persistence_path:
            self.persistence_path = Path(persistence_path)
        else:
            self.persistence_path = self.root_path / ".agent-swarm" / "understanding"

        # Initialize knowledge store
        self.knowledge_store = knowledge_store or KnowledgeStore(
            memory_service=memory_service,
            persistence_path=str(self.persistence_path),
        )

        # Initialize extractors
        self.scanner = FileScanner(self.root_path)
        self.structure_extractor = StructureExtractor()
        self.symbol_extractor = SymbolExtractor()
        self.dependency_extractor = DependencyExtractor()
        self.convention_extractor = ConventionExtractor()
        self.doc_extractor = DocumentationExtractor()

        self._understanding: Optional[CodebaseUnderstanding] = None

    def _generate_project_id(self) -> str:
        """Generate unique project identifier."""
        content = str(self.root_path)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def understanding(self) -> Optional[CodebaseUnderstanding]:
        """Get current understanding."""
        return self._understanding

    async def analyze(
        self,
        mode: str = "deep",
        focus_paths: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[AnalysisProgress], None]] = None,
    ) -> CodebaseUnderstanding:
        """
        Analyze codebase and build understanding.

        Args:
            mode: Analysis depth - "quick", "deep", or "exhaustive"
            focus_paths: Optional paths to prioritize
            exclude_patterns: Additional patterns to exclude
            progress_callback: Progress update callback

        Returns:
            Complete CodebaseUnderstanding
        """
        start_time = datetime.now()
        analysis_mode = AnalysisMode(mode)

        # Step 1: Scan files
        self._report_progress(progress_callback, "Scanning files...", 0, "scanning")
        files = list(self.scanner.scan(exclude_patterns))
        logger.info(f"Scanned {len(files)} files")

        if focus_paths:
            files = self._prioritize_paths(files, focus_paths)

        # Step 2: Extract structure (always)
        self._report_progress(progress_callback, "Analyzing structure...", 10, "structure")
        structure = await self.structure_extractor.extract(self.root_path, files)

        if mode == "quick":
            # Quick mode: structure only
            self._understanding = CodebaseUnderstanding(
                project_id=self.project_id,
                root_path=str(self.root_path),
                structure=structure,
                dependency_graph=DependencyGraph(),
                analysis_mode=AnalysisMode.QUICK,
            )
            self._understanding.unanalyzed_files = [f.path for f in files]

            elapsed = (datetime.now() - start_time).total_seconds()
            self._understanding.total_analysis_time_seconds = elapsed

            self._report_progress(progress_callback, "Quick scan complete!", 100, "done")
            return self._understanding

        # Step 3: Extract symbols (batched for memory efficiency)
        self._report_progress(progress_callback, "Extracting symbols...", 20, "symbols")
        symbols = {}
        file_infos = {}

        batch_size = 50
        total_batches = (len(files) + batch_size - 1) // batch_size

        for i, batch in enumerate(self._batch(files, batch_size)):
            batch_symbols, batch_files = await self.symbol_extractor.extract_batch(batch)
            symbols.update(batch_symbols)
            file_infos.update(batch_files)

            progress = 20 + (50 * (i + 1) / total_batches)
            self._report_progress(
                progress_callback,
                f"Extracting symbols... ({len(symbols)} found)",
                progress,
                "symbols",
            )

            # Help with memory pressure
            if i % 5 == 0:
                gc.collect()

        logger.info(f"Extracted {len(symbols)} symbols from {len(file_infos)} files")

        # Step 4: Build dependency graph
        self._report_progress(progress_callback, "Mapping dependencies...", 75, "dependencies")
        dependency_graph = await self.dependency_extractor.extract(
            files, symbols, file_infos
        )

        # Step 5: Detect conventions
        self._report_progress(progress_callback, "Detecting conventions...", 85, "conventions")
        conventions = await self.convention_extractor.extract(symbols, files)
        structure.naming_convention = conventions.get("naming", "unknown")
        structure.test_framework = conventions.get("test_patterns", {}).get("primary_pattern")

        # Step 6: Extract documentation
        self._report_progress(progress_callback, "Processing documentation...", 90, "documentation")
        docs = await self.doc_extractor.extract(files, symbols)

        # Step 7: Build understanding
        self._report_progress(progress_callback, "Finalizing...", 95, "finalizing")

        elapsed = (datetime.now() - start_time).total_seconds()

        self._understanding = CodebaseUnderstanding(
            project_id=self.project_id,
            root_path=str(self.root_path),
            structure=structure,
            dependency_graph=dependency_graph,
            symbols=symbols,
            files=file_infos,
            analysis_mode=analysis_mode,
            total_analysis_time_seconds=elapsed,
        )

        # Step 8: Store in knowledge store
        await self.knowledge_store.store_understanding(self._understanding)

        # Step 9: Identify initial knowledge gaps
        from .knowledge import GapDetector
        gap_detector = GapDetector(self.knowledge_store)
        gaps = await gap_detector.detect_gaps(self._understanding)
        self._understanding.knowledge_gaps = gaps

        self._report_progress(progress_callback, "Analysis complete!", 100, "done")

        logger.info(
            f"Analysis complete: {len(symbols)} symbols, "
            f"{len(file_infos)} files in {elapsed:.1f}s"
        )

        return self._understanding

    async def analyze_quick(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> CodebaseUnderstanding:
        """Quick analysis - structure only, <30 seconds."""
        return await self.analyze(mode="quick", progress_callback=progress_callback)

    async def analyze_deep(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> CodebaseUnderstanding:
        """Deep analysis - full symbol extraction."""
        return await self.analyze(mode="deep", progress_callback=progress_callback)

    async def update_from_changes(
        self,
        changed_files: List[str],
    ) -> CodebaseUnderstanding:
        """
        Incrementally update understanding from file changes.

        PATTERN: Efficient incremental update
        CRITICAL: Don't re-analyze unchanged files
        """
        if not self._understanding:
            # No existing understanding, do full analysis
            return await self.analyze()

        logger.info(f"Updating from {len(changed_files)} changed files")

        for file_path in changed_files:
            # Remove stale symbols from this file
            old_symbols = [
                sid for sid, sym in self._understanding.symbols.items()
                if sym.file_path == file_path
            ]
            for sid in old_symbols:
                del self._understanding.symbols[sid]

            # Re-extract symbols for changed file
            if Path(file_path).exists():
                new_symbols = await self.symbol_extractor.extract_file(file_path)
                self._understanding.symbols.update(new_symbols)

                # Update file info
                for fi in self.scanner.scan():
                    if fi.path == file_path:
                        fi.symbols = list(new_symbols.keys())
                        self._understanding.files[file_path] = fi
                        break
            else:
                # File deleted
                if file_path in self._understanding.files:
                    del self._understanding.files[file_path]

        # Update dependency graph
        files = list(self._understanding.files.values())
        self._understanding.dependency_graph = await self.dependency_extractor.extract(
            files,
            self._understanding.symbols,
            self._understanding.files,
        )

        # Update knowledge store
        await self.knowledge_store.update_symbols(
            self._understanding.symbols,
            changed_files,
        )

        self._understanding.updated_at = datetime.now()

        return self._understanding

    async def load_existing(self) -> Optional[CodebaseUnderstanding]:
        """Load existing understanding from disk."""
        understanding = await self.knowledge_store.load_from_disk()
        if understanding:
            self._understanding = understanding
            logger.info("Loaded existing understanding from disk")
        return understanding

    def _prioritize_paths(
        self,
        files: List[FileInfo],
        focus_paths: List[str],
    ) -> List[FileInfo]:
        """Prioritize certain paths in file list."""
        focused = []
        other = []

        for f in files:
            is_focused = any(
                fp in f.path or fp in f.relative_path
                for fp in focus_paths
            )
            if is_focused:
                focused.append(f)
            else:
                other.append(f)

        return focused + other

    def _batch(self, items: list, size: int):
        """Yield batches of items."""
        for i in range(0, len(items), size):
            yield items[i:i + size]

    def _report_progress(
        self,
        callback: Optional[Callable],
        message: str,
        progress: float,
        stage: str = "",
    ) -> None:
        """Report progress to callback."""
        if callback:
            progress_obj = AnalysisProgress(
                stage=stage,
                progress=progress,
                message=message,
            )
            try:
                callback(progress_obj)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def get_statistics(self) -> dict:
        """Get analysis statistics."""
        if not self._understanding:
            return {"analyzed": False}

        return {
            "analyzed": True,
            "project_name": self._understanding.structure.project_name,
            "detected_type": self._understanding.structure.detected_type,
            "framework": self._understanding.structure.detected_framework,
            "total_files": len(self._understanding.files),
            "total_symbols": len(self._understanding.symbols),
            "total_lines": self._understanding.structure.total_lines,
            "analysis_time_seconds": self._understanding.total_analysis_time_seconds,
            "analysis_mode": self._understanding.analysis_mode.value,
            "knowledge_gaps": len(self._understanding.knowledge_gaps),
        }
