"""Understanding service orchestrator."""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from ..understanding import (
    CodebaseAnalyzer,
    CodebaseWatcher,
    KnowledgeStore,
    KnowledgeVerifier,
    DuplicateDetector,
    GapDetector,
    ConfidenceScorer,
    KnowledgeQueryHandler,
    SimilaritySearchHandler,
    GapQueryHandler,
    CodebaseUnderstanding,
    AnalysisProgress,
    KnowledgeQuery,
    KnowledgeResponse,
    VerificationResult,
    DuplicateCandidate,
)

logger = logging.getLogger(__name__)


class UnderstandingService:
    """
    Facade for all codebase understanding operations.

    PATTERN: Service orchestrator
    CRITICAL: Single entry point for understanding features
    """

    def __init__(
        self,
        root_path: str,
        memory_service: Optional[Any] = None,
        auto_watch: bool = False,
    ):
        """
        Initialize understanding service.

        Args:
            root_path: Project root directory
            memory_service: Optional memory service for embeddings
            auto_watch: Whether to start file watcher automatically
        """
        self.root_path = Path(root_path).resolve()
        self.persistence_path = self.root_path / ".agent-swarm" / "understanding"

        # Initialize knowledge store
        self.knowledge_store = KnowledgeStore(
            memory_service=memory_service,
            persistence_path=str(self.persistence_path),
        )

        # Initialize analyzer
        self.analyzer = CodebaseAnalyzer(
            root_path=str(self.root_path),
            knowledge_store=self.knowledge_store,
            memory_service=memory_service,
        )

        # Initialize other components
        self.verifier = KnowledgeVerifier(self.knowledge_store)
        self.duplicate_detector = DuplicateDetector(self.knowledge_store)
        self.gap_detector = GapDetector(self.knowledge_store)
        self.confidence_scorer = ConfidenceScorer()

        # Query handlers
        self.knowledge_query = KnowledgeQueryHandler(self.knowledge_store, self.verifier)
        self.similarity_search = SimilaritySearchHandler(self.knowledge_store)
        self.gap_query = GapQueryHandler(self.knowledge_store, self.gap_detector)

        # File watcher
        self._watcher: Optional[CodebaseWatcher] = None

        # Auto-start watcher if requested
        if auto_watch:
            self.start_watcher()

    @property
    def understanding(self) -> Optional[CodebaseUnderstanding]:
        """Get current understanding."""
        return self.analyzer.understanding

    @property
    def is_analyzed(self) -> bool:
        """Check if codebase has been analyzed."""
        return self.analyzer.understanding is not None

    @property
    def is_watching(self) -> bool:
        """Check if file watcher is active."""
        return self._watcher is not None and self._watcher.is_running

    # Analysis Methods

    async def analyze(
        self,
        mode: str = "deep",
        progress_callback: Optional[Callable[[AnalysisProgress], None]] = None,
    ) -> CodebaseUnderstanding:
        """
        Analyze codebase.

        Args:
            mode: "quick" or "deep"
            progress_callback: Optional progress callback

        Returns:
            CodebaseUnderstanding
        """
        logger.info(f"Starting {mode} analysis of {self.root_path}")
        return await self.analyzer.analyze(mode=mode, progress_callback=progress_callback)

    async def analyze_quick(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> CodebaseUnderstanding:
        """Quick analysis - structure only."""
        return await self.analyzer.analyze_quick(progress_callback=progress_callback)

    async def analyze_deep(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> CodebaseUnderstanding:
        """Deep analysis - full symbols."""
        return await self.analyzer.analyze_deep(progress_callback=progress_callback)

    async def refresh(self) -> CodebaseUnderstanding:
        """Re-analyze changed files since last analysis."""
        if not self.understanding:
            return await self.analyze()

        # Get pending changes from watcher
        if self._watcher:
            status = self._watcher.get_status()
            pending = status.get("pending_changes", 0)
            if pending > 0:
                logger.info(f"Refreshing {pending} changed files")

        return await self.analyzer.analyze()

    async def load_existing(self) -> Optional[CodebaseUnderstanding]:
        """Load existing understanding from disk."""
        return await self.analyzer.load_existing()

    # Verification Methods

    async def verify_claim(self, claim: str) -> VerificationResult:
        """Verify a claim about the codebase."""
        return await self.verifier.verify_claim(claim)

    async def verify_symbol_exists(
        self,
        symbol_name: str,
        expected_kind: Optional[str] = None,
    ) -> VerificationResult:
        """Verify a symbol exists."""
        return await self.verifier.verify_symbol_exists(symbol_name, expected_kind)

    async def verify_before_claim(self, claim: str) -> str:
        """Verify and wrap a claim for safety."""
        return await self.verifier.verify_before_claim(claim)

    # Duplicate Detection Methods

    async def check_for_duplicates(
        self,
        proposed_name: str,
        proposed_description: str,
        proposed_kind: str = "function",
        proposed_signature: Optional[str] = None,
    ) -> List[DuplicateCandidate]:
        """Check for potential duplicates before creating code."""
        return await self.duplicate_detector.check_before_create(
            proposed_name=proposed_name,
            proposed_description=proposed_description,
            proposed_kind=proposed_kind,
            proposed_signature=proposed_signature,
        )

    # Query Methods

    async def query(self, query_text: str) -> KnowledgeResponse:
        """Query knowledge about the codebase."""
        query = KnowledgeQuery(query=query_text)
        return await self.knowledge_query.query(query)

    async def find_similar(
        self,
        description: str,
        kind: str = "function",
        limit: int = 5,
    ) -> List[Dict]:
        """Find similar code."""
        if kind == "function":
            results = await self.similarity_search.find_similar_functions(description, limit)
        elif kind == "class":
            results = await self.similarity_search.find_similar_classes(description, limit)
        else:
            results = await self.knowledge_store.semantic_search(description, kind=kind, limit=limit)

        return [
            {
                "name": sym.qualified_name,
                "kind": sym.kind.value,
                "file": sym.file_path,
                "line": sym.line_start,
                "score": score,
                "docstring": sym.docstring[:200] if sym.docstring else None,
            }
            for sym, score in results
        ]

    async def get_symbol_info(self, symbol_name: str) -> Optional[str]:
        """Get verified information about a symbol."""
        return await self.verifier.get_symbol_info(symbol_name)

    # Gap Methods

    async def get_knowledge_gaps(self) -> List[Dict]:
        """Get current knowledge gaps."""
        gaps = await self.gap_query.get_all_gaps()
        return [
            {
                "id": g.id,
                "area": g.area,
                "description": g.description,
                "severity": g.severity,
                "actions": g.suggested_actions,
            }
            for g in gaps
        ]

    async def get_coverage_report(self) -> Dict:
        """Get knowledge coverage report."""
        return await self.gap_query.get_coverage_report()

    async def what_dont_i_know(self) -> str:
        """Get human-readable gap report."""
        return await self.gap_query.query_what_dont_i_know()

    # Watcher Methods

    def start_watcher(self) -> bool:
        """Start background file watcher."""
        if self._watcher and self._watcher.is_running:
            return True

        self._watcher = CodebaseWatcher(
            root_path=str(self.root_path),
            analyzer=self.analyzer,
            on_change_callback=self._on_files_changed,
        )

        return self._watcher.start()

    def stop_watcher(self) -> None:
        """Stop background file watcher."""
        if self._watcher:
            self._watcher.stop()

    def get_watcher_status(self) -> Dict:
        """Get watcher status."""
        if not self._watcher:
            return {"running": False}
        return self._watcher.get_status()

    def _on_files_changed(self, changed_files: List[str]) -> None:
        """Handle file change notifications."""
        logger.info(f"Files changed: {len(changed_files)}")
        # Mark understanding as needing refresh
        if self.understanding:
            self.understanding.pending_changes.extend(changed_files)

    # Statistics Methods

    def get_statistics(self) -> Dict:
        """Get analysis statistics."""
        stats = self.analyzer.get_statistics()

        # Add watcher status
        stats["watcher_active"] = self.is_watching

        # Add confidence summary
        if self.understanding:
            symbols = list(self.knowledge_store.get_all_symbols().values())
            confidence_summary = self.confidence_scorer.get_confidence_summary(symbols)
            stats["confidence"] = confidence_summary

        return stats

    # Lifecycle Methods

    async def initialize(self) -> bool:
        """
        Initialize service - load existing or analyze.

        Returns:
            True if understanding is available
        """
        # Try to load existing
        existing = await self.load_existing()
        if existing:
            logger.info("Loaded existing understanding")
            return True

        # Otherwise, do quick analysis
        logger.info("No existing understanding, performing quick analysis")
        await self.analyze_quick()
        return True

    async def cleanup(self) -> None:
        """Cleanup service resources."""
        self.stop_watcher()
        logger.info("Understanding service cleaned up")


# Convenience function for quick access
async def analyze_codebase(
    root_path: str,
    mode: str = "deep",
    memory_service: Optional[Any] = None,
) -> CodebaseUnderstanding:
    """
    Convenience function to analyze a codebase.

    Args:
        root_path: Project root directory
        mode: Analysis mode ("quick" or "deep")
        memory_service: Optional memory service

    Returns:
        CodebaseUnderstanding
    """
    service = UnderstandingService(root_path, memory_service=memory_service)
    return await service.analyze(mode=mode)
