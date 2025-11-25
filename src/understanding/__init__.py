"""Codebase understanding system.

This module provides comprehensive codebase analysis capabilities:
- File scanning and filtering
- Symbol extraction (functions, classes, methods)
- Dependency graph building
- Convention detection
- Knowledge verification (anti-hallucination)
- Duplicate detection
- Background file watching
- Semantic queries
"""

from .analyzer import CodebaseAnalyzer
from .scanner import FileScanner
from .models import (
    ConfidenceLevel,
    SymbolKind,
    Symbol,
    FileInfo,
    ProjectStructure,
    DependencyGraph,
    KnowledgeGap,
    DuplicateCandidate,
    CodebaseUnderstanding,
    AnalysisProgress,
    AnalysisMode,
    KnowledgeQuery,
    KnowledgeResponse,
    VerificationResult,
)
from .knowledge import (
    KnowledgeStore,
    ConfidenceScorer,
    KnowledgeVerifier,
    GapDetector,
    DuplicateDetector,
)
from .watcher import (
    CodebaseWatcher,
    ChangeProcessor,
)
from .queries import (
    KnowledgeQueryHandler,
    SimilaritySearchHandler,
    GapQueryHandler,
)

__all__ = [
    # Main classes
    "CodebaseAnalyzer",
    "FileScanner",
    # Models
    "ConfidenceLevel",
    "SymbolKind",
    "Symbol",
    "FileInfo",
    "ProjectStructure",
    "DependencyGraph",
    "KnowledgeGap",
    "DuplicateCandidate",
    "CodebaseUnderstanding",
    "AnalysisProgress",
    "AnalysisMode",
    "KnowledgeQuery",
    "KnowledgeResponse",
    "VerificationResult",
    # Knowledge
    "KnowledgeStore",
    "ConfidenceScorer",
    "KnowledgeVerifier",
    "GapDetector",
    "DuplicateDetector",
    # Watcher
    "CodebaseWatcher",
    "ChangeProcessor",
    # Queries
    "KnowledgeQueryHandler",
    "SimilaritySearchHandler",
    "GapQueryHandler",
]
