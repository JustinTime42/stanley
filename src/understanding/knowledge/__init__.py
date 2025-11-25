"""Knowledge management for codebase understanding."""

from .store import KnowledgeStore
from .confidence import ConfidenceScorer
from .verification import KnowledgeVerifier
from .gaps import GapDetector
from .duplicates import DuplicateDetector

__all__ = [
    "KnowledgeStore",
    "ConfidenceScorer",
    "KnowledgeVerifier",
    "GapDetector",
    "DuplicateDetector",
]
