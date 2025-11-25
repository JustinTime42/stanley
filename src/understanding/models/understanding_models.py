"""Data models for codebase understanding system."""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import hashlib


class ConfidenceLevel(str, Enum):
    """Knowledge confidence levels."""

    VERIFIED = "verified"  # Directly analyzed, high certainty
    INFERRED = "inferred"  # Pattern-based, medium certainty
    UNCERTAIN = "uncertain"  # Contextual guess, low certainty
    UNKNOWN = "unknown"  # No information available
    STALE = "stale"  # Was verified but file changed

    def __lt__(self, other: "ConfidenceLevel") -> bool:
        """Enable comparison for confidence levels."""
        order = {
            ConfidenceLevel.VERIFIED: 4,
            ConfidenceLevel.INFERRED: 3,
            ConfidenceLevel.UNCERTAIN: 2,
            ConfidenceLevel.STALE: 1,
            ConfidenceLevel.UNKNOWN: 0,
        }
        return order.get(self, 0) < order.get(other, 0)


class SymbolKind(str, Enum):
    """Types of code symbols."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    TYPE = "type"
    PROPERTY = "property"
    DECORATOR = "decorator"


class AnalysisMode(str, Enum):
    """Analysis depth modes."""

    QUICK = "quick"  # Structure only, <30 seconds
    DEEP = "deep"  # Full analysis, <5 minutes
    EXHAUSTIVE = "exhaustive"  # Maximum detail


class Symbol(BaseModel):
    """A code symbol (function, class, variable, etc.)."""

    id: str = Field(description="Unique symbol identifier")
    name: str = Field(description="Symbol name")
    qualified_name: str = Field(description="Fully qualified name (module.class.method)")
    kind: SymbolKind = Field(description="Symbol type")

    # Location
    file_path: str = Field(description="File containing symbol")
    line_start: int = Field(description="Starting line number")
    line_end: int = Field(description="Ending line number")
    column_start: int = Field(default=0, description="Starting column")
    column_end: int = Field(default=0, description="Ending column")

    # Signature and documentation
    signature: Optional[str] = Field(default=None, description="Function/method signature")
    docstring: Optional[str] = Field(default=None, description="Documentation string")
    return_type: Optional[str] = Field(default=None, description="Return type annotation")
    parameters: List[Dict[str, Any]] = Field(
        default_factory=list, description="Parameters with types"
    )

    # Relationships
    parent_symbol: Optional[str] = Field(default=None, description="Parent class/module ID")
    calls: List[str] = Field(default_factory=list, description="Symbol IDs this calls")
    called_by: List[str] = Field(default_factory=list, description="Symbol IDs that call this")
    imports: List[str] = Field(default_factory=list, description="Imported symbols")

    # Verification
    content_hash: str = Field(description="Hash of symbol content for change detection")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.VERIFIED)
    last_verified: datetime = Field(default_factory=datetime.now)

    # Semantic
    description: Optional[str] = Field(default=None, description="AI-generated description")
    tags: List[str] = Field(default_factory=list, description="Semantic tags")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")

    @classmethod
    def generate_id(cls, file_path: str, name: str, line_start: int) -> str:
        """Generate unique symbol ID."""
        content = f"{file_path}:{name}:{line_start}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @classmethod
    def compute_content_hash(cls, content: str) -> str:
        """Compute hash of symbol content."""
        return hashlib.sha256(content.encode()).hexdigest()[:32]


class FileInfo(BaseModel):
    """Information about a source file."""

    path: str = Field(description="Absolute file path")
    relative_path: str = Field(description="Path relative to project root")
    language: str = Field(description="Programming language")
    size_bytes: int = Field(description="File size in bytes")
    line_count: int = Field(description="Number of lines")
    last_modified: datetime = Field(description="Last modification time")
    content_hash: str = Field(description="Hash of file content")

    # Analysis results
    symbols: List[str] = Field(default_factory=list, description="Symbol IDs in file")
    imports: List[str] = Field(default_factory=list, description="Import statements")
    exports: List[str] = Field(default_factory=list, description="Exported symbols")

    # Status
    analyzed: bool = Field(default=False, description="Whether file has been analyzed")
    analysis_errors: List[str] = Field(default_factory=list, description="Errors during analysis")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.UNKNOWN)


class ProjectStructure(BaseModel):
    """Project structure understanding."""

    root_path: str = Field(description="Project root directory")
    project_name: str = Field(description="Project name")
    detected_type: str = Field(description="python, javascript, typescript, etc.")
    detected_framework: Optional[str] = Field(
        default=None, description="FastAPI, React, Django, etc."
    )

    # Structure
    source_directories: List[str] = Field(default_factory=list, description="Source code dirs")
    test_directories: List[str] = Field(default_factory=list, description="Test dirs")
    config_files: List[str] = Field(default_factory=list, description="Config files found")
    entry_points: List[str] = Field(default_factory=list, description="Main entry points")

    # Statistics
    total_files: int = Field(default=0, description="Total source files")
    total_lines: int = Field(default=0, description="Total lines of code")
    files_by_language: Dict[str, int] = Field(
        default_factory=dict, description="File count per language"
    )

    # Conventions
    naming_convention: str = Field(default="unknown", description="snake_case, camelCase, etc.")
    test_framework: Optional[str] = Field(default=None, description="pytest, jest, etc.")
    package_manager: Optional[str] = Field(default=None, description="pip, npm, etc.")


class DependencyGraph(BaseModel):
    """Module dependency graph."""

    nodes: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Module nodes with metadata"
    )
    edges: List[Dict[str, str]] = Field(
        default_factory=list, description="Import edges (from, to, type)"
    )

    # Analysis
    entry_points: List[str] = Field(default_factory=list, description="Entry point modules")
    leaf_modules: List[str] = Field(default_factory=list, description="No outgoing deps")
    cycles: List[List[str]] = Field(default_factory=list, description="Detected cycles")

    def get_dependents(self, module: str) -> List[str]:
        """Get modules that depend on this module."""
        return [e["from"] for e in self.edges if e.get("to") == module]

    def get_dependencies(self, module: str) -> List[str]:
        """Get modules this module depends on."""
        return [e["to"] for e in self.edges if e.get("from") == module]

    def add_node(self, module: str, **metadata) -> None:
        """Add a module node."""
        self.nodes[module] = metadata

    def add_edge(self, from_module: str, to_module: str, edge_type: str = "import") -> None:
        """Add a dependency edge."""
        self.edges.append({"from": from_module, "to": to_module, "type": edge_type})


class KnowledgeGap(BaseModel):
    """A detected gap in understanding."""

    id: str = Field(description="Unique gap identifier")
    area: str = Field(description="Code area with gap")
    description: str = Field(description="What's unknown")
    severity: str = Field(description="high, medium, low")

    # Context
    related_files: List[str] = Field(default_factory=list, description="Related file paths")
    related_symbols: List[str] = Field(default_factory=list, description="Related symbol IDs")
    triggered_by: Optional[str] = Field(default=None, description="Query that exposed gap")

    # Resolution
    suggested_actions: List[str] = Field(default_factory=list, description="How to fill gap")
    resolved: bool = Field(default=False, description="Whether gap has been filled")
    resolved_at: Optional[datetime] = Field(default=None, description="When resolved")

    @classmethod
    def generate_id(cls, area: str, description: str) -> str:
        """Generate unique gap ID."""
        content = f"{area}:{description}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class DuplicateCandidate(BaseModel):
    """Potential duplicate functionality detection."""

    new_symbol: str = Field(description="Proposed new symbol name")
    existing_symbol: str = Field(description="Existing similar symbol qualified name")
    similarity_score: float = Field(ge=0, le=1, description="0.0-1.0 similarity")
    similarity_type: str = Field(description="name, signature, semantic, implementation")

    # Details
    new_description: str = Field(description="Description of new symbol")
    existing_description: str = Field(description="Description of existing symbol")
    existing_location: str = Field(default="", description="File:line of existing symbol")
    recommendation: str = Field(description="use_existing, extend, create_new")


class AnalysisProgress(BaseModel):
    """Progress update during analysis."""

    stage: str = Field(description="Current analysis stage")
    progress: float = Field(ge=0, le=100, description="Percentage complete")
    message: str = Field(description="Human-readable status")
    files_processed: int = Field(default=0)
    files_total: int = Field(default=0)
    symbols_found: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)


class CodebaseUnderstanding(BaseModel):
    """Complete codebase understanding state."""

    project_id: str = Field(description="Unique project identifier")
    root_path: str = Field(description="Project root directory")

    # Core understanding
    structure: ProjectStructure = Field(description="Project structure")
    dependency_graph: DependencyGraph = Field(
        default_factory=DependencyGraph, description="Module dependencies"
    )
    symbols: Dict[str, Symbol] = Field(default_factory=dict, description="Symbol table")
    files: Dict[str, FileInfo] = Field(default_factory=dict, description="File info")

    # Knowledge state
    knowledge_gaps: List[KnowledgeGap] = Field(
        default_factory=list, description="Detected gaps"
    )
    unanalyzed_files: List[str] = Field(
        default_factory=list, description="Files not yet analyzed"
    )
    analysis_errors: Dict[str, str] = Field(
        default_factory=dict, description="Errors by file"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    analysis_version: str = Field(default="1.0")
    analysis_mode: AnalysisMode = Field(default=AnalysisMode.DEEP)
    total_analysis_time_seconds: float = Field(default=0.0)

    # Watcher state
    watcher_active: bool = Field(default=False)
    last_change_detected: Optional[datetime] = Field(default=None)
    pending_changes: List[str] = Field(default_factory=list)

    def get_symbol_by_name(self, name: str) -> Optional[Symbol]:
        """Find symbol by name (simple or qualified)."""
        # Try exact match first
        for symbol in self.symbols.values():
            if symbol.name == name or symbol.qualified_name == name:
                return symbol
        return None

    def get_symbols_in_file(self, file_path: str) -> List[Symbol]:
        """Get all symbols in a file."""
        return [s for s in self.symbols.values() if s.file_path == file_path]

    def get_symbol_count(self) -> int:
        """Get total symbol count."""
        return len(self.symbols)

    def get_file_count(self) -> int:
        """Get total file count."""
        return len(self.files)
