"""Data models for code analysis."""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    UNKNOWN = "unknown"


class NodeType(str, Enum):
    """AST node types."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    CALL = "call"
    LOOP = "loop"
    CONDITIONAL = "conditional"
    TRY_EXCEPT = "try_except"


class ComplexityType(str, Enum):
    """Types of complexity metrics."""

    CYCLOMATIC = "cyclomatic"
    COGNITIVE = "cognitive"
    HALSTEAD = "halstead"


class PatternType(str, Enum):
    """Code pattern types."""

    DESIGN_PATTERN = "design_pattern"
    ANTI_PATTERN = "anti_pattern"
    CODE_SMELL = "code_smell"
    BEST_PRACTICE = "best_practice"


class ASTNode(BaseModel):
    """Represents an AST node."""

    node_type: str = Field(description="Tree-sitter node type")
    start_byte: int = Field(description="Start byte position")
    end_byte: int = Field(description="End byte position")
    start_point: Tuple[int, int] = Field(description="Start line, column")
    end_point: Tuple[int, int] = Field(description="End line, column")
    text: Optional[str] = Field(default=None, description="Node text content")
    children: List["ASTNode"] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CodeEntity(BaseModel):
    """Represents a code entity (class, function, etc.)."""

    name: str = Field(description="Entity name")
    type: NodeType = Field(description="Entity type")
    file_path: str = Field(description="File location")
    line_start: int = Field(description="Start line number")
    line_end: int = Field(description="End line number")
    signature: Optional[str] = Field(
        default=None, description="Function/method signature"
    )
    docstring: Optional[str] = Field(default=None, description="Documentation string")
    complexity: Dict[ComplexityType, float] = Field(default_factory=dict)
    dependencies: List[str] = Field(
        default_factory=list, description="Direct dependencies"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DependencyGraph(BaseModel):
    """Represents code dependencies."""

    nodes: Dict[str, CodeEntity] = Field(description="Entity nodes by ID")
    edges: List[Tuple[str, str, str]] = Field(
        description="Edges as (source, target, relationship_type)"
    )
    imports: Dict[str, List[str]] = Field(description="Import relationships")
    calls: Dict[str, List[str]] = Field(description="Function call relationships")
    inheritance: Dict[str, List[str]] = Field(description="Class inheritance")
    clusters: Optional[Dict[str, List[str]]] = Field(
        default=None, description="Module/package clusters"
    )


class ComplexityMetrics(BaseModel):
    """Code complexity metrics."""

    file_path: str
    language: Language
    cyclomatic_complexity: int = Field(description="McCabe complexity")
    cognitive_complexity: int = Field(description="Cognitive complexity")
    halstead_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Halstead metrics (volume, difficulty, effort)",
    )
    lines_of_code: int = Field(description="Total lines")
    lines_of_code_without_comments: int = Field(description="Code lines only")
    function_count: int = Field(default=0)
    class_count: int = Field(default=0)
    max_nesting_depth: int = Field(default=0)
    average_function_complexity: float = Field(default=0.0)


class Pattern(BaseModel):
    """Detected code pattern."""

    name: str = Field(description="Pattern name")
    type: PatternType = Field(description="Pattern classification")
    confidence: float = Field(ge=0, le=1, description="Detection confidence")
    location: CodeEntity = Field(description="Where pattern was found")
    description: str = Field(description="Pattern description")
    recommendation: Optional[str] = Field(
        default=None, description="Improvement suggestion"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisRequest(BaseModel):
    """Request for code analysis."""

    file_paths: List[str] = Field(description="Files to analyze")
    language: Optional[Language] = Field(
        default=None, description="Force language detection"
    )
    analysis_types: List[str] = Field(
        default_factory=lambda: ["ast", "dependencies", "complexity", "patterns"],
        description="Types of analysis to perform",
    )
    max_depth: int = Field(default=10, description="Max recursion depth")
    include_imports: bool = Field(default=True, description="Analyze imported modules")
    cache_enabled: bool = Field(default=True, description="Use cached ASTs")


class AnalysisResult(BaseModel):
    """Result of code analysis."""

    file_path: str
    language: Language
    ast: Optional[ASTNode] = Field(default=None, description="Abstract syntax tree")
    entities: List[CodeEntity] = Field(
        default_factory=list, description="Code entities found"
    )
    dependencies: Optional[DependencyGraph] = Field(default=None)
    complexity: Optional[ComplexityMetrics] = Field(default=None)
    patterns: List[Pattern] = Field(
        default_factory=list, description="Detected patterns"
    )
    errors: List[str] = Field(default_factory=list, description="Parsing errors")
    analysis_time_ms: int = Field(description="Analysis duration")
    cache_hit: bool = Field(default=False, description="Whether cache was used")


class SemanticSearchRequest(BaseModel):
    """Request for semantic code search."""

    query: str = Field(description="Search query or code snippet")
    scope_paths: List[str] = Field(description="Paths to search within")
    language: Optional[Language] = Field(default=None)
    max_results: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0, le=1)
    search_type: str = Field(default="semantic", description="semantic or exact")


class SemanticSearchResult(BaseModel):
    """Result from semantic search."""

    code_snippet: str = Field(description="Matching code")
    file_path: str = Field(description="File location")
    line_start: int
    line_end: int
    similarity_score: float = Field(ge=0, le=1)
    entity: Optional[CodeEntity] = Field(default=None)
    context: Optional[str] = Field(default=None, description="Surrounding code context")
