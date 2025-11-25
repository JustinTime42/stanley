# PRP-22: Codebase Understanding System

## Goal

**Feature Goal**: Implement a comprehensive codebase understanding system that enables agents to deeply understand existing codebases, maintain awareness of what they know vs. don't know, actively fill knowledge gaps, and keep understanding current through background file watching.

**Deliverable**: CodebaseAnalyzer service with multi-language analysis, knowledge verification system, duplicate detection, background file watcher, and seamless integration with CLI and agent workflows.

**Success Definition**: 
- Analyze medium codebases (10k lines) in <2 minutes, with quick mode in <30 seconds
- Background watcher detects changes within 5 seconds
- Knowledge verification prevents hallucination about non-existent code
- Duplicate detection catches 95%+ of redundant functionality before creation
- Agents can accurately answer "what do you know about X?" queries
- Understanding persists across sessions and updates incrementally

## Why

- **Cold Start Problem**: Agents currently start with zero context about existing projects, leading to irrelevant suggestions
- **Hallucination Risk**: Without verified knowledge, agents may reference code that doesn't exist or misremember function signatures
- **Duplicate Creation**: Agents may create new functions when equivalent functionality already exists
- **Stale Understanding**: Code changes constantly; static analysis becomes outdated immediately
- **Blind Spots**: Agents don't know what they don't know, leading to overconfident incorrect answers
- **Context Window Waste**: Without smart retrieval, agents load irrelevant code into context

## What

Implement a codebase understanding system that:
1. Analyzes and indexes existing codebases with verified, confidence-scored knowledge
2. Maintains real-time awareness through background file watching
3. Tracks knowledge boundaries (known vs. unknown vs. uncertain)
4. Actively identifies and fills knowledge gaps during conversations
5. Detects potential duplicates before new code is created
6. Integrates seamlessly with CLI onboarding and agent workflows

### Success Criteria

- [ ] Quick scan completes in <30 seconds for typical projects
- [ ] Deep analysis completes in <5 minutes for 50k line codebases
- [ ] File watcher detects and processes changes within 5 seconds
- [ ] Knowledge queries return confidence scores (high/medium/low/unknown)
- [ ] Duplicate detection triggers before creating new functions/classes
- [ ] "What do you know about X?" queries return accurate, verified information
- [ ] Gap detection identifies and flags uncertain areas proactively
- [ ] CLI offers onboarding for new codebases automatically
- [ ] Agents refuse to make claims about unanalyzed code areas

## All Needed Context

### Context Completeness Check

_If someone knew nothing about this codebase, would they have everything needed to implement this successfully?_ YES - This PRP includes complete analysis patterns, knowledge verification strategies, and integration examples.

### Documentation & References

```yaml
- file: src/analysis/ast_parser.py
  why: Existing AST analysis infrastructure (PRP-05)
  pattern: Tree-sitter integration, multi-language support
  critical: Reuse for code structure extraction

- file: src/services/memory_service.py
  why: Memory orchestrator for storing understanding (PRP-01)
  pattern: Project memory tier, semantic search
  critical: Store indexed code in vector store

- file: src/services/rag_service.py
  why: RAG implementation for code retrieval (PRP-07)
  pattern: Hybrid search, relevance scoring
  critical: Query understanding efficiently

- file: src/cli/app.py
  why: CLI integration point (PRP-21)
  pattern: Session management, command system
  critical: Onboarding flow, /codebase commands

- url: https://watchdog.readthedocs.io/
  why: Python file system watcher library
  critical: Use Observer pattern, handle debouncing

- url: https://tree-sitter.github.io/tree-sitter/
  why: Multi-language parsing reference
  critical: Already integrated via PRP-05

- file: PRPs/ai_docs/cc_cli.md
  why: Claude Code patterns for codebase awareness
  pattern: /init command, CLAUDE.md memory files
```

### Current Codebase Tree (Relevant Sections)

```bash
agent-swarm/
├── src/
│   ├── analysis/
│   │   ├── ast_parser.py        # Tree-sitter AST parsing
│   │   └── code_analyzer.py     # Code metrics
│   ├── services/
│   │   ├── memory_service.py    # Memory orchestrator
│   │   ├── rag_service.py       # RAG implementation
│   │   └── analysis_service.py  # Analysis orchestrator
│   ├── memory/
│   │   ├── project.py           # Project memory (Qdrant)
│   │   └── hybrid.py            # Hybrid search
│   └── cli/
│       ├── app.py               # CLI application
│       ├── commands/            # Slash commands
│       └── session/             # Session management
```

### Desired Codebase Tree with Files to be Added

```bash
agent-swarm/
├── src/
│   ├── understanding/                    # NEW: Codebase understanding subsystem
│   │   ├── __init__.py                  # Public exports
│   │   ├── analyzer.py                  # Main CodebaseAnalyzer class
│   │   ├── scanner.py                   # File system scanner
│   │   ├── extractors/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                  # Base extractor interface
│   │   │   ├── structure.py             # Project structure extraction
│   │   │   ├── dependencies.py          # Import/dependency graph
│   │   │   ├── conventions.py           # Coding convention detection
│   │   │   ├── documentation.py         # Docstring/comment extraction
│   │   │   └── symbols.py               # Function/class symbol table
│   │   ├── knowledge/
│   │   │   ├── __init__.py
│   │   │   ├── store.py                 # Knowledge storage interface
│   │   │   ├── verification.py          # Knowledge verification system
│   │   │   ├── confidence.py            # Confidence scoring
│   │   │   ├── gaps.py                  # Gap detection and tracking
│   │   │   └── duplicates.py            # Duplicate functionality detector
│   │   ├── watcher/
│   │   │   ├── __init__.py
│   │   │   ├── file_watcher.py          # Background file watcher
│   │   │   ├── change_processor.py      # Incremental update processor
│   │   │   └── debouncer.py             # Change debouncing logic
│   │   ├── queries/
│   │   │   ├── __init__.py
│   │   │   ├── knowledge_query.py       # "What do you know about X?"
│   │   │   ├── similarity_search.py     # Find similar code
│   │   │   └── gap_query.py             # "What don't you know?"
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── understanding_models.py  # Data models
│   │       └── knowledge_models.py      # Knowledge item models
│   ├── services/
│   │   └── understanding_service.py     # NEW: Understanding orchestrator
│   ├── cli/
│   │   └── commands/
│   │       └── codebase_cmd.py          # NEW: /codebase command family
│   └── tests/
│       └── understanding/                # NEW: Understanding tests
│           ├── test_analyzer.py
│           ├── test_knowledge.py
│           ├── test_watcher.py
│           └── test_duplicates.py
├── .agent-swarm/                         # NEW: Project config directory
│   ├── config.yaml                      # Project configuration
│   └── understanding/                   # Persisted understanding
│       ├── project_summary.md           # Human-readable summary
│       ├── structure.json               # Project structure
│       ├── symbols.json                 # Symbol table
│       ├── dependencies.json            # Dependency graph
│       ├── conventions.yaml             # Detected conventions
│       ├── knowledge_index.json         # Knowledge metadata
│       └── gaps.json                    # Known knowledge gaps
└── requirements.txt                      # MODIFY: Add watchdog
```

### Known Gotchas of our Codebase & Library Quirks

```python
# CRITICAL: Watchdog events fire multiple times for single save
# Must debounce with 500ms-1s window to batch rapid changes
from watchdog.observers import Observer
# Use DebouncedHandler wrapper, not raw FileSystemEventHandler

# CRITICAL: Large codebases can OOM during full analysis
# Process files in batches, use generators, don't load all ASTs at once
for batch in chunked(files, batch_size=100):
    process_batch(batch)
    gc.collect()  # Help with memory pressure

# CRITICAL: Circular imports in dependency graph
# Must handle cycles gracefully, track visited nodes
def build_dependency_graph(entry_point, visited=None):
    visited = visited or set()
    if entry_point in visited:
        return  # Cycle detected, skip
    visited.add(entry_point)
    # ... continue traversal

# CRITICAL: Binary files and generated code
# Must filter intelligently, not just by extension
SKIP_PATTERNS = [
    "node_modules", "__pycache__", ".git", "venv",
    "*.min.js", "*.bundle.js", "*.pyc", "*.so",
    "package-lock.json", "poetry.lock", "yarn.lock",
]

# CRITICAL: Confidence scores must be calibrated
# Don't claim high confidence without verification
class ConfidenceLevel(Enum):
    VERIFIED = "verified"      # Directly analyzed, AST parsed
    INFERRED = "inferred"      # Pattern-based inference
    UNCERTAIN = "uncertain"    # Guess based on naming/context
    UNKNOWN = "unknown"        # No information

# CRITICAL: Knowledge verification before agent claims
# Agent must check knowledge store before making statements
async def verify_before_claim(claim: str, knowledge_store: KnowledgeStore):
    verification = await knowledge_store.verify(claim)
    if verification.confidence < ConfidenceLevel.INFERRED:
        return "I'm not certain about this. Let me check..."
    return claim

# CRITICAL: Symbol table must track file locations
# For "jump to definition" and modification tracking
@dataclass
class Symbol:
    name: str
    kind: SymbolKind  # function, class, variable, etc.
    file_path: str
    line_start: int
    line_end: int
    signature: Optional[str]
    docstring: Optional[str]
    hash: str  # Content hash for change detection
```

## Implementation Blueprint

### Data Models and Structure

```python
# src/understanding/models/understanding_models.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from enum import Enum
from pathlib import Path


class ConfidenceLevel(str, Enum):
    """Knowledge confidence levels"""
    VERIFIED = "verified"      # Directly analyzed, high certainty
    INFERRED = "inferred"      # Pattern-based, medium certainty
    UNCERTAIN = "uncertain"    # Contextual guess, low certainty
    UNKNOWN = "unknown"        # No information available
    STALE = "stale"           # Was verified but file changed


class SymbolKind(str, Enum):
    """Types of code symbols"""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    TYPE = "type"


class Symbol(BaseModel):
    """A code symbol (function, class, variable, etc.)"""
    id: str = Field(description="Unique symbol identifier")
    name: str = Field(description="Symbol name")
    qualified_name: str = Field(description="Fully qualified name (module.class.method)")
    kind: SymbolKind = Field(description="Symbol type")
    
    # Location
    file_path: str = Field(description="File containing symbol")
    line_start: int = Field(description="Starting line number")
    line_end: int = Field(description="Ending line number")
    
    # Signature and documentation
    signature: Optional[str] = Field(default=None, description="Function/method signature")
    docstring: Optional[str] = Field(default=None, description="Documentation string")
    return_type: Optional[str] = Field(default=None, description="Return type annotation")
    parameters: List[Dict[str, Any]] = Field(default_factory=list, description="Parameters with types")
    
    # Relationships
    parent_symbol: Optional[str] = Field(default=None, description="Parent class/module")
    calls: List[str] = Field(default_factory=list, description="Symbols this calls")
    called_by: List[str] = Field(default_factory=list, description="Symbols that call this")
    imports: List[str] = Field(default_factory=list, description="Imported symbols")
    
    # Verification
    content_hash: str = Field(description="Hash of symbol content for change detection")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.VERIFIED)
    last_verified: datetime = Field(default_factory=datetime.now)
    
    # Semantic
    description: Optional[str] = Field(default=None, description="AI-generated description")
    tags: List[str] = Field(default_factory=list, description="Semantic tags")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")


class FileInfo(BaseModel):
    """Information about a source file"""
    path: str
    relative_path: str
    language: str
    size_bytes: int
    line_count: int
    last_modified: datetime
    content_hash: str
    
    # Analysis results
    symbols: List[str] = Field(default_factory=list, description="Symbol IDs in file")
    imports: List[str] = Field(default_factory=list, description="Import statements")
    exports: List[str] = Field(default_factory=list, description="Exported symbols")
    
    # Status
    analyzed: bool = Field(default=False)
    analysis_errors: List[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.UNKNOWN)


class ProjectStructure(BaseModel):
    """Project structure understanding"""
    root_path: str
    project_name: str
    detected_type: str = Field(description="python, javascript, typescript, etc.")
    detected_framework: Optional[str] = Field(default=None, description="FastAPI, React, etc.")
    
    # Structure
    source_directories: List[str] = Field(default_factory=list)
    test_directories: List[str] = Field(default_factory=list)
    config_files: List[str] = Field(default_factory=list)
    entry_points: List[str] = Field(default_factory=list)
    
    # Statistics
    total_files: int = Field(default=0)
    total_lines: int = Field(default=0)
    files_by_language: Dict[str, int] = Field(default_factory=dict)
    
    # Conventions
    naming_convention: str = Field(default="unknown", description="snake_case, camelCase, etc.")
    test_framework: Optional[str] = Field(default=None)
    package_manager: Optional[str] = Field(default=None)


class DependencyGraph(BaseModel):
    """Module dependency graph"""
    nodes: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Module nodes")
    edges: List[Dict[str, str]] = Field(default_factory=list, description="Import edges")
    
    # Analysis
    entry_points: List[str] = Field(default_factory=list)
    leaf_modules: List[str] = Field(default_factory=list)
    cycles: List[List[str]] = Field(default_factory=list, description="Detected cycles")
    
    def get_dependents(self, module: str) -> List[str]:
        """Get modules that depend on this module"""
        return [e["from"] for e in self.edges if e["to"] == module]
    
    def get_dependencies(self, module: str) -> List[str]:
        """Get modules this module depends on"""
        return [e["to"] for e in self.edges if e["from"] == module]


class KnowledgeGap(BaseModel):
    """A detected gap in understanding"""
    id: str
    area: str = Field(description="Code area with gap")
    description: str = Field(description="What's unknown")
    severity: str = Field(description="high, medium, low")
    
    # Context
    related_files: List[str] = Field(default_factory=list)
    related_symbols: List[str] = Field(default_factory=list)
    triggered_by: Optional[str] = Field(default=None, description="Query that exposed gap")
    
    # Resolution
    suggested_actions: List[str] = Field(default_factory=list)
    resolved: bool = Field(default=False)
    resolved_at: Optional[datetime] = Field(default=None)


class DuplicateCandidate(BaseModel):
    """Potential duplicate functionality detection"""
    new_symbol: str = Field(description="Proposed new symbol")
    existing_symbol: str = Field(description="Existing similar symbol")
    similarity_score: float = Field(description="0.0-1.0 similarity")
    similarity_type: str = Field(description="name, signature, semantic, implementation")
    
    # Details
    new_description: str
    existing_description: str
    recommendation: str = Field(description="use_existing, extend, create_new")


class CodebaseUnderstanding(BaseModel):
    """Complete codebase understanding state"""
    project_id: str
    root_path: str
    
    # Core understanding
    structure: ProjectStructure
    dependency_graph: DependencyGraph
    symbols: Dict[str, Symbol] = Field(default_factory=dict)
    files: Dict[str, FileInfo] = Field(default_factory=dict)
    
    # Knowledge state
    knowledge_gaps: List[KnowledgeGap] = Field(default_factory=list)
    unanalyzed_files: List[str] = Field(default_factory=list)
    analysis_errors: Dict[str, str] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    analysis_version: str = Field(default="1.0")
    total_analysis_time_seconds: float = Field(default=0.0)
    
    # Watcher state
    watcher_active: bool = Field(default=False)
    last_change_detected: Optional[datetime] = Field(default=None)
    pending_changes: List[str] = Field(default_factory=list)


# src/understanding/models/knowledge_models.py

class KnowledgeQuery(BaseModel):
    """Query about codebase knowledge"""
    query: str
    query_type: str = Field(description="symbol, file, concept, relationship")
    context: Optional[str] = Field(default=None)


class KnowledgeResponse(BaseModel):
    """Response to knowledge query with confidence"""
    query: str
    answer: str
    confidence: ConfidenceLevel
    
    # Evidence
    sources: List[str] = Field(default_factory=list, description="Files/symbols supporting answer")
    verified_claims: List[str] = Field(default_factory=list)
    uncertain_claims: List[str] = Field(default_factory=list)
    
    # Gaps
    knowledge_gaps: List[str] = Field(default_factory=list, description="What we don't know")
    suggested_investigation: List[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """Result of verifying a claim about the codebase"""
    claim: str
    verified: bool
    confidence: ConfidenceLevel
    
    # Evidence
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    
    # Correction
    correction: Optional[str] = Field(default=None, description="Corrected claim if wrong")
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/understanding/models/
  description: Data models for understanding system
  depends_on: []
  estimated_time: 3 hours
  files:
    - src/understanding/models/__init__.py
    - src/understanding/models/understanding_models.py
    - src/understanding/models/knowledge_models.py
  validation: |
    python -c "from src.understanding.models import CodebaseUnderstanding, Symbol, KnowledgeResponse; print('OK')"

Task 2: CREATE src/understanding/scanner.py
  description: File system scanner with filtering
  depends_on: [Task 1]
  estimated_time: 3 hours
  validation: |
    python -c "from src.understanding import FileScanner; s = FileScanner('.'); print(len(list(s.scan())))"
  implementation_notes: |
    - Respect .gitignore patterns
    - Skip binary files, node_modules, __pycache__, etc.
    - Detect language by extension and content
    - Generate content hashes for change detection
    - Support include/exclude patterns

Task 3: CREATE src/understanding/extractors/base.py
  description: Base extractor interface
  depends_on: [Task 1]
  estimated_time: 1 hour
  implementation_notes: |
    - Abstract base class for all extractors
    - Common extraction interface
    - Error handling patterns

Task 4: CREATE src/understanding/extractors/structure.py
  description: Project structure extractor
  depends_on: [Task 3, Task 2]
  estimated_time: 4 hours
  validation: |
    pytest src/tests/understanding/test_extractors.py::test_structure -v
  implementation_notes: |
    - Detect project type (Python, JS, TS, etc.)
    - Find source directories, test directories
    - Identify entry points (main.py, index.js, etc.)
    - Detect frameworks (FastAPI, React, Django, etc.)
    - Find config files (pyproject.toml, package.json, etc.)

Task 5: CREATE src/understanding/extractors/symbols.py
  description: Symbol table extractor using AST
  depends_on: [Task 3]
  estimated_time: 6 hours
  validation: |
    pytest src/tests/understanding/test_extractors.py::test_symbols -v
  implementation_notes: |
    - Integrate with existing ast_parser.py (PRP-05)
    - Extract functions, classes, methods, variables
    - Capture signatures, docstrings, type annotations
    - Build qualified names (module.class.method)
    - Generate content hashes per symbol

Task 6: CREATE src/understanding/extractors/dependencies.py
  description: Dependency graph extractor
  depends_on: [Task 5]
  estimated_time: 4 hours
  validation: |
    pytest src/tests/understanding/test_extractors.py::test_dependencies -v
  implementation_notes: |
    - Parse import statements
    - Build import graph (internal and external)
    - Detect circular dependencies
    - Identify entry points and leaf modules
    - Track symbol-level dependencies (what calls what)

Task 7: CREATE src/understanding/extractors/conventions.py
  description: Coding convention detector
  depends_on: [Task 5]
  estimated_time: 3 hours
  implementation_notes: |
    - Detect naming conventions (snake_case, camelCase, etc.)
    - Identify test patterns (pytest, unittest, jest)
    - Find documentation patterns
    - Detect formatting style

Task 8: CREATE src/understanding/extractors/documentation.py
  description: Documentation extractor
  depends_on: [Task 5]
  estimated_time: 3 hours
  implementation_notes: |
    - Extract docstrings from all symbols
    - Parse README files
    - Extract inline comments (especially TODOs, FIXMEs)
    - Identify API documentation

Task 9: CREATE src/understanding/knowledge/store.py
  description: Knowledge storage with vector embeddings
  depends_on: [Task 1]
  estimated_time: 4 hours
  validation: |
    pytest src/tests/understanding/test_knowledge.py::test_store -v
  implementation_notes: |
    - Integrate with Memory Service (Qdrant)
    - Store symbols with embeddings for semantic search
    - Support fast lookup by qualified name
    - Track confidence levels per item
    - Persist to .agent-swarm/understanding/

Task 10: CREATE src/understanding/knowledge/confidence.py
  description: Confidence scoring system
  depends_on: [Task 9]
  estimated_time: 3 hours
  implementation_notes: |
    - Score based on: direct analysis, inference, age, verification
    - Degrade confidence over time without verification
    - Boost confidence when claims are confirmed
    - Track confidence history

Task 11: CREATE src/understanding/knowledge/verification.py
  description: Knowledge verification system
  depends_on: [Task 10]
  estimated_time: 5 hours
  validation: |
    pytest src/tests/understanding/test_knowledge.py::test_verification -v
  implementation_notes: |
    CRITICAL: This is the anti-hallucination system
    - Verify claims against symbol table before agent makes them
    - Check if referenced functions/classes exist
    - Verify parameter names and types
    - Verify file paths exist
    - Return confidence level with evidence
    - Flag claims that cannot be verified

Task 12: CREATE src/understanding/knowledge/gaps.py
  description: Knowledge gap detection and tracking
  depends_on: [Task 11]
  estimated_time: 4 hours
  validation: |
    pytest src/tests/understanding/test_knowledge.py::test_gaps -v
  implementation_notes: |
    - Track what areas haven't been analyzed
    - Detect when queries hit unknown areas
    - Suggest files/areas to analyze
    - Priority rank gaps by query frequency
    - Auto-fill gaps opportunistically

Task 13: CREATE src/understanding/knowledge/duplicates.py
  description: Duplicate functionality detector
  depends_on: [Task 9]
  estimated_time: 5 hours
  validation: |
    pytest src/tests/understanding/test_duplicates.py -v
  implementation_notes: |
    CRITICAL: Prevents creating redundant code
    - Check before creating new function/class
    - Compare by: name similarity, signature similarity, semantic similarity
    - Use embeddings for semantic comparison
    - Return existing alternatives with recommendation
    - Configurable similarity threshold (default 0.8)

Task 14: CREATE src/understanding/watcher/file_watcher.py
  description: Background file system watcher
  depends_on: [Task 2]
  estimated_time: 4 hours
  validation: |
    pytest src/tests/understanding/test_watcher.py -v
  implementation_notes: |
    - Use watchdog library
    - Watch source directories only (respect ignore patterns)
    - Handle all event types (create, modify, delete, move)
    - Run in background thread/process
    - Graceful shutdown on CLI exit

Task 15: CREATE src/understanding/watcher/debouncer.py
  description: Change debouncing for rapid edits
  depends_on: [Task 14]
  estimated_time: 2 hours
  implementation_notes: |
    - Batch rapid changes (500ms-1s window)
    - Deduplicate same-file events
    - Priority queue for processing order
    - Configurable debounce delay

Task 16: CREATE src/understanding/watcher/change_processor.py
  description: Incremental update processor
  depends_on: [Task 14, Task 15, Task 9]
  estimated_time: 4 hours
  implementation_notes: |
    - Process file changes incrementally
    - Update only affected symbols
    - Recalculate affected dependencies
    - Mark stale knowledge as STALE confidence
    - Trigger gap detection for deleted code

Task 17: CREATE src/understanding/analyzer.py
  description: Main CodebaseAnalyzer class
  depends_on: [Task 4, Task 5, Task 6, Task 7, Task 8, Task 9]
  estimated_time: 6 hours
  validation: |
    pytest src/tests/understanding/test_analyzer.py -v
  implementation_notes: |
    - Orchestrate all extractors
    - Support quick mode (structure only) and deep mode (full analysis)
    - Progress callbacks for UI
    - Memory-efficient batched processing
    - Async analysis with cancellation support
    - Persist results to .agent-swarm/understanding/

Task 18: CREATE src/understanding/queries/knowledge_query.py
  description: "What do you know about X?" query handler
  depends_on: [Task 9, Task 11]
  estimated_time: 4 hours
  validation: |
    pytest src/tests/understanding/test_queries.py -v
  implementation_notes: |
    - Natural language query parsing
    - Symbol lookup by name, fuzzy match
    - Semantic search for concepts
    - Return with confidence scores
    - Highlight knowledge gaps

Task 19: CREATE src/understanding/queries/similarity_search.py
  description: Find similar code functionality
  depends_on: [Task 9]
  estimated_time: 3 hours
  implementation_notes: |
    - Vector similarity search
    - Find functions with similar purpose
    - Find classes with similar structure
    - Support code snippet input

Task 20: CREATE src/understanding/queries/gap_query.py
  description: "What don't you know?" query handler
  depends_on: [Task 12]
  estimated_time: 2 hours
  implementation_notes: |
    - List unanalyzed areas
    - Show low-confidence knowledge
    - Suggest investigation priorities

Task 21: CREATE src/services/understanding_service.py
  description: Understanding orchestrator service
  depends_on: [Task 17, Task 14, Task 11, Task 13]
  estimated_time: 5 hours
  validation: |
    pytest src/tests/understanding/test_service.py -v
  implementation_notes: |
    - Facade for all understanding operations
    - Manage watcher lifecycle
    - Integrate with memory service
    - Expose to agents and CLI

Task 22: CREATE src/cli/commands/codebase_cmd.py
  description: /codebase command family for CLI
  depends_on: [Task 21]
  estimated_time: 4 hours
  validation: |
    python cli.py
    > /codebase
    > /codebase analyze
    > /codebase find authentication
  implementation_notes: |
    Commands:
    - /codebase - Show understanding status
    - /codebase analyze [path] - Analyze codebase
    - /codebase refresh - Re-analyze changed files
    - /codebase find <query> - Find code by description
    - /codebase deps [module] - Show dependencies
    - /codebase symbols [file] - List symbols in file
    - /codebase gaps - Show knowledge gaps
    - /codebase watch start|stop|status - Control watcher

Task 23: UPDATE src/cli/app.py
  description: Add onboarding flow for new codebases
  depends_on: [Task 22]
  estimated_time: 3 hours
  implementation_notes: |
    - Detect new codebase on startup
    - Offer quick/deep/skip analysis options
    - Show progress during analysis
    - Start watcher after analysis
    - Background analysis option

Task 24: CREATE agent integration hooks
  description: Integrate understanding with agent workflow
  depends_on: [Task 21, Task 11, Task 13]
  estimated_time: 5 hours
  files:
    - src/agents/mixins/understanding_mixin.py
  implementation_notes: |
    - Mixin for agents to query understanding
    - verify_before_claim() hook
    - check_for_duplicates() before creating code
    - get_relevant_context() for task planning
    - flag_uncertainty() when confidence is low

Task 25: CREATE persistence layer
  description: Persist understanding to .agent-swarm/
  depends_on: [Task 17]
  estimated_time: 3 hours
  implementation_notes: |
    Files to create/update:
    - .agent-swarm/understanding/project_summary.md (human-readable)
    - .agent-swarm/understanding/structure.json
    - .agent-swarm/understanding/symbols.json
    - .agent-swarm/understanding/dependencies.json
    - .agent-swarm/understanding/conventions.yaml
    - .agent-swarm/understanding/knowledge_index.json
    - .agent-swarm/understanding/gaps.json

Task 26: CREATE test suite
  description: Comprehensive tests for understanding system
  depends_on: [Task 21]
  estimated_time: 4 hours
  files:
    - src/tests/understanding/test_analyzer.py
    - src/tests/understanding/test_knowledge.py
    - src/tests/understanding/test_watcher.py
    - src/tests/understanding/test_duplicates.py
    - src/tests/understanding/test_queries.py
    - src/tests/understanding/test_service.py
  validation: |
    pytest src/tests/understanding/ -v --cov=src/understanding

Task 27: UPDATE requirements.txt
  description: Add understanding dependencies
  depends_on: []
  estimated_time: 15 minutes
  implementation_notes: |
    Add:
    - watchdog>=3.0.0
    - python-frontmatter>=1.0.0  (if not already added)
```

### Core Implementation Patterns

```python
# src/understanding/analyzer.py - Main analyzer
import asyncio
from pathlib import Path
from typing import Optional, Callable, AsyncIterator
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
from .models import CodebaseUnderstanding, ProjectStructure, AnalysisProgress


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
        memory_service: Optional["MemoryOrchestrator"] = None,
    ):
        self.root_path = Path(root_path).resolve()
        self.project_id = self._generate_project_id()
        self.knowledge_store = knowledge_store or KnowledgeStore(memory_service)
        
        # Initialize extractors
        self.scanner = FileScanner(self.root_path)
        self.structure_extractor = StructureExtractor()
        self.symbol_extractor = SymbolExtractor()
        self.dependency_extractor = DependencyExtractor()
        self.convention_extractor = ConventionExtractor()
        self.doc_extractor = DocumentationExtractor()
        
        self._understanding: Optional[CodebaseUnderstanding] = None
    
    async def analyze(
        self,
        mode: str = "deep",  # "quick" | "deep" | "exhaustive"
        focus_paths: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[AnalysisProgress], None]] = None,
    ) -> CodebaseUnderstanding:
        """
        Analyze codebase and build understanding.
        
        Args:
            mode: Analysis depth
            focus_paths: Optional paths to prioritize
            exclude_patterns: Additional patterns to exclude
            progress_callback: Progress update callback
            
        Returns:
            Complete CodebaseUnderstanding
        """
        start_time = datetime.now()
        
        # Step 1: Scan files
        self._report_progress(progress_callback, "Scanning files...", 0)
        files = list(self.scanner.scan(exclude_patterns))
        
        if focus_paths:
            files = self._prioritize_paths(files, focus_paths)
        
        # Step 2: Extract structure (always quick)
        self._report_progress(progress_callback, "Analyzing structure...", 10)
        structure = await self.structure_extractor.extract(self.root_path, files)
        
        if mode == "quick":
            # Quick mode: structure only
            self._understanding = CodebaseUnderstanding(
                project_id=self.project_id,
                root_path=str(self.root_path),
                structure=structure,
                dependency_graph=DependencyGraph(),
            )
            self._understanding.unanalyzed_files = [f.path for f in files]
            return self._understanding
        
        # Step 3: Extract symbols (batched for memory efficiency)
        self._report_progress(progress_callback, "Extracting symbols...", 20)
        symbols = {}
        file_infos = {}
        
        batch_size = 50
        for i, batch in enumerate(self._batch(files, batch_size)):
            batch_symbols, batch_files = await self.symbol_extractor.extract_batch(batch)
            symbols.update(batch_symbols)
            file_infos.update(batch_files)
            
            progress = 20 + (60 * (i * batch_size) / len(files))
            self._report_progress(
                progress_callback, 
                f"Extracting symbols... ({len(symbols)} found)", 
                progress
            )
        
        # Step 4: Build dependency graph
        self._report_progress(progress_callback, "Mapping dependencies...", 80)
        dependency_graph = await self.dependency_extractor.extract(
            files, symbols, file_infos
        )
        
        # Step 5: Detect conventions
        self._report_progress(progress_callback, "Detecting conventions...", 90)
        conventions = await self.convention_extractor.extract(symbols, files)
        structure.naming_convention = conventions.get("naming", "unknown")
        structure.test_framework = conventions.get("test_framework")
        
        # Step 6: Build understanding
        self._report_progress(progress_callback, "Finalizing...", 95)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        self._understanding = CodebaseUnderstanding(
            project_id=self.project_id,
            root_path=str(self.root_path),
            structure=structure,
            dependency_graph=dependency_graph,
            symbols=symbols,
            files=file_infos,
            total_analysis_time_seconds=elapsed,
        )
        
        # Step 7: Store in knowledge store
        await self.knowledge_store.store_understanding(self._understanding)
        
        # Step 8: Identify initial knowledge gaps
        gaps = await self._identify_gaps(self._understanding)
        self._understanding.knowledge_gaps = gaps
        
        # Step 9: Persist to disk
        await self._persist_understanding()
        
        self._report_progress(progress_callback, "Complete!", 100)
        
        return self._understanding
    
    async def analyze_quick(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> CodebaseUnderstanding:
        """Quick analysis - structure only, <30 seconds"""
        return await self.analyze(mode="quick", progress_callback=progress_callback)
    
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
                new_symbols, file_info = await self.symbol_extractor.extract_file(
                    file_path
                )
                self._understanding.symbols.update(new_symbols)
                self._understanding.files[file_path] = file_info
            else:
                # File deleted
                if file_path in self._understanding.files:
                    del self._understanding.files[file_path]
        
        # Update dependency graph
        self._understanding.dependency_graph = await self.dependency_extractor.extract(
            list(self._understanding.files.values()),
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


# src/understanding/knowledge/verification.py - Anti-hallucination system
from typing import Tuple, List, Optional
from ..models import (
    VerificationResult,
    ConfidenceLevel,
    Symbol,
)


class KnowledgeVerifier:
    """
    Verify claims about the codebase before agents make them.
    
    CRITICAL: This is the anti-hallucination system
    PATTERN: Check symbol table and knowledge store before any claim
    """
    
    def __init__(self, knowledge_store: "KnowledgeStore"):
        self.store = knowledge_store
    
    async def verify_symbol_exists(
        self,
        symbol_name: str,
        expected_kind: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a symbol (function, class, etc.) exists.
        
        Args:
            symbol_name: Name or qualified name of symbol
            expected_kind: Optional expected kind (function, class, etc.)
            
        Returns:
            VerificationResult with confidence
        """
        # Try exact match first
        symbol = await self.store.get_symbol(symbol_name)
        
        if symbol:
            # Check kind if specified
            if expected_kind and symbol.kind.value != expected_kind:
                return VerificationResult(
                    claim=f"{symbol_name} is a {expected_kind}",
                    verified=False,
                    confidence=ConfidenceLevel.VERIFIED,
                    contradicting_evidence=[
                        f"{symbol_name} is actually a {symbol.kind.value}"
                    ],
                    correction=f"{symbol_name} is a {symbol.kind.value}, not a {expected_kind}",
                )
            
            return VerificationResult(
                claim=f"{symbol_name} exists",
                verified=True,
                confidence=symbol.confidence,
                supporting_evidence=[
                    f"Found in {symbol.file_path}:{symbol.line_start}"
                ],
            )
        
        # Try fuzzy match
        similar = await self.store.find_similar_symbols(symbol_name, limit=3)
        
        if similar:
            return VerificationResult(
                claim=f"{symbol_name} exists",
                verified=False,
                confidence=ConfidenceLevel.VERIFIED,
                contradicting_evidence=[f"Symbol '{symbol_name}' not found"],
                correction=f"Did you mean: {', '.join(s.name for s in similar)}?",
            )
        
        return VerificationResult(
            claim=f"{symbol_name} exists",
            verified=False,
            confidence=ConfidenceLevel.UNKNOWN,
            contradicting_evidence=[
                f"Symbol '{symbol_name}' not found in analyzed codebase"
            ],
        )
    
    async def verify_function_signature(
        self,
        function_name: str,
        expected_params: List[str],
    ) -> VerificationResult:
        """Verify a function has the expected parameters"""
        symbol = await self.store.get_symbol(function_name)
        
        if not symbol:
            return VerificationResult(
                claim=f"{function_name} has params {expected_params}",
                verified=False,
                confidence=ConfidenceLevel.UNKNOWN,
                contradicting_evidence=[f"Function '{function_name}' not found"],
            )
        
        actual_params = [p["name"] for p in symbol.parameters]
        
        if set(expected_params) == set(actual_params):
            return VerificationResult(
                claim=f"{function_name} has params {expected_params}",
                verified=True,
                confidence=symbol.confidence,
                supporting_evidence=[f"Signature: {symbol.signature}"],
            )
        
        return VerificationResult(
            claim=f"{function_name} has params {expected_params}",
            verified=False,
            confidence=symbol.confidence,
            contradicting_evidence=[f"Actual params: {actual_params}"],
            correction=f"{function_name} actually has parameters: {actual_params}",
        )
    
    async def verify_file_exists(self, file_path: str) -> VerificationResult:
        """Verify a file exists in the codebase"""
        file_info = await self.store.get_file(file_path)
        
        if file_info:
            return VerificationResult(
                claim=f"File {file_path} exists",
                verified=True,
                confidence=ConfidenceLevel.VERIFIED,
                supporting_evidence=[f"File has {file_info.line_count} lines"],
            )
        
        # Check if similar file exists
        similar = await self.store.find_similar_files(file_path)
        
        return VerificationResult(
            claim=f"File {file_path} exists",
            verified=False,
            confidence=ConfidenceLevel.VERIFIED,
            contradicting_evidence=[f"File '{file_path}' not found"],
            correction=f"Similar files: {', '.join(similar)}" if similar else None,
        )
    
    async def verify_claim(self, claim: str) -> VerificationResult:
        """
        Verify a natural language claim about the codebase.
        
        PATTERN: Parse claim, extract entities, verify each
        """
        # Extract entities from claim
        entities = await self._extract_claim_entities(claim)
        
        results = []
        for entity_type, entity_name in entities:
            if entity_type == "symbol":
                results.append(await self.verify_symbol_exists(entity_name))
            elif entity_type == "file":
                results.append(await self.verify_file_exists(entity_name))
        
        # Aggregate results
        if not results:
            return VerificationResult(
                claim=claim,
                verified=False,
                confidence=ConfidenceLevel.UNKNOWN,
                contradicting_evidence=["Could not extract verifiable entities from claim"],
            )
        
        all_verified = all(r.verified for r in results)
        min_confidence = min(r.confidence for r in results)
        
        return VerificationResult(
            claim=claim,
            verified=all_verified,
            confidence=min_confidence,
            supporting_evidence=[e for r in results for e in r.supporting_evidence],
            contradicting_evidence=[e for r in results for e in r.contradicting_evidence],
        )


# src/understanding/knowledge/duplicates.py - Duplicate detection
class DuplicateDetector:
    """
    Detect potential duplicate functionality before creating new code.
    
    CRITICAL: Prevents redundant code creation
    PATTERN: Check similarity before every new function/class
    """
    
    def __init__(
        self,
        knowledge_store: "KnowledgeStore",
        similarity_threshold: float = 0.8,
    ):
        self.store = knowledge_store
        self.threshold = similarity_threshold
    
    async def check_before_create(
        self,
        proposed_name: str,
        proposed_description: str,
        proposed_kind: str = "function",
        proposed_signature: Optional[str] = None,
    ) -> List[DuplicateCandidate]:
        """
        Check for duplicates before creating new symbol.
        
        Args:
            proposed_name: Name of proposed new symbol
            proposed_description: What it should do
            proposed_kind: function, class, method, etc.
            proposed_signature: Optional function signature
            
        Returns:
            List of potential duplicate candidates
        """
        candidates = []
        
        # Check 1: Name similarity
        name_matches = await self.store.find_similar_symbols(
            proposed_name,
            kind=proposed_kind,
            limit=5,
        )
        
        for match in name_matches:
            similarity = self._name_similarity(proposed_name, match.name)
            if similarity >= self.threshold:
                candidates.append(DuplicateCandidate(
                    new_symbol=proposed_name,
                    existing_symbol=match.qualified_name,
                    similarity_score=similarity,
                    similarity_type="name",
                    new_description=proposed_description,
                    existing_description=match.docstring or match.description or "",
                    recommendation=self._recommend(similarity, "name"),
                ))
        
        # Check 2: Signature similarity (if provided)
        if proposed_signature:
            sig_matches = await self.store.find_by_signature_similarity(
                proposed_signature,
                kind=proposed_kind,
                limit=5,
            )
            
            for match in sig_matches:
                if match.qualified_name not in [c.existing_symbol for c in candidates]:
                    similarity = self._signature_similarity(
                        proposed_signature, match.signature
                    )
                    if similarity >= self.threshold:
                        candidates.append(DuplicateCandidate(
                            new_symbol=proposed_name,
                            existing_symbol=match.qualified_name,
                            similarity_score=similarity,
                            similarity_type="signature",
                            new_description=proposed_description,
                            existing_description=match.docstring or "",
                            recommendation=self._recommend(similarity, "signature"),
                        ))
        
        # Check 3: Semantic similarity (most important)
        semantic_matches = await self.store.semantic_search(
            proposed_description,
            kind=proposed_kind,
            limit=5,
        )
        
        for match, score in semantic_matches:
            if match.qualified_name not in [c.existing_symbol for c in candidates]:
                if score >= self.threshold:
                    candidates.append(DuplicateCandidate(
                        new_symbol=proposed_name,
                        existing_symbol=match.qualified_name,
                        similarity_score=score,
                        similarity_type="semantic",
                        new_description=proposed_description,
                        existing_description=match.docstring or match.description or "",
                        recommendation=self._recommend(score, "semantic"),
                    ))
        
        # Sort by similarity score
        candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        
        return candidates
    
    def _recommend(self, similarity: float, match_type: str) -> str:
        """Generate recommendation based on similarity"""
        if similarity >= 0.95:
            return "use_existing"
        elif similarity >= 0.85:
            if match_type == "semantic":
                return "use_existing"
            return "consider_existing"
        elif similarity >= 0.7:
            return "extend_existing"
        return "create_new"


# src/understanding/watcher/file_watcher.py - Background watcher
import asyncio
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent


class CodebaseWatcher:
    """
    Background file system watcher for keeping understanding current.
    
    PATTERN: Run in background, debounce rapid changes
    CRITICAL: Must not block main thread
    """
    
    def __init__(
        self,
        root_path: str,
        analyzer: "CodebaseAnalyzer",
        debounce_seconds: float = 1.0,
        ignore_patterns: Optional[List[str]] = None,
    ):
        self.root_path = Path(root_path)
        self.analyzer = analyzer
        self.debounce_seconds = debounce_seconds
        self.ignore_patterns = ignore_patterns or [
            "*.pyc", "__pycache__", ".git", "node_modules",
            ".agent-swarm", "*.log", "*.tmp",
        ]
        
        self._observer: Optional[Observer] = None
        self._handler: Optional["DebouncedHandler"] = None
        self._running = False
        self._change_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
    
    def start(self):
        """Start watching for file changes"""
        if self._running:
            return
        
        self._handler = DebouncedHandler(
            callback=self._on_change,
            debounce_seconds=self.debounce_seconds,
            ignore_patterns=self.ignore_patterns,
        )
        
        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            str(self.root_path),
            recursive=True,
        )
        self._observer.start()
        self._running = True
        
        # Start async processor
        self._processor_task = asyncio.create_task(self._process_changes())
        
        logger.info(f"Started watching: {self.root_path}")
    
    def stop(self):
        """Stop watching"""
        if not self._running:
            return
        
        self._running = False
        
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        
        if self._processor_task:
            self._processor_task.cancel()
            self._processor_task = None
        
        logger.info("Stopped watching")
    
    def _on_change(self, changed_files: List[str]):
        """Callback when files change (debounced)"""
        for file_path in changed_files:
            self._change_queue.put_nowait(file_path)
    
    async def _process_changes(self):
        """Process file changes asynchronously"""
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
                if batch and (batch_deadline and now >= batch_deadline):
                    unique_files = list(set(batch))
                    logger.info(f"Processing {len(unique_files)} changed files")
                    
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


class DebouncedHandler(FileSystemEventHandler):
    """Debounced file system event handler"""
    
    def __init__(
        self,
        callback: Callable[[List[str]], None],
        debounce_seconds: float,
        ignore_patterns: List[str],
    ):
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self.ignore_patterns = ignore_patterns
        self._pending: Dict[str, float] = {}
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
    
    def on_any_event(self, event: FileSystemEvent):
        if event.is_directory:
            return
        
        # Check ignore patterns
        if self._should_ignore(event.src_path):
            return
        
        with self._lock:
            self._pending[event.src_path] = time.time()
            
            # Reset timer
            if self._timer:
                self._timer.cancel()
            
            self._timer = threading.Timer(
                self.debounce_seconds,
                self._flush,
            )
            self._timer.start()
    
    def _flush(self):
        with self._lock:
            if self._pending:
                files = list(self._pending.keys())
                self._pending.clear()
                self.callback(files)
    
    def _should_ignore(self, path: str) -> bool:
        from fnmatch import fnmatch
        return any(fnmatch(path, pattern) for pattern in self.ignore_patterns)
```

### Integration Points

```yaml
MEMORY_SERVICE:
  - Store symbols with embeddings in project memory
  - Use semantic search for similarity queries
  - Persist understanding metadata

AST_PARSER:
  - Reuse existing Tree-sitter integration from PRP-05
  - Extract symbols, imports, dependencies

CLI:
  - /codebase command family
  - Onboarding flow for new codebases
  - Background watcher status display

AGENT_WORKFLOW:
  - verify_before_claim() hook in agent execution
  - check_for_duplicates() before code generation
  - get_relevant_context() in planning phase
  - Implementer uses understanding to avoid duplicates

CONFIG:
  - .agent-swarm/config.yaml for project settings
  - .agent-swarm/understanding/ for persisted understanding

WATCHER:
  - Background process/thread
  - Debounced change processing
  - Incremental updates only
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# After creating each Python file
ruff check src/understanding/ --fix
mypy src/understanding/ --strict
ruff format src/understanding/

# Validate imports
python -c "from src.understanding import CodebaseAnalyzer; print('OK')"
python -c "from src.understanding.knowledge import KnowledgeVerifier, DuplicateDetector; print('OK')"

# Expected: Zero errors, all imports resolve
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test individual components
pytest src/tests/understanding/test_analyzer.py -v --cov=src/understanding/analyzer
pytest src/tests/understanding/test_knowledge.py -v --cov=src/understanding/knowledge
pytest src/tests/understanding/test_watcher.py -v --cov=src/understanding/watcher
pytest src/tests/understanding/test_duplicates.py -v --cov=src/understanding/knowledge/duplicates

# Run all understanding tests
pytest src/tests/understanding/ -v --cov=src/understanding --cov-report=term-missing

# Expected: 80%+ coverage, all tests pass
```

### Level 3: Integration Testing (System Validation)

```bash
# Test on real codebase (agent-swarm itself)
python -c "
import asyncio
from src.understanding import CodebaseAnalyzer

async def test():
    analyzer = CodebaseAnalyzer('.')
    understanding = await analyzer.analyze_quick()
    print(f'Scanned {understanding.structure.total_files} files')
    print(f'Found {len(understanding.symbols)} symbols')

asyncio.run(test())
"
# Expected: Analyzes agent-swarm codebase successfully

# Test CLI integration
python cli.py
> /codebase analyze --mode quick
> /codebase find "workflow"
> /codebase deps src.core.workflow
> /codebase gaps
# Expected: All commands work

# Test watcher
python cli.py
> /codebase watch start
# In another terminal, modify a file
> /codebase watch status
# Expected: Shows pending changes detected

# Test verification
python -c "
import asyncio
from src.understanding import CodebaseAnalyzer
from src.understanding.knowledge import KnowledgeVerifier

async def test():
    analyzer = CodebaseAnalyzer('.')
    await analyzer.analyze()
    verifier = KnowledgeVerifier(analyzer.knowledge_store)
    
    # Should verify existing function
    result = await verifier.verify_symbol_exists('create_workflow')
    print(f'create_workflow exists: {result.verified}')
    
    # Should fail for non-existent function
    result = await verifier.verify_symbol_exists('nonexistent_function_xyz')
    print(f'nonexistent_function_xyz exists: {result.verified}')

asyncio.run(test())
"
# Expected: First True, second False

# Test duplicate detection
python -c "
import asyncio
from src.understanding import CodebaseAnalyzer
from src.understanding.knowledge import DuplicateDetector

async def test():
    analyzer = CodebaseAnalyzer('.')
    await analyzer.analyze()
    detector = DuplicateDetector(analyzer.knowledge_store)
    
    # Should find similar to existing
    dupes = await detector.check_before_create(
        proposed_name='create_new_workflow',
        proposed_description='Create a new LangGraph workflow for agent orchestration',
        proposed_kind='function',
    )
    
    for dupe in dupes:
        print(f'Similar: {dupe.existing_symbol} ({dupe.similarity_score:.2f})')
        print(f'Recommendation: {dupe.recommendation}')

asyncio.run(test())
"
# Expected: Should find create_workflow as similar
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Performance: Quick scan < 30 seconds
time python -c "
import asyncio
from src.understanding import CodebaseAnalyzer

async def test():
    analyzer = CodebaseAnalyzer('.')
    await analyzer.analyze_quick()

asyncio.run(test())
"
# Expected: < 30 seconds for agent-swarm

# Performance: Deep analysis < 5 minutes
time python -c "
import asyncio
from src.understanding import CodebaseAnalyzer

async def test():
    analyzer = CodebaseAnalyzer('.')
    await analyzer.analyze(mode='deep')

asyncio.run(test())
"
# Expected: < 5 minutes for agent-swarm

# Watcher latency test
# 1. Start watcher
# 2. Modify a file
# 3. Check understanding update time
# Expected: < 5 seconds

# Knowledge gap detection
python cli.py
> /codebase gaps
# Expected: Shows unanalyzed areas

# Anti-hallucination test
# Have agent make claims about codebase, verify all are correct
# Expected: Zero false claims about non-existent code

# Duplicate prevention test
# Ask agent to create function similar to existing one
# Expected: Agent suggests using existing function instead
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] Understanding tests achieve 80%+ coverage
- [ ] No linting errors: `ruff check src/understanding/`
- [ ] No type errors: `mypy src/understanding/`
- [ ] Quick analysis completes in <30 seconds
- [ ] Deep analysis completes in <5 minutes

### Feature Validation

- [ ] File scanner correctly filters ignored patterns
- [ ] Symbol extraction works for Python, JS, TS
- [ ] Dependency graph correctly maps imports
- [ ] Convention detection identifies naming patterns
- [ ] Background watcher detects file changes
- [ ] Incremental updates work without full re-analysis
- [ ] Understanding persists to .agent-swarm/understanding/
- [ ] CLI /codebase commands all functional

### Knowledge Verification Validation

- [ ] verify_symbol_exists() correctly identifies existing symbols
- [ ] verify_symbol_exists() correctly rejects non-existent symbols
- [ ] verify_function_signature() checks parameters accurately
- [ ] verify_file_exists() correctly validates paths
- [ ] Confidence levels accurately reflect knowledge certainty

### Duplicate Detection Validation

- [ ] Name similarity detection works
- [ ] Signature similarity detection works
- [ ] Semantic similarity detection works
- [ ] Recommendations are appropriate for similarity levels
- [ ] Agent integration prevents duplicate creation

### CLI Integration Validation

- [ ] New codebase detection works on first run
- [ ] Onboarding flow offers analysis options
- [ ] /codebase analyze works in quick and deep modes
- [ ] /codebase find returns relevant results
- [ ] /codebase gaps shows unknown areas
- [ ] /codebase watch manages watcher correctly

---

## Anti-Patterns to Avoid

- ❌ Don't load entire codebase into memory at once (use streaming/batching)
- ❌ Don't claim high confidence without AST verification
- ❌ Don't let agents make claims about unanalyzed code areas
- ❌ Don't process every file change immediately (use debouncing)
- ❌ Don't store binary files or generated code in understanding
- ❌ Don't ignore circular dependencies in graph building
- ❌ Don't assume file extensions always match language
- ❌ Don't skip content hashing (needed for change detection)
- ❌ Don't let watcher block main thread
- ❌ Don't forget to degrade confidence for stale knowledge

---

## Future Enhancements (Out of Scope for This PRP)

- Multi-repo understanding (monorepo support)
- Git history analysis (who changed what, when)
- Runtime analysis (tracing, profiling integration)
- Documentation generation from understanding
- Architecture visualization diagrams
- AI-powered code quality suggestions based on understanding
- Cross-project pattern learning
