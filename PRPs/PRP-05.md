# PRP-05: AST-Based Code Analysis Engine

## Goal

**Feature Goal**: Implement a comprehensive Abstract Syntax Tree (AST) based code analysis engine that provides deep code understanding capabilities including dependency mapping, complexity metrics, pattern recognition, and semantic code search across multiple programming languages.

**Deliverable**: Code analysis service with Tree-sitter integration, dependency graph generation, complexity calculation, pattern detection, and semantic search capabilities integrated with the agent workflow system.

**Success Definition**:
- Analyze Python, JavaScript, TypeScript, Java, and Go codebases accurately
- Generate complete dependency graphs in <500ms for 1000-line files
- Calculate complexity metrics (cyclomatic, cognitive, Halstead) with 95% accuracy
- Detect 20+ common code patterns and anti-patterns
- Semantic search returns relevant code snippets with 85%+ precision
- AST parsing handles syntax errors gracefully with partial analysis

## Why

- Current code tools only have placeholder implementations with no actual code understanding
- Agents cannot analyze existing codebases to understand structure and dependencies
- No way to calculate code complexity to guide refactoring decisions
- Missing pattern detection for identifying best practices and anti-patterns
- No semantic search capability to find similar code across projects
- Critical for enabling agents to understand and modify existing code intelligently

## What

Implement a powerful AST-based code analysis engine using Tree-sitter that provides agents with deep understanding of code structure, dependencies, complexity, and patterns, enabling intelligent code generation, refactoring, and optimization decisions.

### Success Criteria

- [ ] Tree-sitter parsers integrated for 5+ languages
- [ ] Dependency graphs generated with imports, calls, and references
- [ ] Complexity metrics calculated (cyclomatic, cognitive, Halstead)
- [ ] 20+ patterns detected (singleton, factory, observer, etc.)
- [ ] Semantic search with vector embeddings operational
- [ ] AST caching reduces parse time by 70% on repeated analysis

## All Needed Context

### Context Completeness Check

_If someone knew nothing about this codebase, would they have everything needed to implement this successfully?_ YES - This PRP includes complete Tree-sitter integration patterns, AST traversal algorithms, complexity calculation formulas, and pattern detection strategies.

### Documentation & References

```yaml
- url: https://tree-sitter.github.io/tree-sitter/using-parsers
  why: Tree-sitter Python bindings documentation for AST parsing
  critical: Shows language loading, parsing, and tree traversal patterns
  section: Using parsers with Python
  
- url: https://github.com/tree-sitter/py-tree-sitter
  why: Python Tree-sitter bindings with examples
  critical: Installation and language grammar loading patterns
  
- url: https://en.wikipedia.org/wiki/Cyclomatic_complexity
  why: Cyclomatic complexity calculation algorithm
  critical: Formula: M = E - N + 2P where E=edges, N=nodes, P=connected components
  
- url: https://www.sonarsource.com/resources/cognitive-complexity/
  why: Cognitive complexity calculation methodology
  critical: Increments for nesting, flow breaks, and logical operators
  
- file: src/tools/implementations/code_tools.py
  why: Existing code tools that will use AST analysis
  pattern: BaseTool inheritance, tool schema definition
  gotcha: All methods must be async, results must be JSON-serializable
  
- file: src/agents/implementer.py
  why: Primary agent that will use code analysis
  pattern: Agent structure, tool service integration
  gotcha: State updates must be immutable
  
- file: src/services/llm_service.py
  why: Pattern for high-level service orchestration
  pattern: Service facade, async context managers
  gotcha: Error handling with fallbacks
  
- url: https://github.com/github/semantic
  why: Reference implementation for semantic code analysis
  critical: Shows AST diffing and semantic search patterns
```

### Current Codebase Tree

```bash
agent-swarm/
├── src/
│   ├── tools/
│   │   ├── implementations/
│   │   │   ├── code_tools.py      # Has placeholder code generation
│   │   │   ├── file_tools.py      # File operations
│   │   │   └── validation_tools.py # Linting tools
│   │   ├── base.py                # BaseTool class
│   │   └── executor.py            # Tool execution
│   ├── agents/
│   │   ├── implementer.py         # Needs code analysis
│   │   ├── validator.py           # Needs complexity metrics
│   │   └── debugger.py           # Needs dependency analysis
│   ├── services/
│   │   └── tool_service.py        # Tool orchestration
│   └── models/
│       └── tool_models.py         # Tool-related models
```

### Desired Codebase Tree with Files to be Added

```bash
agent-swarm/
├── src/
│   ├── analysis/                      # NEW: Code analysis subsystem
│   │   ├── __init__.py                # Export main interfaces
│   │   ├── base.py                    # BaseAnalyzer abstract class
│   │   ├── ast_parser.py              # Tree-sitter AST parsing engine
│   │   ├── dependency_analyzer.py     # Dependency graph generation
│   │   ├── complexity_analyzer.py     # Complexity metrics calculation
│   │   ├── pattern_detector.py        # Code pattern recognition
│   │   ├── semantic_search.py         # Semantic code search
│   │   ├── cache.py                   # AST caching layer
│   │   └── languages/                 # Language-specific analyzers
│   │       ├── __init__.py
│   │       ├── python_analyzer.py     # Python-specific analysis
│   │       ├── javascript_analyzer.py # JavaScript/TypeScript analysis
│   │       ├── java_analyzer.py       # Java analysis
│   │       └── go_analyzer.py         # Go analysis
│   ├── models/
│   │   └── analysis_models.py        # NEW: Analysis-related models
│   ├── services/
│   │   └── analysis_service.py       # NEW: High-level analysis service
│   ├── tools/implementations/
│   │   └── analysis_tools.py         # NEW: Analysis tools for agents
│   └── tests/
│       └── analysis/                  # NEW: Analysis tests
│           ├── test_ast_parser.py
│           ├── test_dependency.py
│           ├── test_complexity.py
│           ├── test_patterns.py
│           └── test_semantic_search.py
├── grammars/                          # NEW: Tree-sitter grammars
│   ├── tree-sitter-python.wasm
│   ├── tree-sitter-javascript.wasm
│   ├── tree-sitter-typescript.wasm
│   ├── tree-sitter-java.wasm
│   └── tree-sitter-go.wasm
```

### Known Gotchas of our Codebase & Library Quirks

```python
# CRITICAL: Tree-sitter requires compiled language grammars (.so or .wasm files)
# Must download and compile grammars during setup

# CRITICAL: AST nodes are not JSON-serializable directly
# Must convert to dictionaries before returning from tools

# CRITICAL: Tree-sitter uses byte offsets, not character offsets
# Important for position tracking in multi-byte character encodings

# CRITICAL: Large files (>10MB) can cause memory issues
# Implement streaming/chunking for large file analysis

# CRITICAL: Semantic search requires embeddings generation
# Integrate with existing LLM service for embeddings

# CRITICAL: Pattern detection can have false positives
# Use confidence scores and thresholds

# CRITICAL: Cyclomatic complexity calculation differs by language
# Language-specific control flow must be considered

# CRITICAL: AST caching keys must include file hash
# File changes must invalidate cache entries
```

## Implementation Blueprint

### Data Models and Structure

```python
# src/models/analysis_models.py
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime

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
    signature: Optional[str] = Field(default=None, description="Function/method signature")
    docstring: Optional[str] = Field(default=None, description="Documentation string")
    complexity: Dict[ComplexityType, float] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list, description="Direct dependencies")
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
        description="Halstead metrics (volume, difficulty, effort)"
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
    recommendation: Optional[str] = Field(default=None, description="Improvement suggestion")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AnalysisRequest(BaseModel):
    """Request for code analysis."""
    file_paths: List[str] = Field(description="Files to analyze")
    language: Optional[Language] = Field(default=None, description="Force language detection")
    analysis_types: List[str] = Field(
        default_factory=lambda: ["ast", "dependencies", "complexity", "patterns"],
        description="Types of analysis to perform"
    )
    max_depth: int = Field(default=10, description="Max recursion depth")
    include_imports: bool = Field(default=True, description="Analyze imported modules")
    cache_enabled: bool = Field(default=True, description="Use cached ASTs")

class AnalysisResult(BaseModel):
    """Result of code analysis."""
    file_path: str
    language: Language
    ast: Optional[ASTNode] = Field(default=None, description="Abstract syntax tree")
    entities: List[CodeEntity] = Field(default_factory=list, description="Code entities found")
    dependencies: Optional[DependencyGraph] = Field(default=None)
    complexity: Optional[ComplexityMetrics] = Field(default=None)
    patterns: List[Pattern] = Field(default_factory=list, description="Detected patterns")
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
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: INSTALL Tree-sitter and download language grammars
  - IMPLEMENT: Setup script for Tree-sitter and grammars
  - EXECUTE: pip install tree-sitter
  - DOWNLOAD: Language .wasm files from Tree-sitter repos
  - COMPILE: Build language libraries if needed
  - PLACEMENT: grammars/ directory in project root

Task 2: CREATE src/analysis/base.py
  - IMPLEMENT: BaseAnalyzer abstract class
  - FOLLOW pattern: src/tools/base.py abstract pattern
  - NAMING: BaseAnalyzer, analyze, get_language methods
  - PLACEMENT: Analysis subsystem base module

Task 3: CREATE src/analysis/ast_parser.py
  - IMPLEMENT: ASTParser class with Tree-sitter integration
  - FOLLOW pattern: Tree-sitter Python documentation
  - NAMING: ASTParser, parse_file, parse_code, traverse_tree methods
  - DEPENDENCIES: tree-sitter, language grammars
  - PLACEMENT: Core AST parsing module

Task 4: CREATE src/analysis/languages/python_analyzer.py
  - IMPLEMENT: PythonAnalyzer for Python-specific analysis
  - FOLLOW pattern: BaseAnalyzer inheritance
  - NAMING: PythonAnalyzer, extract_functions, extract_classes methods
  - DEPENDENCIES: ASTParser, Python grammar
  - PLACEMENT: Language-specific analyzer

Task 5: CREATE src/analysis/dependency_analyzer.py
  - IMPLEMENT: DependencyAnalyzer for graph generation
  - FOLLOW pattern: Graph traversal algorithms
  - NAMING: DependencyAnalyzer, build_graph, find_cycles methods
  - DEPENDENCIES: AST parser, language analyzers
  - PLACEMENT: Analysis subsystem

Task 6: CREATE src/analysis/complexity_analyzer.py
  - IMPLEMENT: ComplexityAnalyzer for metrics calculation
  - FOLLOW pattern: McCabe and Halstead algorithms
  - NAMING: ComplexityAnalyzer, calculate_cyclomatic, calculate_cognitive methods
  - DEPENDENCIES: AST parser
  - PLACEMENT: Analysis subsystem

Task 7: CREATE src/analysis/pattern_detector.py
  - IMPLEMENT: PatternDetector for pattern recognition
  - FOLLOW pattern: Visitor pattern for AST traversal
  - NAMING: PatternDetector, detect_patterns, register_pattern methods
  - DEPENDENCIES: AST parser, pattern definitions
  - PLACEMENT: Analysis subsystem

Task 8: CREATE src/analysis/semantic_search.py
  - IMPLEMENT: SemanticSearch with embeddings
  - FOLLOW pattern: Vector similarity search
  - NAMING: SemanticSearch, index_code, search methods
  - DEPENDENCIES: LLM service for embeddings, vector store
  - PLACEMENT: Analysis subsystem

Task 9: CREATE src/analysis/cache.py
  - IMPLEMENT: ASTCache for caching parsed trees
  - FOLLOW pattern: src/llm/cache.py caching pattern
  - NAMING: ASTCache, get_cached_ast, store_ast methods
  - DEPENDENCIES: Redis, file hashing
  - PLACEMENT: Analysis subsystem

Task 10: CREATE src/services/analysis_service.py
  - IMPLEMENT: AnalysisOrchestrator high-level service
  - FOLLOW pattern: src/services/llm_service.py facade pattern
  - NAMING: AnalysisOrchestrator, analyze_codebase, search_code methods
  - DEPENDENCIES: All analysis components
  - PLACEMENT: Services layer

Task 11: CREATE src/tools/implementations/analysis_tools.py
  - IMPLEMENT: Analysis tools for agent usage
  - FOLLOW pattern: src/tools/base.py tool pattern
  - NAMING: AnalyzeCodeTool, FindDependenciesTool, CalculateComplexityTool
  - DEPENDENCIES: Analysis service
  - PLACEMENT: Tool implementations

Task 12: MODIFY src/agents/implementer.py
  - INTEGRATE: Add code analysis before implementation
  - FIND pattern: execute method, _implement_code placeholder
  - ADD: Call to analyze existing code before generating new code
  - PRESERVE: Existing state management

Task 13: CREATE src/tests/analysis/test_ast_parser.py
  - IMPLEMENT: Unit tests for AST parsing
  - FOLLOW pattern: pytest-asyncio with fixtures
  - COVERAGE: Multiple languages, syntax errors, large files
  - PLACEMENT: Analysis test directory

Task 14: CREATE src/tests/analysis/test_complexity.py
  - IMPLEMENT: Tests for complexity calculations
  - FOLLOW pattern: Known complexity examples
  - COVERAGE: Verify against manual calculations
  - PLACEMENT: Analysis test directory

Task 15: CREATE src/tests/analysis/test_patterns.py
  - IMPLEMENT: Tests for pattern detection
  - FOLLOW pattern: Known pattern examples
  - COVERAGE: True positives, false positives, confidence scores
  - PLACEMENT: Analysis test directory
```

### Implementation Patterns & Key Details

```python
# AST parsing pattern
import tree_sitter
from tree_sitter import Language, Parser

class ASTParser:
    """
    PATTERN: Tree-sitter integration with caching
    CRITICAL: Load languages once, reuse parser
    """
    
    def __init__(self):
        self.parser = Parser()
        self.languages = {}
        self._load_languages()
    
    def _load_languages(self):
        """Load Tree-sitter language grammars."""
        # Load compiled language libraries
        for lang_name, lib_path in self.LANGUAGE_LIBS.items():
            language = Language(lib_path, lang_name)
            self.languages[lang_name] = language
    
    async def parse_file(self, file_path: str, language: str) -> ASTNode:
        """
        Parse file into AST.
        GOTCHA: Tree-sitter uses bytes, not strings
        """
        with open(file_path, "rb") as f:
            content = f.read()
        
        self.parser.set_language(self.languages[language])
        tree = self.parser.parse(content)
        
        return self._convert_tree_to_ast(tree.root_node, content)
    
    def _convert_tree_to_ast(self, node, source_code: bytes) -> ASTNode:
        """
        Convert Tree-sitter node to our AST model.
        CRITICAL: Must be recursive but watch depth
        """
        return ASTNode(
            node_type=node.type,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_point=(node.start_point[0], node.start_point[1]),
            end_point=(node.end_point[0], node.end_point[1]),
            text=source_code[node.start_byte:node.end_byte].decode('utf-8'),
            children=[
                self._convert_tree_to_ast(child, source_code)
                for child in node.children
            ]
        )

# Complexity calculation pattern
class ComplexityAnalyzer:
    """
    PATTERN: Visitor pattern for AST traversal
    GOTCHA: Different languages have different control flow
    """
    
    def calculate_cyclomatic(self, ast: ASTNode) -> int:
        """
        Calculate McCabe cyclomatic complexity.
        Formula: M = E - N + 2P
        """
        edges = 0
        nodes = 1  # Start with 1 for entry point
        
        def visit(node: ASTNode):
            nonlocal edges, nodes
            
            # Count decision points
            if node.node_type in ["if", "while", "for", "case"]:
                edges += 2  # True and false branches
                nodes += 1
            elif node.node_type in ["and", "or"]:
                edges += 1  # Short-circuit evaluation
            
            # Recurse through children
            for child in node.children:
                visit(child)
        
        visit(ast)
        return edges - nodes + 2  # 2 for connected component
    
    def calculate_cognitive(self, ast: ASTNode, depth: int = 0) -> int:
        """
        Calculate cognitive complexity.
        PATTERN: Increment for nesting and flow breaks
        """
        complexity = 0
        
        for node in ast.children:
            if node.node_type in ["if", "switch", "for", "while"]:
                complexity += 1 + depth  # Base + nesting penalty
                complexity += self.calculate_cognitive(node, depth + 1)
            elif node.node_type in ["catch", "else", "elif"]:
                complexity += 1
                complexity += self.calculate_cognitive(node, depth)
            else:
                complexity += self.calculate_cognitive(node, depth)
        
        return complexity

# Pattern detection pattern
class PatternDetector:
    """
    PATTERN: Registered pattern matchers
    CRITICAL: Balance between accuracy and performance
    """
    
    def __init__(self):
        self.patterns = {}
        self._register_patterns()
    
    def _register_patterns(self):
        """Register pattern detection functions."""
        self.patterns["singleton"] = self._detect_singleton
        self.patterns["factory"] = self._detect_factory
        self.patterns["god_class"] = self._detect_god_class
        # ... more patterns
    
    async def detect_patterns(
        self, ast: ASTNode, entity: CodeEntity
    ) -> List[Pattern]:
        """Detect all patterns in code."""
        detected = []
        
        for pattern_name, detector in self.patterns.items():
            result = await detector(ast, entity)
            if result and result.confidence > 0.7:
                detected.append(result)
        
        return detected
    
    async def _detect_singleton(
        self, ast: ASTNode, entity: CodeEntity
    ) -> Optional[Pattern]:
        """
        Detect singleton pattern.
        PATTERN: Private constructor + static instance
        """
        has_private_constructor = False
        has_static_instance = False
        
        # Check for singleton characteristics
        for node in ast.children:
            if node.node_type == "constructor" and "private" in node.text:
                has_private_constructor = True
            if node.node_type == "field" and "static" in node.text:
                has_static_instance = True
        
        if has_private_constructor and has_static_instance:
            return Pattern(
                name="Singleton",
                type=PatternType.DESIGN_PATTERN,
                confidence=0.9,
                location=entity,
                description="Singleton pattern detected",
                recommendation="Consider if singleton is necessary"
            )
        
        return None

# Semantic search pattern
class SemanticSearch:
    """
    PATTERN: Code embeddings with vector similarity
    GOTCHA: Requires LLM service for embeddings
    """
    
    def __init__(self, llm_service, vector_store):
        self.llm_service = llm_service
        self.vector_store = vector_store
    
    async def index_code(self, entities: List[CodeEntity]):
        """
        Index code entities for semantic search.
        CRITICAL: Batch embeddings for efficiency
        """
        # Prepare code snippets for embedding
        snippets = [
            f"{e.type}: {e.name}\n{e.signature}\n{e.docstring or ''}"
            for e in entities
        ]
        
        # Generate embeddings (batch for efficiency)
        embeddings = await self.llm_service.generate_embeddings(snippets)
        
        # Store in vector database
        for entity, embedding in zip(entities, embeddings):
            await self.vector_store.add(
                id=f"{entity.file_path}:{entity.name}",
                embedding=embedding,
                metadata=entity.dict()
            )
    
    async def search(
        self, query: str, max_results: int = 10
    ) -> List[SemanticSearchResult]:
        """
        Semantic search for code.
        PATTERN: Embed query, find similar vectors
        """
        # Generate query embedding
        query_embedding = await self.llm_service.generate_embedding(query)
        
        # Search vector store
        results = await self.vector_store.search(
            embedding=query_embedding,
            limit=max_results
        )
        
        return [
            SemanticSearchResult(
                code_snippet=r.metadata["text"],
                file_path=r.metadata["file_path"],
                line_start=r.metadata["line_start"],
                line_end=r.metadata["line_end"],
                similarity_score=r.score,
                entity=CodeEntity(**r.metadata)
            )
            for r in results
        ]
```

### Integration Points

```yaml
TREE_SITTER:
  - dependency: "tree-sitter>=0.20.0"
  - grammars:
    - python: "tree-sitter-python==0.20.4"
    - javascript: "tree-sitter-javascript==0.20.1"
    - typescript: "tree-sitter-typescript==0.20.3"
    - java: "tree-sitter-java==0.20.2"
    - go: "tree-sitter-go==0.20.0"

CONFIG:
  - add to: .env
  - variables: |
      # Analysis Configuration
      ANALYSIS_CACHE_ENABLED=true
      ANALYSIS_CACHE_TTL=3600
      ANALYSIS_MAX_FILE_SIZE_MB=10
      ANALYSIS_MAX_DEPTH=10
      
      # Complexity Thresholds
      COMPLEXITY_HIGH_THRESHOLD=10
      COMPLEXITY_VERY_HIGH_THRESHOLD=20
      
      # Pattern Detection
      PATTERN_CONFIDENCE_THRESHOLD=0.7
      PATTERN_MAX_PATTERNS_PER_FILE=50
      
      # Semantic Search
      SEMANTIC_SEARCH_MODEL=text-embedding-ada-002
      SEMANTIC_SEARCH_CACHE_SIZE=10000

REDIS:
  - cache_keys: |
      ast:{file_hash} - Cached AST
      complexity:{file_hash} - Cached complexity metrics
      patterns:{file_hash} - Cached patterns
      embeddings:{entity_id} - Code embeddings

INTEGRATION:
  - llm_service: "Use for code embeddings generation"
  - memory_service: "Store analysis results in project memory"
  - tool_service: "Register analysis tools"
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# Check new analysis module
ruff check src/analysis/ --fix
mypy src/analysis/ --strict
ruff format src/analysis/

# Verify imports
python -c "from src.analysis import ASTParser; print('Analysis imports OK')"
python -c "from src.services.analysis_service import AnalysisOrchestrator; print('Service imports OK')"

# Expected: Zero errors, all imports resolve
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test AST parsing
pytest src/tests/analysis/test_ast_parser.py -v --cov=src/analysis/ast_parser

# Test complexity calculations
pytest src/tests/analysis/test_complexity.py -v --cov=src/analysis/complexity_analyzer

# Test pattern detection
pytest src/tests/analysis/test_patterns.py -v --cov=src/analysis/pattern_detector

# Test dependency analysis
pytest src/tests/analysis/test_dependency.py -v --cov=src/analysis/dependency_analyzer

# Full analysis test suite
pytest src/tests/analysis/ -v --cov=src/analysis --cov-report=term-missing

# Expected: 90%+ coverage, all tests pass
```

### Level 3: Integration Testing (System Validation)

```bash
# Test Tree-sitter setup
python scripts/test_tree_sitter_setup.py \
  --verify-languages python javascript typescript java go
# Expected: All languages load successfully

# Test AST parsing on real files
python scripts/test_ast_parsing.py \
  --file src/agents/implementer.py \
  --language python
# Expected: Complete AST generated, no parse errors

# Test dependency graph generation
python scripts/test_dependency_graph.py \
  --directory src/ \
  --output-format json
# Expected: Dependency graph with imports and calls

# Test complexity on known examples
python scripts/test_complexity_examples.py \
  --verify-against-manual
# Expected: Complexity matches manual calculations

# Test pattern detection
python scripts/test_pattern_detection.py \
  --directory src/ \
  --patterns singleton factory observer
# Expected: Patterns detected with confidence scores

# Test semantic search
python scripts/test_semantic_search.py \
  --index-directory src/ \
  --query "function that handles memory storage" \
  --expected-results 5
# Expected: Relevant code snippets returned
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Large Codebase Performance Test
python scripts/benchmark_analysis.py \
  --repository https://github.com/langchain-ai/langchain \
  --measure-time \
  --measure-memory
# Expected: <500ms per 1000 lines for parsing

# Multi-Language Analysis
python scripts/test_multilanguage.py \
  --analyze-project ./test-projects/polyglot \
  --languages python javascript java \
  --generate-report
# Expected: All languages analyzed correctly

# Syntax Error Resilience
python scripts/test_error_recovery.py \
  --inject-syntax-errors \
  --verify-partial-analysis
# Expected: Partial AST generated despite errors

# Pattern Detection Accuracy
python scripts/test_pattern_accuracy.py \
  --test-set ./pattern-examples/ \
  --calculate-precision-recall
# Expected: 85%+ precision, 75%+ recall

# Complexity Correlation Test
python scripts/test_complexity_correlation.py \
  --compare-with-sonarqube \
  --calculate-correlation
# Expected: >0.8 correlation with SonarQube metrics

# Cache Performance Test
python scripts/test_cache_performance.py \
  --files 100 \
  --iterations 3 \
  --measure-speedup
# Expected: 70%+ speedup on cached analysis

# Semantic Search Relevance
python scripts/test_search_relevance.py \
  --queries ./test-queries.json \
  --measure-precision-at-k 5
# Expected: 85%+ precision@5

# Agent Integration Test
python scripts/test_agent_with_analysis.py \
  --agent implementer \
  --task "Refactor complex function" \
  --verify-analysis-usage
# Expected: Agent uses complexity metrics to guide refactoring

# Memory Usage Test
python scripts/test_memory_usage.py \
  --large-file ./test-files/10mb_file.py \
  --monitor-memory \
  --verify-limits
# Expected: Memory usage stays under 500MB
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] Analysis tests achieve 90%+ coverage: `pytest src/tests/analysis/ --cov=src/analysis`
- [ ] No linting errors: `ruff check src/analysis/`
- [ ] No type errors: `mypy src/analysis/ --strict`
- [ ] Tree-sitter grammars installed and working

### Feature Validation

- [ ] 5+ languages supported (Python, JS, TS, Java, Go)
- [ ] Dependency graphs generated in <500ms for 1000 lines
- [ ] Complexity metrics calculated with 95% accuracy
- [ ] 20+ patterns detected successfully
- [ ] Semantic search returns relevant results with 85%+ precision
- [ ] Syntax errors handled gracefully with partial analysis

### Code Quality Validation

- [ ] Follows existing service and tool patterns
- [ ] All analysis operations async-compatible
- [ ] AST nodes properly converted to JSON-serializable format
- [ ] Caching reduces parse time by 70%+
- [ ] Memory usage controlled for large files
- [ ] Pattern detection uses confidence thresholds

### Documentation & Deployment

- [ ] Tree-sitter setup documented
- [ ] Grammar installation automated
- [ ] Complexity thresholds configurable
- [ ] Pattern definitions documented
- [ ] API endpoints for analysis service

---

## Anti-Patterns to Avoid

- ❌ Don't store raw Tree-sitter nodes (not serializable)
- ❌ Don't parse without language detection
- ❌ Don't ignore syntax errors (provide partial analysis)
- ❌ Don't analyze files larger than configured limit
- ❌ Don't skip caching for repeated analysis
- ❌ Don't use synchronous file operations
- ❌ Don't calculate all metrics if not requested
- ❌ Don't ignore encoding issues (use proper decoding)
- ❌ Don't traverse AST without depth limits
- ❌ Don't generate embeddings for every code snippet (batch operations)