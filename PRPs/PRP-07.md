# PRP-07: Context-Aware RAG Implementation

## Goal

**Feature Goal**: Implement an advanced context-aware Retrieval-Augmented Generation (RAG) system that intelligently chunks documents and code, performs semantic and structural analysis, provides hybrid search with dynamic relevance scoring, and optimizes context window usage for the agent workflow system.

**Deliverable**: Enhanced RAG service with document ingestion pipeline, code-aware chunking strategies, improved hybrid search (vector + keyword + structural), dynamic relevance scoring, context window optimization, and integration with the existing memory, analysis, and LLM systems.

**Success Definition**:
- Document ingestion processing 100+ pages in <30 seconds
- Code-aware chunking preserves 95%+ syntactic integrity
- Hybrid search achieves 90%+ precision@5 for relevant queries
- Dynamic relevance scoring adapts to query types (code/docs/QA)
- Context window optimization reduces token usage by 40%
- Support for 10+ document formats (.md, .py, .js, .txt, .pdf, .json, .yaml, .rst, .ipynb, .html)
- Streaming ingestion for large documents (>10MB)

## Why

- Current RAG service is basic with simple context building and no advanced chunking
- No document ingestion pipeline for knowledge bases and documentation
- Missing code-aware chunking that preserves function/class boundaries
- Simple hybrid search without structural or semantic understanding
- No dynamic relevance scoring based on query intent
- Context window not optimized, leading to token waste
- Critical for agents to access and understand large codebases and documentation

## What

Implement a comprehensive context-aware RAG system that intelligently processes documents and code, understands structure and semantics, provides multi-modal search capabilities, dynamically scores relevance based on query intent, and optimizes context window usage for maximum efficiency.

### Success Criteria

- [ ] Document ingestion handles 10+ formats efficiently
- [ ] Code chunking preserves syntactic boundaries (functions, classes)
- [ ] Hybrid search combines vector, keyword, and structural signals
- [ ] Relevance scoring adapts to query types automatically
- [ ] Context optimization reduces token usage by 40%+
- [ ] Streaming support for large document processing

## All Needed Context

### Context Completeness Check

_If someone knew nothing about this codebase, would they have everything needed to implement this successfully?_ YES - This PRP includes complete RAG enhancement patterns, chunking strategies, document processing pipelines, and integration with existing systems.

### Documentation & References

```yaml
- url: https://docs.llamaindex.ai/en/stable/understanding/indexing/indexing.html
  why: Advanced document chunking and indexing strategies
  critical: Shows semantic chunking, sliding window, and sentence-based splitting
  section: Document chunking strategies

- url: https://www.pinecone.io/learn/chunking-strategies/
  why: Comprehensive guide to different chunking strategies
  critical: Comparison of fixed-size, semantic, and structure-aware chunking
  
- url: https://github.com/langchain-ai/langchain/blob/main/docs/docs/modules/data_connection/document_loaders/
  why: Document loader patterns for various file formats
  critical: Shows loaders for PDF, HTML, Markdown, code files
  
- url: https://arxiv.org/abs/2307.03172
  why: Lost in the Middle paper - context window optimization
  critical: Shows importance of relevance ordering in context window

- file: src/services/rag_service.py
  why: Existing RAG service to enhance
  pattern: RAGService class, generate_with_context method
  gotcha: Currently uses simple context building

- file: src/memory/hybrid.py
  why: Existing hybrid search implementation
  pattern: HybridSearchManager, BM25KeywordSearch
  gotcha: Need to enhance with structural search

- file: src/analysis/ast_parser.py
  why: AST parser for code-aware chunking
  pattern: ASTParser for extracting code structure
  gotcha: Already handles multiple languages

- file: src/services/llm_service.py
  why: LLM service for embeddings and generation
  pattern: LLMOrchestrator for model routing
  gotcha: Must use appropriate models for embeddings

- file: src/decomposition/fractal_decomposer.py
  why: Task decomposition that will use RAG context
  pattern: Uses context for decomposition decisions
  gotcha: Needs relevant documentation/code context
```

### Current Codebase Tree

```bash
agent-swarm/
├── src/
│   ├── services/
│   │   ├── rag_service.py          # Basic RAG implementation
│   │   ├── memory_service.py       # Memory orchestration
│   │   └── llm_service.py          # LLM integration
│   ├── memory/
│   │   ├── hybrid.py               # Hybrid search (vector + keyword)
│   │   ├── project.py              # Vector store integration
│   │   └── cache.py                # Response caching
│   ├── analysis/
│   │   ├── ast_parser.py           # AST parsing for code
│   │   ├── semantic_search.py      # Semantic code search
│   │   └── complexity_analyzer.py  # Code complexity metrics
│   └── models/
│       └── memory_models.py        # Memory data models
```

### Desired Codebase Tree with Files to be Added

```bash
agent-swarm/
├── src/
│   ├── rag/                              # NEW: Enhanced RAG subsystem
│   │   ├── __init__.py                   # Export main interfaces
│   │   ├── base.py                       # BaseRAG abstract class
│   │   ├── ingestion/                    # NEW: Document ingestion
│   │   │   ├── __init__.py
│   │   │   ├── document_loader.py        # Multi-format document loading
│   │   │   ├── preprocessor.py           # Document preprocessing
│   │   │   └── loaders/                  # Format-specific loaders
│   │   │       ├── markdown_loader.py    # Markdown documents
│   │   │       ├── code_loader.py        # Source code files
│   │   │       ├── pdf_loader.py         # PDF documents
│   │   │       ├── notebook_loader.py    # Jupyter notebooks
│   │   │       └── json_loader.py        # JSON/YAML files
│   │   ├── chunking/                     # NEW: Chunking strategies
│   │   │   ├── __init__.py
│   │   │   ├── base_chunker.py           # Abstract chunker
│   │   │   ├── semantic_chunker.py       # Semantic-based chunking
│   │   │   ├── code_chunker.py           # AST-aware code chunking
│   │   │   ├── markdown_chunker.py       # Structure-aware markdown
│   │   │   └── sliding_window_chunker.py # Overlapping chunks
│   │   ├── retrieval/                    # NEW: Advanced retrieval
│   │   │   ├── __init__.py
│   │   │   ├── query_analyzer.py         # Query intent analysis
│   │   │   ├── relevance_scorer.py       # Dynamic relevance scoring
│   │   │   ├── structural_search.py      # Code structure search
│   │   │   └── reranker.py               # Result re-ranking
│   │   ├── context/                      # NEW: Context optimization
│   │   │   ├── __init__.py
│   │   │   ├── context_builder.py        # Optimized context building
│   │   │   ├── context_compressor.py     # Context compression
│   │   │   └── window_optimizer.py       # Token window optimization
│   │   └── pipeline.py                   # RAG pipeline orchestration
│   ├── models/
│   │   ├── rag_models.py                 # NEW: RAG-specific models
│   │   └── document_models.py            # NEW: Document/chunk models
│   ├── services/
│   │   ├── rag_service.py                # MODIFY: Enhance with new RAG
│   │   └── document_service.py           # NEW: Document management service
│   └── tests/
│       └── rag/                          # NEW: RAG tests
│           ├── test_ingestion.py
│           ├── test_chunking.py
│           ├── test_retrieval.py
│           └── test_context.py
```

### Known Gotchas of our Codebase & Library Quirks

```python
# CRITICAL: Code chunks must preserve syntactic validity
# Don't split in the middle of functions or classes

# CRITICAL: Embedding model consistency
# Must use same model for indexing and querying

# CRITICAL: Context window ordering matters
# Most relevant content should be at beginning and end (U-shaped attention)

# CRITICAL: Chunk overlap for context preservation
# 10-20% overlap prevents information loss at boundaries

# CRITICAL: Document metadata must be preserved
# Source, page, section information needed for citations

# CRITICAL: Async processing for large documents
# Use streaming/chunking to prevent memory issues

# CRITICAL: Vector dimension consistency
# All embeddings must have same dimensions for search

# CRITICAL: Token counting accuracy
# Different models count tokens differently (GPT vs Claude)
```

## Implementation Blueprint

### Data Models and Structure

```python
# src/models/document_models.py
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal
from enum import Enum
from datetime import datetime

class DocumentType(str, Enum):
    """Supported document types."""
    MARKDOWN = "markdown"
    CODE = "code"
    PDF = "pdf"
    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    HTML = "html"
    NOTEBOOK = "notebook"
    RESTRUCTURED_TEXT = "rst"

class ChunkingStrategy(str, Enum):
    """Chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    SLIDING_WINDOW = "sliding_window"
    CODE_AWARE = "code_aware"

class Document(BaseModel):
    """Represents a document for ingestion."""
    id: str = Field(description="Unique document ID")
    content: str = Field(description="Document content")
    type: DocumentType = Field(description="Document type")
    source: str = Field(description="Document source (path/URL)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    created_at: datetime = Field(default_factory=datetime.now)
    processed: bool = Field(default=False)
    chunks: List["Chunk"] = Field(default_factory=list, description="Document chunks")
    embeddings_generated: bool = Field(default=False)
    
class Chunk(BaseModel):
    """Represents a document chunk."""
    id: str = Field(description="Unique chunk ID")
    document_id: str = Field(description="Parent document ID")
    content: str = Field(description="Chunk content")
    start_index: int = Field(description="Start position in document")
    end_index: int = Field(description="End position in document")
    chunk_index: int = Field(description="Chunk number in document")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_type: str = Field(default="text", description="Type of chunk (text/code/header)")
    language: Optional[str] = Field(default=None, description="Programming language if code")
    
    # Structural information
    section: Optional[str] = Field(default=None, description="Document section")
    subsection: Optional[str] = Field(default=None, description="Document subsection")
    page_number: Optional[int] = Field(default=None, description="Page number if applicable")
    
    # Code-specific
    function_name: Optional[str] = Field(default=None, description="Function name if code")
    class_name: Optional[str] = Field(default=None, description="Class name if code")
    
    # Embeddings and search
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    token_count: int = Field(default=0, description="Number of tokens")

# src/models/rag_models.py
class QueryIntent(str, Enum):
    """Query intent types for relevance scoring."""
    CODE_SEARCH = "code_search"
    DOCUMENTATION = "documentation"
    QUESTION_ANSWER = "question_answer"
    DEBUGGING = "debugging"
    EXAMPLE_SEARCH = "example_search"
    DEFINITION = "definition"
    EXPLANATION = "explanation"

class QueryAnalysis(BaseModel):
    """Analysis of a search query."""
    query: str = Field(description="Original query")
    intent: QueryIntent = Field(description="Detected query intent")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    entities: List[str] = Field(default_factory=list, description="Named entities")
    programming_terms: List[str] = Field(default_factory=list, description="Programming-specific terms")
    requires_code: bool = Field(default=False, description="Whether query needs code context")
    requires_docs: bool = Field(default=False, description="Whether query needs documentation")
    confidence: float = Field(default=0.5, ge=0, le=1, description="Intent detection confidence")

class RetrievalRequest(BaseModel):
    """Request for document retrieval."""
    query: str = Field(description="Search query")
    k: int = Field(default=5, ge=1, le=100, description="Number of results")
    document_types: Optional[List[DocumentType]] = Field(default=None, description="Filter by document type")
    chunking_strategy: Optional[ChunkingStrategy] = Field(default=None, description="Preferred chunking")
    use_hybrid: bool = Field(default=True, description="Use hybrid search")
    use_reranking: bool = Field(default=True, description="Apply re-ranking")
    max_tokens: int = Field(default=4000, description="Maximum context tokens")
    include_metadata: bool = Field(default=True, description="Include chunk metadata")

class RetrievalResult(BaseModel):
    """Result from document retrieval."""
    chunk: Chunk = Field(description="Retrieved chunk")
    score: float = Field(ge=0, le=1, description="Relevance score")
    search_type: str = Field(description="Search method used")
    rerank_score: Optional[float] = Field(default=None, description="Re-ranking score")
    highlights: List[str] = Field(default_factory=list, description="Highlighted excerpts")
    
class IngestionRequest(BaseModel):
    """Request for document ingestion."""
    source_path: str = Field(description="Path to document or directory")
    document_type: Optional[DocumentType] = Field(default=None, description="Force document type")
    chunking_strategy: ChunkingStrategy = Field(default=ChunkingStrategy.SEMANTIC)
    chunk_size: int = Field(default=500, description="Target chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks in tokens")
    generate_embeddings: bool = Field(default=True, description="Generate embeddings immediately")
    extract_metadata: bool = Field(default=True, description="Extract document metadata")
    recursive: bool = Field(default=False, description="Process directories recursively")

class ContextOptimization(BaseModel):
    """Context window optimization settings."""
    max_tokens: int = Field(default=4000, description="Maximum context window tokens")
    optimization_strategy: Literal["truncate", "summarize", "compress"] = Field(default="compress")
    relevance_ordering: Literal["score", "diversity", "recency"] = Field(default="score")
    include_system_prompt: bool = Field(default=True)
    reserve_tokens: int = Field(default=500, description="Tokens reserved for response")
    use_sliding_window: bool = Field(default=False, description="Use sliding window for long contexts")
```

### Implementation Tasks (ordered by dependencies)

```yaml
Task 1: CREATE src/rag/base.py
  - IMPLEMENT: BaseRAG abstract class defining RAG interface
  - FOLLOW pattern: Abstract base class pattern
  - NAMING: BaseRAG, ingest, retrieve, generate methods
  - PLACEMENT: RAG subsystem base module

Task 2: CREATE src/models/document_models.py
  - IMPLEMENT: Document, Chunk, and related models
  - FOLLOW pattern: Pydantic models with validation
  - NAMING: Models as specified in data models section
  - PLACEMENT: Models directory

Task 3: CREATE src/rag/ingestion/loaders/code_loader.py
  - IMPLEMENT: CodeLoader for source code files
  - FOLLOW pattern: Document loader interface
  - NAMING: CodeLoader, load_file, extract_metadata methods
  - DEPENDENCIES: AST parser for structure extraction
  - PLACEMENT: Document loaders module

Task 4: CREATE src/rag/ingestion/loaders/markdown_loader.py
  - IMPLEMENT: MarkdownLoader for .md files
  - FOLLOW pattern: Extract structure (headers, code blocks)
  - NAMING: MarkdownLoader, parse_structure methods
  - DEPENDENCIES: Markdown parsing library
  - PLACEMENT: Document loaders module

Task 5: CREATE src/rag/ingestion/document_loader.py
  - IMPLEMENT: DocumentLoader orchestrating all loaders
  - FOLLOW pattern: Factory pattern for loader selection
  - NAMING: DocumentLoader, load_document, detect_type methods
  - DEPENDENCIES: All specific loaders
  - PLACEMENT: Ingestion module

Task 6: CREATE src/rag/chunking/semantic_chunker.py
  - IMPLEMENT: SemanticChunker using sentence embeddings
  - FOLLOW pattern: Sliding window with semantic similarity
  - NAMING: SemanticChunker, chunk_by_similarity methods
  - DEPENDENCIES: LLM service for embeddings
  - PLACEMENT: Chunking module

Task 7: CREATE src/rag/chunking/code_chunker.py
  - IMPLEMENT: CodeChunker using AST boundaries
  - FOLLOW pattern: Preserve function/class boundaries
  - NAMING: CodeChunker, chunk_by_ast methods
  - DEPENDENCIES: AST parser from analysis module
  - PLACEMENT: Chunking module

Task 8: CREATE src/rag/retrieval/query_analyzer.py
  - IMPLEMENT: QueryAnalyzer for intent detection
  - FOLLOW pattern: Keyword extraction, intent classification
  - NAMING: QueryAnalyzer, analyze_query, detect_intent methods
  - DEPENDENCIES: NLP utilities
  - PLACEMENT: Retrieval module

Task 9: CREATE src/rag/retrieval/relevance_scorer.py
  - IMPLEMENT: RelevanceScorer with dynamic scoring
  - FOLLOW pattern: Multi-factor scoring based on intent
  - NAMING: RelevanceScorer, score_chunk, adjust_weights methods
  - DEPENDENCIES: Query analysis, chunk metadata
  - PLACEMENT: Retrieval module

Task 10: CREATE src/rag/retrieval/structural_search.py
  - IMPLEMENT: StructuralSearch for code structure queries
  - FOLLOW pattern: AST-based matching
  - NAMING: StructuralSearch, search_by_structure methods
  - DEPENDENCIES: AST parser, code analysis
  - PLACEMENT: Retrieval module

Task 11: CREATE src/rag/context/context_builder.py
  - IMPLEMENT: ContextBuilder with optimization
  - FOLLOW pattern: U-shaped attention, relevance ordering
  - NAMING: ContextBuilder, build_context, optimize_ordering methods
  - DEPENDENCIES: Token counting, relevance scores
  - PLACEMENT: Context module

Task 12: CREATE src/rag/context/window_optimizer.py
  - IMPLEMENT: WindowOptimizer for token management
  - FOLLOW pattern: Dynamic truncation, compression
  - NAMING: WindowOptimizer, optimize_window, estimate_tokens methods
  - DEPENDENCIES: Token counter (tiktoken)
  - PLACEMENT: Context module

Task 13: CREATE src/rag/pipeline.py
  - IMPLEMENT: RAGPipeline orchestrating all components
  - FOLLOW pattern: Pipeline pattern with stages
  - NAMING: RAGPipeline, process_query, ingest_documents methods
  - DEPENDENCIES: All RAG components
  - PLACEMENT: RAG module root

Task 14: CREATE src/services/document_service.py
  - IMPLEMENT: DocumentService for document management
  - FOLLOW pattern: Service facade pattern
  - NAMING: DocumentService, ingest, search, get_document methods
  - DEPENDENCIES: RAG pipeline, memory service
  - PLACEMENT: Services layer

Task 15: MODIFY src/services/rag_service.py
  - ENHANCE: Integrate new RAG pipeline
  - FIND pattern: generate_with_context method
  - REPLACE: Use new context builder and retrieval
  - PRESERVE: Backward compatibility

Task 16: CREATE src/tests/rag/test_chunking.py
  - IMPLEMENT: Unit tests for chunking strategies
  - FOLLOW pattern: Test each chunking strategy
  - COVERAGE: Semantic, structural, code-aware chunking
  - PLACEMENT: RAG test directory

Task 17: CREATE src/tests/rag/test_retrieval.py
  - IMPLEMENT: Tests for retrieval and scoring
  - FOLLOW pattern: Mock embeddings, test ranking
  - COVERAGE: Query analysis, relevance scoring, reranking
  - PLACEMENT: RAG test directory

Task 18: CREATE src/tests/rag/test_context.py
  - IMPLEMENT: Tests for context optimization
  - FOLLOW pattern: Token counting, window optimization
  - COVERAGE: Context building, compression, ordering
  - PLACEMENT: RAG test directory
```

### Implementation Patterns & Key Details

```python
# Semantic chunking pattern
class SemanticChunker:
    """
    PATTERN: Chunk by semantic similarity
    CRITICAL: Maintain context at boundaries
    """
    
    def __init__(self, llm_service, similarity_threshold: float = 0.7):
        self.llm_service = llm_service
        self.similarity_threshold = similarity_threshold
    
    async def chunk_document(
        self,
        document: Document,
        target_chunk_size: int = 500,
        overlap: int = 50
    ) -> List[Chunk]:
        """
        Chunk document by semantic similarity.
        GOTCHA: Must handle sentence boundaries
        """
        # Split into sentences first
        sentences = self._split_sentences(document.content)
        
        # Generate embeddings for sentences
        embeddings = await self.llm_service.generate_embeddings(sentences)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += self._estimate_tokens(sentence)
            
            # Check if we should start new chunk
            should_split = False
            
            # Size-based split
            if current_tokens >= target_chunk_size:
                should_split = True
            
            # Semantic boundary split
            elif i < len(sentences) - 1:
                next_embedding = embeddings[i + 1]
                similarity = self._cosine_similarity(embedding, next_embedding)
                if similarity < self.similarity_threshold:
                    should_split = True
            
            if should_split or i == len(sentences) - 1:
                # Create chunk
                chunk_content = " ".join(current_chunk)
                chunk = Chunk(
                    id=f"{document.id}_chunk_{len(chunks)}",
                    document_id=document.id,
                    content=chunk_content,
                    chunk_index=len(chunks),
                    token_count=current_tokens,
                    metadata=document.metadata
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if overlap > 0 and i < len(sentences) - 1:
                    overlap_sentences = current_chunk[-overlap:]
                    current_chunk = overlap_sentences
                    current_tokens = sum(
                        self._estimate_tokens(s) for s in overlap_sentences
                    )
                else:
                    current_chunk = []
                    current_tokens = 0
        
        return chunks

# Code-aware chunking pattern
class CodeChunker:
    """
    PATTERN: Chunk code by AST boundaries
    CRITICAL: Preserve syntactic validity
    """
    
    def __init__(self, ast_parser):
        self.ast_parser = ast_parser
    
    async def chunk_code(
        self,
        document: Document,
        max_chunk_size: int = 500
    ) -> List[Chunk]:
        """
        Chunk code preserving function/class boundaries.
        PATTERN: Use AST to find natural boundaries
        """
        # Parse AST
        ast = await self.ast_parser.parse_code(
            document.content,
            language=document.metadata.get("language", "python")
        )
        
        # Extract code entities (functions, classes)
        entities = self._extract_entities(ast)
        
        chunks = []
        for entity in entities:
            # Check entity size
            entity_tokens = self._estimate_tokens(entity.content)
            
            if entity_tokens <= max_chunk_size:
                # Entity fits in one chunk
                chunk = Chunk(
                    id=f"{document.id}_chunk_{len(chunks)}",
                    document_id=document.id,
                    content=entity.content,
                    chunk_index=len(chunks),
                    chunk_type="code",
                    function_name=entity.name if entity.type == "function" else None,
                    class_name=entity.name if entity.type == "class" else None,
                    language=document.metadata.get("language"),
                    token_count=entity_tokens
                )
                chunks.append(chunk)
            else:
                # Entity too large, split by logical sections
                sub_chunks = self._split_large_entity(entity, max_chunk_size)
                chunks.extend(sub_chunks)
        
        return chunks

# Query intent analysis pattern
class QueryAnalyzer:
    """
    PATTERN: Detect query intent for better retrieval
    GOTCHA: Balance between accuracy and speed
    """
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query to detect intent and extract features.
        PATTERN: Multi-signal classification
        """
        # Extract keywords and entities
        keywords = self._extract_keywords(query)
        programming_terms = self._detect_programming_terms(query)
        
        # Detect intent based on patterns
        intent = self._detect_intent(query, keywords, programming_terms)
        
        # Determine content requirements
        requires_code = any(term in query.lower() for term in [
            "function", "class", "implement", "code", "example", "syntax"
        ])
        requires_docs = any(term in query.lower() for term in [
            "how", "what", "why", "explain", "documentation", "guide"
        ])
        
        return QueryAnalysis(
            query=query,
            intent=intent,
            keywords=keywords,
            programming_terms=programming_terms,
            requires_code=requires_code,
            requires_docs=requires_docs,
            confidence=0.85
        )
    
    def _detect_intent(
        self,
        query: str,
        keywords: List[str],
        programming_terms: List[str]
    ) -> QueryIntent:
        """Detect primary query intent."""
        query_lower = query.lower()
        
        # Pattern matching for intent
        if any(word in query_lower for word in ["how to", "implement", "create"]):
            return QueryIntent.EXAMPLE_SEARCH
        elif any(word in query_lower for word in ["what is", "define", "meaning"]):
            return QueryIntent.DEFINITION
        elif any(word in query_lower for word in ["why", "explain", "understand"]):
            return QueryIntent.EXPLANATION
        elif any(word in query_lower for word in ["error", "bug", "fix", "debug"]):
            return QueryIntent.DEBUGGING
        elif len(programming_terms) > 2:
            return QueryIntent.CODE_SEARCH
        elif any(word in query_lower for word in ["docs", "documentation", "guide"]):
            return QueryIntent.DOCUMENTATION
        else:
            return QueryIntent.QUESTION_ANSWER

# Dynamic relevance scoring pattern
class RelevanceScorer:
    """
    PATTERN: Adjust scoring weights based on query intent
    CRITICAL: Different intents need different ranking signals
    """
    
    def score_chunk(
        self,
        chunk: Chunk,
        query_analysis: QueryAnalysis,
        base_score: float
    ) -> float:
        """
        Calculate relevance score with intent-based adjustments.
        PATTERN: Multi-factor weighted scoring
        """
        weights = self._get_weights_for_intent(query_analysis.intent)
        
        # Calculate component scores
        semantic_score = base_score  # From vector similarity
        keyword_score = self._calculate_keyword_overlap(
            chunk.keywords,
            query_analysis.keywords
        )
        structural_score = self._calculate_structural_relevance(
            chunk,
            query_analysis
        )
        recency_score = self._calculate_recency_score(chunk)
        
        # Apply weighted combination
        final_score = (
            weights["semantic"] * semantic_score +
            weights["keyword"] * keyword_score +
            weights["structural"] * structural_score +
            weights["recency"] * recency_score
        )
        
        # Apply boosting for exact matches
        if query_analysis.requires_code and chunk.chunk_type == "code":
            final_score *= 1.2
        elif query_analysis.requires_docs and chunk.chunk_type == "text":
            final_score *= 1.1
        
        return min(1.0, final_score)
    
    def _get_weights_for_intent(self, intent: QueryIntent) -> Dict[str, float]:
        """Get scoring weights based on query intent."""
        weight_profiles = {
            QueryIntent.CODE_SEARCH: {
                "semantic": 0.3,
                "keyword": 0.2,
                "structural": 0.4,
                "recency": 0.1
            },
            QueryIntent.DOCUMENTATION: {
                "semantic": 0.5,
                "keyword": 0.3,
                "structural": 0.1,
                "recency": 0.1
            },
            QueryIntent.DEBUGGING: {
                "semantic": 0.4,
                "keyword": 0.4,
                "structural": 0.1,
                "recency": 0.1
            },
            QueryIntent.EXAMPLE_SEARCH: {
                "semantic": 0.3,
                "keyword": 0.2,
                "structural": 0.3,
                "recency": 0.2
            }
        }
        
        return weight_profiles.get(intent, {
            "semantic": 0.4,
            "keyword": 0.3,
            "structural": 0.2,
            "recency": 0.1
        })

# Context window optimization pattern
class ContextBuilder:
    """
    PATTERN: Build optimized context with U-shaped attention
    CRITICAL: Most relevant info at beginning and end
    """
    
    def build_context(
        self,
        chunks: List[RetrievalResult],
        max_tokens: int = 4000,
        optimization: ContextOptimization = None
    ) -> str:
        """
        Build optimized context from retrieved chunks.
        PATTERN: U-shaped relevance ordering
        """
        if not optimization:
            optimization = ContextOptimization()
        
        # Sort chunks by relevance
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
        
        # Apply U-shaped ordering
        if len(sorted_chunks) > 2:
            # Most relevant at beginning
            top_chunks = sorted_chunks[:len(sorted_chunks)//3]
            # Moderately relevant in middle
            middle_chunks = sorted_chunks[len(sorted_chunks)//3:2*len(sorted_chunks)//3]
            # Second most relevant at end
            bottom_chunks = sorted_chunks[2*len(sorted_chunks)//3:]
            
            # Reorder for U-shape
            reordered = top_chunks + middle_chunks + bottom_chunks
        else:
            reordered = sorted_chunks
        
        # Build context with token limit
        context_parts = []
        total_tokens = 0
        
        for result in reordered:
            chunk = result.chunk
            
            # Format chunk with metadata
            chunk_text = self._format_chunk(chunk, result)
            chunk_tokens = self._estimate_tokens(chunk_text)
            
            if total_tokens + chunk_tokens > max_tokens:
                # Try compression
                if optimization.optimization_strategy == "compress":
                    compressed = self._compress_chunk(chunk_text, max_tokens - total_tokens)
                    if compressed:
                        context_parts.append(compressed)
                        total_tokens += self._estimate_tokens(compressed)
                break
            
            context_parts.append(chunk_text)
            total_tokens += chunk_tokens
        
        return "\n\n---\n\n".join(context_parts)
    
    def _format_chunk(self, chunk: Chunk, result: RetrievalResult) -> str:
        """Format chunk with metadata for context."""
        metadata = []
        
        if chunk.chunk_type == "code":
            if chunk.function_name:
                metadata.append(f"Function: {chunk.function_name}")
            if chunk.class_name:
                metadata.append(f"Class: {chunk.class_name}")
            if chunk.language:
                metadata.append(f"Language: {chunk.language}")
        
        if chunk.section:
            metadata.append(f"Section: {chunk.section}")
        
        metadata_str = " | ".join(metadata) if metadata else ""
        
        return f"""[Source: {chunk.document_id} | Score: {result.score:.3f} | {metadata_str}]
{chunk.content}"""
```

### Integration Points

```yaml
DEPENDENCIES:
  - markdown: For markdown parsing
  - pypdf2: For PDF extraction
  - beautifulsoup4: For HTML parsing
  - nbformat: For Jupyter notebook handling
  - nltk: For sentence splitting and NLP
  
MEMORY_SERVICE:
  - integration: "Store chunks in vector database"
  - pattern: "Use project memory for document storage"
  
AST_PARSER:
  - integration: "Use for code-aware chunking"
  - pattern: "Extract function/class boundaries"
  
LLM_SERVICE:
  - integration: "Generate embeddings for chunks"
  - pattern: "Use appropriate embedding model"
  
HYBRID_SEARCH:
  - enhancement: "Add structural search to existing hybrid"
  - pattern: "Combine vector, keyword, and structural signals"

REDIS:
  - cache_keys: |
      rag:embeddings:{chunk_id} - Cached embeddings
      rag:documents:{doc_id} - Document metadata
      rag:chunks:{doc_id} - Document chunks

QDRANT:
  - collections: |
      documents - Document embeddings
      chunks - Chunk embeddings with metadata

CONFIG:
  - add to: .env
  - variables: |
      # RAG Configuration
      RAG_CHUNK_SIZE=500
      RAG_CHUNK_OVERLAP=50
      RAG_MAX_CONTEXT_TOKENS=4000
      RAG_EMBEDDING_MODEL=text-embedding-ada-002
      
      # Chunking Strategies
      RAG_DEFAULT_STRATEGY=semantic
      RAG_SIMILARITY_THRESHOLD=0.7
      
      # Retrieval Settings
      RAG_DEFAULT_K=5
      RAG_USE_RERANKING=true
      RAG_HYBRID_ALPHA=0.7
      
      # Document Processing
      RAG_MAX_FILE_SIZE_MB=50
      RAG_BATCH_SIZE=10
      RAG_STREAMING_THRESHOLD_MB=10
```

## Validation Loop

### Level 1: Syntax & Style (Immediate Feedback)

```bash
# Check new RAG module
ruff check src/rag/ --fix
mypy src/rag/ --strict
ruff format src/rag/

# Verify imports
python -c "from src.rag import RAGPipeline; print('RAG imports OK')"
python -c "from src.services.document_service import DocumentService; print('Service imports OK')"

# Expected: Zero errors, all imports resolve
```

### Level 2: Unit Tests (Component Validation)

```bash
# Test document ingestion
pytest src/tests/rag/test_ingestion.py -v --cov=src/rag/ingestion

# Test chunking strategies
pytest src/tests/rag/test_chunking.py -v --cov=src/rag/chunking

# Test retrieval and scoring
pytest src/tests/rag/test_retrieval.py -v --cov=src/rag/retrieval

# Test context optimization
pytest src/tests/rag/test_context.py -v --cov=src/rag/context

# Full RAG test suite
pytest src/tests/rag/ -v --cov=src/rag --cov-report=term-missing

# Expected: 90%+ coverage, all tests pass
```

### Level 3: Integration Testing (System Validation)

```bash
# Test document ingestion pipeline
python scripts/test_document_ingestion.py \
  --source ./test-docs/ \
  --types markdown code pdf \
  --verify-chunks
# Expected: All document types processed successfully

# Test code-aware chunking
python scripts/test_code_chunking.py \
  --file src/agents/implementer.py \
  --verify-boundaries \
  --check-syntax
# Expected: Functions/classes preserved, syntax valid

# Test semantic chunking
python scripts/test_semantic_chunking.py \
  --document ./docs/architecture.md \
  --measure-coherence
# Expected: Semantically coherent chunks

# Test hybrid search
python scripts/test_hybrid_search.py \
  --index ./src/ \
  --query "How to implement error handling in async functions" \
  --compare-methods
# Expected: Hybrid outperforms individual methods

# Test query intent detection
python scripts/test_query_intent.py \
  --queries ./test-queries.json \
  --verify-classification
# Expected: 85%+ intent detection accuracy

# Test context optimization
python scripts/test_context_optimization.py \
  --large-retrieval \
  --max-tokens 4000 \
  --verify-compression
# Expected: Context fits within token limit
```

### Level 4: Creative & Domain-Specific Validation

```bash
# Large Codebase Ingestion Test
python scripts/test_large_codebase_ingestion.py \
  --repository https://github.com/langchain-ai/langchain \
  --measure-time \
  --verify-quality
# Expected: 100+ files processed in <30 seconds

# Multi-Format Document Processing
python scripts/test_multi_format.py \
  --test-all-formats \
  --verify-extraction \
  --check-metadata
# Expected: 10+ formats handled correctly

# Chunking Quality Evaluation
python scripts/test_chunking_quality.py \
  --strategies all \
  --measure-coherence \
  --measure-information-loss
# Expected: <5% information loss at boundaries

# Query Performance Benchmark
python scripts/benchmark_rag_queries.py \
  --queries 100 \
  --measure-latency \
  --measure-relevance
# Expected: <500ms latency, 90%+ precision@5

# Context Window Optimization Test
python scripts/test_context_window.py \
  --vary-sizes 2000 4000 8000 \
  --measure-quality \
  --measure-compression
# Expected: 40%+ token reduction with quality preservation

# Streaming Ingestion Test
python scripts/test_streaming_ingestion.py \
  --large-file ./data/50mb_document.pdf \
  --monitor-memory \
  --verify-chunks
# Expected: Memory usage <500MB throughout

# Re-ranking Effectiveness
python scripts/test_reranking.py \
  --queries ./test-queries.json \
  --compare-with-baseline \
  --measure-ndcg
# Expected: 15%+ improvement in NDCG

# Agent Integration Test
python scripts/test_agent_with_rag.py \
  --agent planner \
  --task "Research and implement OAuth2 authentication" \
  --verify-context-usage
# Expected: Agent uses relevant documentation and code examples

# Incremental Indexing Test
python scripts/test_incremental_indexing.py \
  --add-documents ./new-docs/ \
  --verify-consistency \
  --measure-speed
# Expected: Incremental updates without full reindex

# Cross-Language Code Search
python scripts/test_cross_language_search.py \
  --query "error handling patterns" \
  --languages python javascript java \
  --verify-results
# Expected: Relevant results from all languages
```

## Final Validation Checklist

### Technical Validation

- [ ] All 4 validation levels completed successfully
- [ ] RAG tests achieve 90%+ coverage: `pytest src/tests/rag/ --cov=src/rag`
- [ ] No linting errors: `ruff check src/rag/`
- [ ] No type errors: `mypy src/rag/ --strict`
- [ ] All document loaders working

### Feature Validation

- [ ] 10+ document formats supported
- [ ] Code chunking preserves 95%+ syntactic integrity
- [ ] Hybrid search achieves 90%+ precision@5
- [ ] Query intent detection 85%+ accurate
- [ ] Context window optimization reduces tokens by 40%
- [ ] Streaming ingestion handles large files

### Code Quality Validation

- [ ] Follows existing service patterns
- [ ] All operations async-compatible
- [ ] Proper error handling for document processing
- [ ] Embeddings cached appropriately
- [ ] Chunk boundaries preserve context
- [ ] Token counting accurate across models

### Documentation & Deployment

- [ ] Environment variables documented
- [ ] Chunking strategies documented
- [ ] Document format support listed
- [ ] API endpoints for document service
- [ ] Query examples provided

---

## Anti-Patterns to Avoid

- ❌ Don't split code in middle of functions/classes
- ❌ Don't mix embedding models (consistency required)
- ❌ Don't ignore chunk overlap (causes context loss)
- ❌ Don't put all relevant content in middle (U-shaped attention)
- ❌ Don't skip metadata extraction (needed for filtering)
- ❌ Don't load large files entirely in memory (use streaming)
- ❌ Don't ignore query intent (affects ranking)
- ❌ Don't use fixed weights for all query types
- ❌ Don't exceed context window limits
- ❌ Don't forget to cache embeddings
