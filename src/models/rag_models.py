"""RAG-specific data models."""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal
from enum import Enum

from .document_models import Chunk, DocumentType, ChunkingStrategy


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
    programming_terms: List[str] = Field(
        default_factory=list, description="Programming-specific terms"
    )
    requires_code: bool = Field(
        default=False, description="Whether query needs code context"
    )
    requires_docs: bool = Field(
        default=False, description="Whether query needs documentation"
    )
    confidence: float = Field(
        default=0.5, ge=0, le=1, description="Intent detection confidence"
    )


class RetrievalRequest(BaseModel):
    """Request for document retrieval."""

    query: str = Field(description="Search query")
    k: int = Field(default=5, ge=1, le=100, description="Number of results")
    document_types: Optional[List[DocumentType]] = Field(
        default=None, description="Filter by document type"
    )
    chunking_strategy: Optional[ChunkingStrategy] = Field(
        default=None, description="Preferred chunking"
    )
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
    highlights: List[str] = Field(
        default_factory=list, description="Highlighted excerpts"
    )


class IngestionRequest(BaseModel):
    """Request for document ingestion."""

    source_path: str = Field(description="Path to document or directory")
    document_type: Optional[DocumentType] = Field(
        default=None, description="Force document type"
    )
    chunking_strategy: ChunkingStrategy = Field(default=ChunkingStrategy.SEMANTIC)
    chunk_size: int = Field(default=500, description="Target chunk size in tokens")
    chunk_overlap: int = Field(
        default=50, description="Overlap between chunks in tokens"
    )
    generate_embeddings: bool = Field(
        default=True, description="Generate embeddings immediately"
    )
    extract_metadata: bool = Field(
        default=True, description="Extract document metadata"
    )
    recursive: bool = Field(
        default=False, description="Process directories recursively"
    )


class ContextOptimization(BaseModel):
    """Context window optimization settings."""

    max_tokens: int = Field(default=4000, description="Maximum context window tokens")
    optimization_strategy: Literal["truncate", "summarize", "compress"] = Field(
        default="compress"
    )
    relevance_ordering: Literal["score", "diversity", "recency"] = Field(
        default="score"
    )
    include_system_prompt: bool = Field(default=True)
    include_metadata: bool = Field(default=True, description="Include chunk metadata in context")
    reserve_tokens: int = Field(default=500, description="Tokens reserved for response")
    use_sliding_window: bool = Field(
        default=False, description="Use sliding window for long contexts"
    )
