"""Document data models for RAG system."""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
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
    chunk_type: str = Field(
        default="text", description="Type of chunk (text/code/header)"
    )
    language: Optional[str] = Field(
        default=None, description="Programming language if code"
    )

    # Structural information
    section: Optional[str] = Field(default=None, description="Document section")
    subsection: Optional[str] = Field(default=None, description="Document subsection")
    page_number: Optional[int] = Field(
        default=None, description="Page number if applicable"
    )

    # Code-specific
    function_name: Optional[str] = Field(
        default=None, description="Function name if code"
    )
    class_name: Optional[str] = Field(default=None, description="Class name if code")

    # Embeddings and search
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector embedding"
    )
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    token_count: int = Field(default=0, description="Number of tokens")


class Document(BaseModel):
    """Represents a document for ingestion."""

    id: str = Field(description="Unique document ID")
    content: str = Field(description="Document content")
    type: DocumentType = Field(description="Document type")
    source: str = Field(description="Document source (path/URL)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    processed: bool = Field(default=False)
    chunks: List[Chunk] = Field(default_factory=list, description="Document chunks")
    embeddings_generated: bool = Field(default=False)
