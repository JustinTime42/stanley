"""Memory data models for the hierarchical memory system."""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class MemoryType(str, Enum):
    """Memory tier enumeration."""

    WORKING = "working"
    PROJECT = "project"
    GLOBAL = "global"


class MemoryItem(BaseModel):
    """Base memory item model."""

    id: str = Field(description="Unique memory identifier")
    content: str = Field(description="Memory content/text")
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector embedding"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Memory metadata"
    )
    memory_type: MemoryType = Field(description="Memory tier")
    agent_id: str = Field(description="Agent that created memory")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    project_id: Optional[str] = Field(default=None, description="Project identifier")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Creation time"
    )
    importance: float = Field(
        default=0.5, ge=0, le=1, description="Memory importance score"
    )
    access_count: int = Field(default=0, description="Number of times accessed")
    tags: List[str] = Field(default_factory=list, description="Memory tags")


class MemorySearchRequest(BaseModel):
    """Memory search request model."""

    query: str = Field(description="Search query")
    memory_types: List[MemoryType] = Field(default_factory=lambda: [MemoryType.PROJECT])
    k: int = Field(default=5, ge=1, le=100, description="Number of results")
    filters: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata filters"
    )
    use_hybrid: bool = Field(default=True, description="Use hybrid search")
    alpha: float = Field(default=0.7, ge=0, le=1, description="Hybrid search weight")
    score_threshold: float = Field(default=0.0, description="Minimum similarity score")


class MemorySearchResult(BaseModel):
    """Memory search result model."""

    memory: MemoryItem
    score: float = Field(description="Similarity/relevance score")
    source: str = Field(description="Source collection/tier")
    highlights: Optional[List[str]] = Field(
        default=None, description="Relevant excerpts"
    )
