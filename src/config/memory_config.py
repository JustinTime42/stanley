"""Memory system configuration with environment variable loading."""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class MemoryConfig(BaseModel):
    """Configuration for the hierarchical memory system."""

    # Redis Configuration
    redis_url: str = Field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        description="Redis connection URL",
    )
    redis_maxmemory: str = Field(
        default_factory=lambda: os.getenv("REDIS_MAXMEMORY", "2gb"),
        description="Redis max memory limit",
    )
    redis_maxmemory_policy: str = Field(
        default_factory=lambda: os.getenv("REDIS_MAXMEMORY_POLICY", "allkeys-lru"),
        description="Redis eviction policy",
    )

    # Qdrant Configuration
    qdrant_url: str = Field(
        default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"),
        description="Qdrant connection URL",
    )
    qdrant_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY"),
        description="Qdrant API key (optional for local)",
    )

    # Embedding Configuration
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        description="Embedding model to use",
    )
    vector_size: int = Field(
        default_factory=lambda: int(os.getenv("VECTOR_SIZE", "1536")),
        description="Vector embedding dimension",
    )

    # Memory Configuration
    memory_ttl_working: int = Field(
        default_factory=lambda: int(os.getenv("MEMORY_TTL_WORKING", "3600")),
        description="TTL for working memory (seconds)",
    )
    memory_cache_size: int = Field(
        default_factory=lambda: int(os.getenv("MEMORY_CACHE_SIZE", "1000")),
        description="Size of semantic cache",
    )

    # Hybrid Search Configuration
    hybrid_search_alpha: float = Field(
        default_factory=lambda: float(os.getenv("HYBRID_SEARCH_ALPHA", "0.7")),
        description="Hybrid search weight (0=keyword, 1=vector)",
    )

    # Collection Names
    project_collection_name: str = Field(
        default="project_memory",
        description="Qdrant collection for project memory",
    )
    global_collection_name: str = Field(
        default="global_memory",
        description="Qdrant collection for global memory",
    )

    # Performance Configuration
    batch_size: int = Field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", "100")),
        description="Batch size for bulk operations",
    )
    connection_pool_size: int = Field(
        default_factory=lambda: int(os.getenv("CONNECTION_POOL_SIZE", "10")),
        description="Connection pool size for Redis",
    )

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
