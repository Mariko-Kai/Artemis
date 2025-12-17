"""
Pydantic data models for the memory module.

These models define the data structures used throughout the system
with comprehensive validation and type safety.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class MemoryType(str, Enum):
    """Types of memories in the system."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryRecord(BaseModel):
    """
    Core memory record model.

    Represents a single memory with content, metadata, and embeddings.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    content: str = Field(..., min_length=1, description="Memory content text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Embedding vector")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    memory_type: MemoryType = Field(
        default=MemoryType.SEMANTIC, description="Type of memory"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    source: Optional[str] = Field(default=None, description="Source of the memory")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score"
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Ensure tags are non-empty and unique."""
        return list(set(tag.strip() for tag in v if tag.strip()))

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError("Content cannot be empty or only whitespace")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Paris is the capital of France",
                "metadata": {"category": "geography"},
                "memory_type": "semantic",
                "tags": ["geography", "fact", "europe"],
                "source": "user_input",
                "confidence": 0.95,
            }
        }


class SearchResult(BaseModel):
    """Search result with score and metadata."""

    id: UUID = Field(..., description="Memory ID")
    content: str = Field(..., description="Memory content")
    score: float = Field(..., description="Final relevance score (normalized 0-1)")
    
    # Hybrid Search Explainability
    semantic_score: Optional[float] = Field(None, description="Semantic similarity score (0-1)")
    lexical_score: Optional[float] = Field(None, description="Lexical BM25 score (normalized 0-1)")
    temporal_score: Optional[float] = Field(None, description="Temporal boost factor (1.0 = no boost)")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Memory metadata")
    memory_type: MemoryType = Field(..., description="Type of memory")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    matched_keywords: List[str] = Field(default_factory=list, description="Keywords matched in lexical search")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "content": "Paris is the capital of France",
                "score": 0.92,
                "semantic_score": 0.88,
                "lexical_score": 0.95,
                "metadata": {"category": "geography"},
                "memory_type": "semantic",
                "tags": ["geography", "fact"],
                "created_at": "2024-01-01T00:00:00Z",
                "matched_keywords": ["Paris", "France"]
            }
        }


class QueryRequest(BaseModel):
    """Request model for memory search."""

    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata filters"
    )
    hybrid_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector search (1=pure vector, 0=pure lexical)",
    )
    memory_types: Optional[List[MemoryType]] = Field(
        default=None, description="Filter by memory types"
    )
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    min_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Minimum score threshold"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or only whitespace")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "query": "capital of France",
                "top_k": 5,
                "hybrid_weight": 0.7,
                "memory_types": ["semantic"],
                "tags": ["geography"],
                "min_score": 0.5,
            }
        }


class QueryResponse(BaseModel):
    """Response model for memory search."""

    results: List[SearchResult] = Field(..., description="Search results")
    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Total number of results found")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")
    hybrid_weight: float = Field(..., description="Hybrid weight used")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [],
                "query": "capital of France",
                "total_results": 5,
                "query_time_ms": 42.5,
                "hybrid_weight": 0.7,
            }
        }


class StoreRequest(BaseModel):
    """Request model for storing a memory."""

    content: str = Field(..., min_length=1, description="Memory content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")
    memory_type: MemoryType = Field(default=MemoryType.SEMANTIC, description="Type of memory")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    source: Optional[str] = Field(default=None, description="Source of the memory")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "The Earth orbits around the Sun",
                "metadata": {"category": "astronomy"},
                "memory_type": "semantic",
                "tags": ["science", "astronomy"],
                "source": "textbook",
                "confidence": 1.0,
            }
        }


class BatchStoreRequest(BaseModel):
    """Request model for storing multiple memories."""

    memories: List[StoreRequest] = Field(..., min_items=1, description="List of memories to store")

    class Config:
        json_schema_extra = {
            "example": {
                "memories": [
                    {
                        "content": "Paris is the capital of France",
                        "tags": ["geography"],
                    },
                    {
                        "content": "The Earth orbits around the Sun",
                        "tags": ["astronomy"],
                    },
                ]
            }
        }


class StatsResponse(BaseModel):
    """Statistics response model."""

    total_memories: int = Field(..., description="Total number of memories")
    memory_types_count: Dict[str, int] = Field(..., description="Count by memory type")
    total_tags: int = Field(..., description="Total number of unique tags")
    vector_dimension: int = Field(..., description="Embedding vector dimension")
    index_size_bytes: Optional[int] = Field(default=None, description="Vector index size in bytes")

    class Config:
        json_schema_extra = {
            "example": {
                "total_memories": 100,
                "memory_types_count": {
                    "semantic": 50,
                    "episodic": 30,
                    "long_term": 20,
                },
                "total_tags": 25,
                "vector_dimension": 768,
                "index_size_bytes": 1024000,
            }
        }
