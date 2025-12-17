"""
Memory Module - A comprehensive memory and search system.

This module provides:
- Semantic search using Snowflake Arctic Embed 2
- Vector search with FAISS
- Lexical search with BM25
- Hybrid search combining vector and lexical approaches
- Metadata storage with SQLAlchemy
- FastAPI endpoints for memory management
"""

__version__ = "0.1.0"

from .interfaces import (
    IEmbeddingService,
    IVectorStore,
    ILexicalIndex,
    IMetadataStore,
    IMemoryService,
)
from .models import (
    MemoryRecord,
    QueryRequest,
    QueryResponse,
    SearchResult,
    MemoryType,
)

__all__ = [
    "IEmbeddingService",
    "IVectorStore",
    "ILexicalIndex",
    "IMetadataStore",
    "IMemoryService",
    "MemoryRecord",
    "QueryRequest",
    "QueryResponse",
    "SearchResult",
    "MemoryType",
]
