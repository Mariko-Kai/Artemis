"""
High-level memory service orchestrating all components.

This service coordinates embedding generation, storage, and search.
"""

import logging
import time
from typing import Any, Dict, Optional, List
from uuid import UUID

from .config.settings import Settings
from .embeddings.service import EmbeddingService
from .interfaces import IMemoryService
from .models import MemoryRecord, MemoryType, QueryRequest, QueryResponse, StoreRequest
from .search.hybrid import HybridSearchService
from .search.lexical import BM25LexicalIndex
from .search.vector_store import FAISSVectorStore
from .storage.postgresql import PostgreSQLMetadataStore

logger = logging.getLogger(__name__)


class MemoryService(IMemoryService):
    """
    High-level memory service implementation.

    Coordinates all components to provide a unified memory API.
    """

    def __init__(self, settings: Settings):
        """
        Initialize the memory service.

        Args:
            settings: Configuration settings
        """
        self.settings = settings

        # Initialize components
        self.embedding_service = EmbeddingService(settings)
        self.vector_dim = self.embedding_service.get_dimension()

        self.vector_store = FAISSVectorStore(
            dimension=settings.mrl_index_dim if settings.enable_mrl else self.vector_dim,
            index_path=settings.faiss_index_path,
            database_url=settings.database_url,
            index_type=settings.faiss_index_type,
            mrl_enabled=settings.enable_mrl,
            storage_dimension=settings.mrl_storage_dim,
            faiss_m=settings.faiss_m,
            faiss_ef_search=settings.faiss_ef_search,
        )

        self.lexical_index = BM25LexicalIndex(index_path=settings.lexical_index_path)

        self.metadata_store = PostgreSQLMetadataStore(settings.database_url)
        
        self.hybrid_service = HybridSearchService(
            settings=settings,
            vector_store=self.vector_store,
            lexical_index=self.lexical_index
        )

        logger.info("Memory service initialized successfully")

    async def initialize(self) -> None:
        """Initialize the service (create database tables, load indexes, etc.)."""
        await self.metadata_store.init_db()
        logger.info("Memory service initialization complete")

    async def store_memory(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        memory_type: MemoryType = MemoryType.SEMANTIC
    ) -> MemoryRecord:
        """
        Store a new memory.

        This orchestrates:
        1. Create memory record
        2. Generate embedding
        3. Save to database
        4. Index in vector store
        5. Index in lexical search

        Args:
            content: Memory content
            metadata: Optional metadata
            tags: Optional tags
            memory_type: Memory type

        Returns:
            Stored memory record
        """
        # Create memory record
        memory = MemoryRecord(
            content=content,
            metadata=metadata or {},
            tags=tags or [],
            memory_type=memory_type
        )

        # Generate embedding
        embedding = await self.embedding_service.embed(content)
        memory.embedding = embedding

        # Save to database
        saved_memory = await self.metadata_store.save(memory)

        # Index in vector store
        await self.vector_store.add(
            id=str(saved_memory.id),
            vector=embedding,
            metadata={
                "content": saved_memory.content,
                "metadata": saved_memory.metadata,
                "memory_type": saved_memory.memory_type.value,
                "tags": saved_memory.tags,
                "created_at": saved_memory.created_at,
            },
        )

        # Index in lexical search
        await self.lexical_index.index_document(
            id=str(saved_memory.id),
            text=saved_memory.content,
            metadata={
                "content": saved_memory.content,
                "metadata": saved_memory.metadata,
                "memory_type": saved_memory.memory_type.value,
                "tags": saved_memory.tags,
                "created_at": saved_memory.created_at,
            },
        )

        logger.info(f"Stored memory {saved_memory.id}")
        return saved_memory

    async def store_memory_from_request(self, request: StoreRequest) -> MemoryRecord:
        """
        Store a memory from a StoreRequest.

        Args:
            request: Store request with all fields

        Returns:
            Stored memory record
        """
        memory = MemoryRecord(
            content=request.content,
            metadata=request.metadata or {},
            memory_type=request.memory_type,
            tags=request.tags,
            source=request.source,
            confidence=request.confidence,
        )

        # Generate embedding
        embedding = await self.embedding_service.embed(memory.content)
        memory.embedding = embedding

        # Save to database
        saved_memory = await self.metadata_store.save(memory)

        # Index in vector store
        await self.vector_store.add(
            id=str(saved_memory.id),
            vector=embedding,
            metadata={
                "content": saved_memory.content,
                "metadata": saved_memory.metadata,
                "memory_type": saved_memory.memory_type.value,
                "tags": saved_memory.tags,
                "created_at": saved_memory.created_at,
            },
        )

        # Index in lexical search
        await self.lexical_index.index_document(
            id=str(saved_memory.id),
            text=saved_memory.content,
            metadata={
                "content": saved_memory.content,
                "metadata": saved_memory.metadata,
                "memory_type": saved_memory.memory_type.value,
                "tags": saved_memory.tags,
                "created_at": saved_memory.created_at,
            },
        )

        logger.info(f"Stored memory {saved_memory.id}")
        return saved_memory

    async def search_memories(self, request: QueryRequest) -> QueryResponse:
        """
        Search for memories using hybrid search.

        Args:
            request: Query request

        Returns:
            Query response with results
        """
        start_time = time.time()

        # Generate query embedding
        query_embedding = await self.embedding_service.embed(request.query)

        # Prepare filters
        filters = request.filters or {}
        
        # Perform hybrid search
        results = await self.hybrid_service.search(
            query_text=request.query,
            query_vector=query_embedding,
            top_k=request.top_k,
            filters=filters,
            hybrid_weight=request.hybrid_weight,
        )

        # Apply post-filters if provided (Hybrid service handles semantic filters, but we do explicit check here too?)
        # FAISS search supports filters, but lexical search currently doesn't (unless we add it to BM25 search).
        # Since Hybrid Service merges them, we might get lexical results that don't match filters.
        # We should apply post-filtering here to be safe.
        
        if request.memory_types:
            type_values = [t.value for t in request.memory_types]
            results = [r for r in results if r.memory_type.value in type_values]

        if request.tags:
            results = [r for r in results if any(tag in r.tags for tag in request.tags)]

        if request.min_score is not None:
            results = [r for r in results if r.score >= request.min_score]

        # Limit to top_k after filtering
        results = results[: request.top_k]

        # Calculate query time
        query_time_ms = (time.time() - start_time) * 1000

        return QueryResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            query_time_ms=query_time_ms,
            hybrid_weight=request.hybrid_weight,
        )

    async def get_memory(self, id: str) -> Optional[MemoryRecord]:
        """
        Retrieve a specific memory by ID.

        Args:
            id: Memory ID

        Returns:
            Memory record if found
        """
        return await self.metadata_store.get(id)

    async def delete_memory(self, id: str) -> bool:
        """
        Delete a memory.

        This removes from:
        1. Database
        2. Vector store
        3. Lexical index

        Args:
            id: Memory ID

        Returns:
            True if deleted, False if not found
        """
        # Delete from database
        db_deleted = await self.metadata_store.delete(id)
        if not db_deleted:
            return False

        # Delete from vector store
        await self.vector_store.delete(id)

        # Delete from lexical index
        await self.lexical_index.delete(id)

        logger.info(f"Deleted memory {id}")
        return True

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics.

        Returns:
            Statistics dictionary
        """
        db_stats = await self.metadata_store.get_stats()

        return {
            **db_stats,
            "vector_dimension": self.vector_dim,
            "total_tags": 0,  # TODO: Calculate unique tags
        }

    async def save_indexes(self) -> None:
        """Save indexes to disk for persistence."""
        await self.vector_store.save(self.settings.faiss_index_path)
        if self.settings.lexical_index_path:
             self.lexical_index.save(self.settings.lexical_index_path)
        logger.info("Indexes saved to disk")
