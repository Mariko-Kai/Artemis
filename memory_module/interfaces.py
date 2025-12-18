"""
Abstract interfaces for the memory module.

These interfaces define the contracts for all major components,
allowing for dependency injection and easier testing.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import MemoryRecord, QueryRequest, QueryResponse, SearchResult


class IEmbeddingService(ABC):
    """Interface for embedding generation services."""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    async def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Dimension of embedding vectors
        """
        pass


class IVectorStore(ABC):
    """Interface for vector similarity search."""

    @abstractmethod
    async def add(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """
        Add a single vector to the store.

        Args:
            id: Unique identifier for the vector
            vector: Embedding vector
            metadata: Associated metadata
        """
        pass

    @abstractmethod
    async def batch_add(
        self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Add multiple vectors to the store.

        Args:
            ids: List of unique identifiers
            vectors: List of embedding vectors
            metadatas: List of associated metadata dictionaries
        """
        pass

    @abstractmethod
    async def search(
        self, query_vector: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results sorted by similarity
        """
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """
        Delete a vector from the store.

        Args:
            id: Unique identifier of the vector to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def save(self, path: str) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Path to save the index
        """
        pass

    @abstractmethod
    async def load(self, path: str) -> None:
        """
        Load the vector store from disk.

        Args:
            path: Path to load the index from
        """
        pass


class ILexicalIndex(ABC):
    """Interface for lexical (BM25) search."""

    @abstractmethod
    async def index_document(self, id: str, text: str, metadata: Dict[str, Any]) -> None:
        """
        Index a document for lexical search.

        Args:
            id: Unique identifier for the document
            text: Document text content
            metadata: Associated metadata
        """
        pass

    @abstractmethod
    async def batch_index(
        self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Index multiple documents.

        Args:
            ids: List of unique identifiers
            texts: List of document texts
            metadatas: List of associated metadata dictionaries
        """
        pass

    @abstractmethod
    async def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search for documents using BM25.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of search results sorted by BM25 score
        """
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """
        Remove a document from the index.

        Args:
            id: Unique identifier of the document to delete

        Returns:
            True if deleted, False if not found
        """
        pass


class IMetadataStore(ABC):
    """Interface for metadata persistence."""

    @abstractmethod
    async def save(self, record: MemoryRecord) -> MemoryRecord:
        """
        Save a memory record to the database.

        Args:
            record: Memory record to save

        Returns:
            Saved memory record with generated ID
        """
        pass

    @abstractmethod
    async def batch_save(self, records: List[MemoryRecord]) -> List[MemoryRecord]:
        """
        Save multiple memory records.

        Args:
            records: List of memory records to save

        Returns:
            List of saved records with generated IDs
        """
        pass

    @abstractmethod
    async def get(self, id: str) -> Optional[MemoryRecord]:
        """
        Retrieve a memory record by ID.

        Args:
            id: Unique identifier of the record

        Returns:
            Memory record if found, None otherwise
        """
        pass

    @abstractmethod
    async def query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[MemoryRecord]:
        """
        Query memory records with filters.

        Args:
            filters: Optional filter criteria
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching memory records
        """
        pass

    @abstractmethod
    async def update(self, id: str, updates: Dict[str, Any]) -> Optional[MemoryRecord]:
        """
        Update a memory record.

        Args:
            id: Unique identifier of the record
            updates: Dictionary of fields to update

        Returns:
            Updated memory record if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """
        Delete a memory record.

        Args:
            id: Unique identifier of the record to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 20) -> List[MemoryRecord]:
        """
        Search for memory records in SQL storage (hot storage).
        Typically used for records not yet in vector/lexical indexes.

        Args:
            query: Search query string
            filters: Optional metadata filters
            limit: Maximum number of results

        Returns:
            List of matching memory records
        """
        pass


class IMemoryService(ABC):
    """High-level interface for memory operations."""

    @abstractmethod
    async def store_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryRecord:
        """
        Store a new memory.

        This orchestrates:
        1. Embedding generation
        2. Database persistence
        3. Vector indexing
        4. Lexical indexing

        Args:
            content: Memory content text
            metadata: Optional metadata

        Returns:
            Stored memory record
        """
        pass

    @abstractmethod
    async def search_memories(self, request: QueryRequest) -> QueryResponse:
        """
        Search for memories using hybrid search.

        Args:
            request: Query request with parameters

        Returns:
            Query response with results and metadata
        """
        pass

    @abstractmethod
    async def get_memory(self, id: str) -> Optional[MemoryRecord]:
        """
        Retrieve a specific memory by ID.

        Args:
            id: Memory ID

        Returns:
            Memory record if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_memory(self, id: str) -> bool:
        """
        Delete a memory.

        Args:
            id: Memory ID

        Returns:
            True if deleted, False if not found
        """
        pass
