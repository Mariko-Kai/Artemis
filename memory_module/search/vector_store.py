"""
FAISS-based vector store implementation.

Provides similarity search using FAISS with metadata mapping.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from ..interfaces import IVectorStore
from ..models import SearchResult

logger = logging.getLogger(__name__)


class FAISSVectorStore(IVectorStore):
    """Vector store implementation using FAISS."""

    def __init__(self, dimension: int, index_path: str, index_type: str = "Flat"):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Dimension of embedding vectors
            index_path: Path to save/load the index
            index_type: Type of FAISS index (Flat, IVF, HNSW)
        """
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index_type = index_type

        # Initialize index
        self.index = self._create_index()

        # Metadata mapping: FAISS internal ID -> (UUID, metadata)
        self.id_to_metadata: Dict[int, tuple[str, Dict[str, Any]]] = {}
        self.uuid_to_faiss_id: Dict[str, int] = {}
        self.next_id = 0

        # Try to load existing index
        if self.index_path.exists():
            try:
                self._load_index()
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}. Starting with empty index.")

        logger.info(
            f"FAISS vector store initialized (dimension={dimension}, type={index_type})"
        )

    def _create_index(self) -> faiss.Index:
        """
        Create a FAISS index based on type.

        Returns:
            FAISS index
        """
        if self.index_type == "Flat":
            # Flat index with inner product (for normalized vectors)
            return faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IVF":
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatIP(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "HNSW":
            # HNSW index for fast approximate search
            return faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    async def add(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """
        Add a single vector to the store.

        Args:
            id: UUID as string
            vector: Embedding vector
            metadata: Associated metadata
        """
        if id in self.uuid_to_faiss_id:
            logger.warning(f"Vector with ID {id} already exists, skipping")
            return

        # Convert to numpy array and normalize
        vec = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(vec)

        # Add to index
        self.index.add(vec)

        # Store metadata mapping
        faiss_id = self.next_id
        self.id_to_metadata[faiss_id] = (id, metadata)
        self.uuid_to_faiss_id[id] = faiss_id
        self.next_id += 1

        logger.debug(f"Added vector {id} (FAISS ID: {faiss_id})")

    async def batch_add(
        self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Add multiple vectors to the store.

        Args:
            ids: List of UUIDs
            vectors: List of embedding vectors
            metadatas: List of metadata dictionaries
        """
        if not ids or not vectors or not metadatas:
            return

        if len(ids) != len(vectors) != len(metadatas):
            raise ValueError("Length mismatch between ids, vectors, and metadatas")

        # Filter out existing IDs
        new_entries = [
            (id, vec, meta)
            for id, vec, meta in zip(ids, vectors, metadatas)
            if id not in self.uuid_to_faiss_id
        ]

        if not new_entries:
            logger.warning("All provided IDs already exist, skipping batch add")
            return

        new_ids, new_vectors, new_metadatas = zip(*new_entries)

        # Convert to numpy array and normalize
        vecs = np.array(new_vectors, dtype=np.float32)
        faiss.normalize_L2(vecs)

        # Add to index
        self.index.add(vecs)

        # Store metadata mappings
        for i, (id, metadata) in enumerate(zip(new_ids, new_metadatas)):
            faiss_id = self.next_id + i
            self.id_to_metadata[faiss_id] = (id, metadata)
            self.uuid_to_faiss_id[id] = faiss_id

        self.next_id += len(new_entries)
        logger.info(f"Batch added {len(new_entries)} vectors")

    async def search(
        self, query_vector: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding
            top_k: Number of results
            filters: Optional metadata filters (currently not implemented)

        Returns:
            List of search results
        """
        if self.index.ntotal == 0:
            return []

        # Convert and normalize query vector
        query = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query)

        # Search
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        # Convert to SearchResult objects
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue

            uuid_str, metadata = self.id_to_metadata.get(int(idx), (None, {}))
            if uuid_str is None:
                continue

            # TODO: Apply filters if provided
            results.append(
                SearchResult(
                    id=uuid_str,
                    content=metadata.get("content", ""),
                    score=float(score),
                    metadata=metadata.get("metadata", {}),
                    memory_type=metadata.get("memory_type", "semantic"),
                    tags=metadata.get("tags", []),
                    created_at=metadata.get("created_at"),
                )
            )

        return results

    async def delete(self, id: str) -> bool:
        """
        Delete a vector from the store.

        Note: FAISS doesn't support deletion, so we just remove from metadata.
        The actual vector remains in the index.

        Args:
            id: UUID to delete

        Returns:
            True if deleted, False if not found
        """
        faiss_id = self.uuid_to_faiss_id.get(id)
        if faiss_id is None:
            return False

        # Remove from mappings
        del self.id_to_metadata[faiss_id]
        del self.uuid_to_faiss_id[id]

        logger.debug(f"Deleted vector {id} (FAISS ID: {faiss_id})")
        return True

    async def save(self, path: str) -> None:
        """
        Save the index to disk.

        Args:
            path: Directory path to save the index
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = save_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))

        # Save metadata
        metadata_file = save_path / "metadata.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump(
                {
                    "id_to_metadata": self.id_to_metadata,
                    "uuid_to_faiss_id": self.uuid_to_faiss_id,
                    "next_id": self.next_id,
                },
                f,
            )

        logger.info(f"Saved FAISS index to {save_path}")

    async def load(self, path: str) -> None:
        """
        Load the index from disk.

        Args:
            path: Directory path to load the index from
        """
        self._load_index(Path(path))

    def _load_index(self, path: Optional[Path] = None) -> None:
        """
        Internal method to load index.

        Args:
            path: Optional path override
        """
        load_path = path or self.index_path
        if not load_path.exists():
            raise FileNotFoundError(f"Index path does not exist: {load_path}")

        # Load FAISS index
        index_file = load_path / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        self.index = faiss.read_index(str(index_file))

        # Load metadata
        metadata_file = load_path / "metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, "rb") as f:
                data = pickle.load(f)
                self.id_to_metadata = data["id_to_metadata"]
                self.uuid_to_faiss_id = data["uuid_to_faiss_id"]
                self.next_id = data["next_id"]

        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
