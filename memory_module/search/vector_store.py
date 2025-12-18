"""
FAISS-based vector store implementation with MRL and HNSW support.
"""

import logging
import os
import pickle
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, delete
from sqlalchemy.orm import sessionmaker

from ..interfaces import IVectorStore
from ..models import SearchResult
from ..storage.models import Base, VectorMapping

logger = logging.getLogger(__name__)

class FAISSVectorStore(IVectorStore):
    """
    Vector store implementation using FAISS with MRL (Matryoshka Representation Learning).
    
    Features:
    - HNSW index support
    - Two-stage retrieval (Truncated search -> Full rescore)
    - Persistent ID mapping via Database
    - Async operations
    """

    def __init__(
        self, 
        dimension: int, 
        index_path: str,
        database_url: str,
        index_type: str = "HNSW",
        mrl_enabled: bool = True,
        storage_dimension: int = 768,
        faiss_m: int = 32,
        faiss_ef_search: int = 64
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Index dimension (truncated if MRL enabled)
            index_path: Path to save/load index
            database_url: URL for ID mapping database
            index_type: FAISS index type (Flat, HNSW)
            mrl_enabled: Enable two-stage retrieval
            storage_dimension: Full vector dimension
        """
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index_type = index_type
        self.mrl_enabled = mrl_enabled
        self.storage_dimension = storage_dimension
        self.database_url = database_url
        self.faiss_m = faiss_m
        self.faiss_ef_search = faiss_ef_search

        # Async DB setup
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False)
        
        # Thread pool for FAISS CPU operations
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Initialize index
        self.index = self._create_index()
        self.next_id = 0
        
        # Load index from disk if exists
        if self.index_path.exists():
            try:
                self._load_index_sync()
                # We need to sync next_id from DB or index size
                self.next_id = self.index.ntotal
                logger.info(f"Loaded existing FAISS index from {self.index_path} with {self.next_id} vectors")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}. Starting with empty index.")
        
        logger.info(
            f"FAISS vector store initialized (dim={dimension}, storage_dim={storage_dimension}, type={index_type})"
        )

    async def init_db(self) -> None:
        """Initialize database mappings table."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Vector mapping table verified")

    def _create_index(self) -> faiss.Index:
        """Create FAISS index."""
        if self.index_type == "Flat":
            return faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self.dimension, self.faiss_m)
            index.hnsw.efSearch = self.faiss_ef_search
            return index
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def _load_index_sync(self):
        """Synchronous load of FAISS index."""
        index_file = self.index_path / "index.faiss"
        if index_file.exists():
            self.index = faiss.read_index(str(index_file))

    async def add(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Add a single vector."""
        await self.batch_add([id], [vector], [metadata])

    async def batch_add(
        self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Add multiple vectors with MRL support.
        
        - Slices vectors to index dimension for FAISS
        - Stores full vectors in DB for re-ranking
        """
        if not ids:
            return

        loop = asyncio.get_running_loop()
        
        # 1. Prepare vectors
        # If MRL, vector is full size. We slice for index.
        full_vectors = np.array(vectors, dtype=np.float32)
        
        if self.mrl_enabled:
            # Slice to index dimension
            index_vectors = full_vectors[:, :self.dimension].copy()
            # Normalize sliced vectors for Cosine (Inner Product)
            faiss.normalize_L2(index_vectors)
        else:
            index_vectors = full_vectors
            faiss.normalize_L2(index_vectors)

        # 2. Add to FAISS (Sync in executor)
        start_id = self.next_id
        count = len(ids)
        
        def _add_to_faiss():
            self.index.add(index_vectors)
            return self.index.ntotal
            
        new_total = await loop.run_in_executor(self.executor, _add_to_faiss)
        self.next_id = new_total

        # 3. Store mappings in DB
        async with self.async_session() as session:
            try:
                for i, (uid, full_vec) in enumerate(zip(ids, vectors)):
                    faiss_id = start_id + i
                    mapping = VectorMapping(
                        faiss_id=faiss_id,
                        memory_id=uid,
                        full_vector=full_vec  # Store full vector
                    )
                    session.add(mapping)
                await session.commit()
            except Exception as e:
                logger.error(f"Failed to save vector mappings: {e}")
                await session.rollback()
                raise

        logger.info(f"Added {count} vectors to index and DB")

    async def search(
        self, query_vector: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Two-stage retrieval if MRL enabled.
        """
        if self.index.ntotal == 0:
            return []

        loop = asyncio.get_running_loop()
        
        # 1. Prepare query
        query_np = np.array([query_vector], dtype=np.float32)
        
        if self.mrl_enabled:
            # Stage 1: Over-fetch candidates using truncated vector
            search_k = min(top_k * 5, self.index.ntotal) # Fetch 5x candidates
            query_truncated = query_np[:, :self.dimension].copy()
            faiss.normalize_L2(query_truncated)
            
            def _search_faiss():
                return self.index.search(query_truncated, search_k)
                
            scores, indices = await loop.run_in_executor(self.executor, _search_faiss)
            
            # Stage 2: Re-rank
            return await self._rerank_candidates(query_vector, indices[0], scores[0], top_k)
            
        else:
            # Standard search
            faiss.normalize_L2(query_np)
            def _search_faiss_std():
                return self.index.search(query_np, top_k)
                
            scores, indices = await loop.run_in_executor(self.executor, _search_faiss_std)
            return await self._fetch_results(indices[0], scores[0])

    async def _rerank_candidates(
        self, query_vector: List[float], faiss_ids: np.ndarray, initial_scores: np.ndarray, top_k: int
    ) -> List[SearchResult]:
        """Fetch full vectors and rescore."""
        valid_ids = [int(idx) for idx in faiss_ids if idx >= 0]
        if not valid_ids:
            return []

        # Fetch full vectors from DB
        async with self.async_session() as session:
            stmt = select(VectorMapping).where(VectorMapping.faiss_id.in_(valid_ids))
            result = await session.execute(stmt)
            mappings = result.scalars().all()
            
        mapping_dict = {m.faiss_id: m for m in mappings}
        
        # Compute exact cosine similarity
        candidates = []
        query_norm = np.array(query_vector) / np.linalg.norm(query_vector)
        
        for idx in valid_ids:
            mapping = mapping_dict.get(idx)
            if not mapping:
                continue
                
            full_vec = np.array(mapping.full_vector)
            full_vec_norm = full_vec / np.linalg.norm(full_vec)
            
            score = np.dot(query_norm, full_vec_norm)
            candidates.append((score, mapping.memory_id))
            
        # Sort by new score
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:top_k]
        
        # Hydrate result logic?
        # SearchResult needs content, which is in MemoryRecord.
        # But IVectorStore search usually assumes metadata is returned.
        # Since we moved metadata to DB (MemoryRecord), we might need to fetch it here OR 
        # let the caller handle it. The interface says `search` returns `SearchResult` which has content.
        # BUT `FAISSVectorStore` previously had `id_to_metadata`. Now we don't.
        # We only have `VectorMapping`.
        # So we return minimal SearchResults and let Service hydrate? 
        # OR we join MemoryRecord here.
        # Joining MemoryRecord is better for `SearchResult`.
        
        # Let's fetch MemoryRecords for top candidates
        return await self._hydrate_results([c[1] for c in top_candidates], [c[0] for c in top_candidates])

    async def _fetch_results(self, faiss_ids: np.ndarray, scores: np.ndarray) -> List[SearchResult]:
        """Fetch results for non-MRL search."""
        valid_ids = [int(idx) for idx in faiss_ids if idx >= 0]
        valid_scores = [float(scores[i]) for i, idx in enumerate(faiss_ids) if idx >= 0]
        
        async with self.async_session() as session:
            stmt = select(VectorMapping).where(VectorMapping.faiss_id.in_(valid_ids))
            result = await session.execute(stmt)
            mappings = result.scalars().all()
            
        mapping_dict = {m.faiss_id: m.memory_id for m in mappings}
        memory_ids = [mapping_dict[idx] for idx in valid_ids if idx in mapping_dict]
        
        # Maintain score order? 
        # The DB return order is undefined. We need to map back.
        
        # Simplify: just hydrate based on retrieved mappings
        # Note: If some IDs are missing in DB, we skip them.
        
        final_ids = []
        final_scores = []
        for idx, score in zip(valid_ids, valid_scores):
            if idx in mapping_dict:
                final_ids.append(mapping_dict[idx])
                final_scores.append(score)
                
        return await self._hydrate_results(final_ids, final_scores)

    async def _hydrate_results(self, memory_ids: List[str], scores: List[float]) -> List[SearchResult]:
        """Fetch MemoryRecords and create SearchResults."""
        if not memory_ids:
            return []
            
        # We need to import MemoryRecordDB to query it, but circular import risk?
        # Models are in ..storage.models, likely safe.
        from ..storage.models import MemoryRecordDB
        
        async with self.async_session() as session:
            stmt = select(MemoryRecordDB).where(MemoryRecordDB.id.in_(memory_ids))
            result = await session.execute(stmt)
            records = result.scalars().all()
            
        record_dict = {r.id: r for r in records}
        results = []
        
        for mid, score in zip(memory_ids, scores):
            record = record_dict.get(mid)
            if record:
                results.append(SearchResult(
                    id=record.id,
                    content=record.content,
                    score=float(score),
                    metadata=record.record_metadata,
                    memory_type=record.memory_type,
                    tags=record.tags,
                    created_at=record.created_at
                ))
                
        return results

    async def delete(self, id: str) -> bool:
        """
        Delete vector.
        Since FAISS deletion is hard, we just delete the mapping.
        Search results will be filtered out because they won't be found in DB.
        """
        async with self.async_session() as session:
            stmt = delete(VectorMapping).where(VectorMapping.memory_id == id)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    async def save(self, path: str) -> None:
        """Save FAISS index to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        index_file = save_path / "index.faiss"
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, faiss.write_index, self.index, str(index_file))
        logger.info(f"Saved FAISS index to {save_path}")

    async def load(self, path: str) -> None:
        """Load index manually."""
        self.index_path = Path(path)
        loop = asyncio.get_running_loop()
        self.index = await loop.run_in_executor(self.executor, faiss.read_index, str(self.index_path / "index.faiss"))

    async def close(self) -> None:
        """Close database connections and executor."""
        if hasattr(self, 'engine'):
            await self.engine.dispose()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        logger.info("FAISS vector store connections closed")

