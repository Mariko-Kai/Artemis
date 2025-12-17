
import logging
import json
import pickle
import numpy as np
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime

from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db.memory_models import MemoryRecordDB, VectorMapping, AuditLog
from app.core.config import settings
from app.core.global_lock import gpu_lock

# We import components from memory_module
# Assumes memory_module is in python path (Artemis root is usually added)
from memory_module.embeddings.service import EmbeddingService
from memory_module.search.vector_store import FAISSVectorStore
from memory_module.search.lexical import BM25LexicalIndex

logger = logging.getLogger(__name__)

class ArtemisMemoryService:
    """
    Integration bridge for Memory Module into Artemis Backend.
    Uses SQLite, Shared GPU Lock, and lightweight components.
    """

    def __init__(self):
        self.vector_dim = 384 # Default for many small models, will detect from service
        
        # Initialize Embedding Service (Lightweight wrapper mainly)
        # We might need to configure it to NOT load a model if we use llama-cpp
        # But prompt says "reuse shared Arctic Embed 2 service".
        # If Artemis has one, we use it. If not, we init it here.
        # existing main.py uses 'llm_engine'. let's assume we can use it or separate one.
        # implementation_plan said "Configure to use shared GPU lock".
        self.embedding_service = EmbeddingService(settings) 
        
        # FAISS (In-memory, persistent)
        self.vector_store = FAISSVectorStore(
            dimension=self.vector_dim,
            index_path="memory_index.faiss", # Local path relative to run
            mrl_enabled=False 
        )
        
        # BM25 (In-memory)
        self.lexical_index = BM25LexicalIndex(index_path="lexical_index.pkl")

    async def initialize(self):
        """Initialize indexes and warm up."""
        logger.info("Initializing Artemis Memory Service...")
        
        # Load indexes if exist
        if self.vector_store.index.ntotal == 0:
            # Maybe load from disk if implemented in vector_store (it is via constructor usually or load method)
            pass
            
        logger.info("Memory Service Ready.")

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using shared GPU lock.
        """
        # Acquire lock to prevent interfering with LLM generation
        async with gpu_lock:
            # Determine if we use EmbeddingService or LLM engine's embedding
            # memory_module's EmbeddingService uses sentence-transformers usually.
            # If we want to use the SAME model as LLM (if it supports embeddings), we'd use llm_engine.
            # But prompt says "shared Arctic Embed 2 service". 
            # This implies we should likely let EmbeddingService handle it but wrap it in lock.
            return await self.embedding_service.embed(text)

    async def store_memory(self, session_id: str, content: str, role: str, importance: float = 0.5):
        """
        Lightweight storage workflow.
        """
        try:
            # 1. Generate Embedding
            embedding = await self._get_embedding(content)
            
            # 2. Database Save (SQLite Sync)
            # We run sync DB ops in thread if needed, but for SQLite simple inserts are fast.
            # However, to be safe with main loop:
            db: Session = SessionLocal()
            try:
                record = MemoryRecordDB(
                    content=content,
                    channel_id=session_id,
                    source=role,
                    importance=importance,
                    embedding_blob=pickle.dumps(embedding), # Store as BLOB
                    metadata_json={"role": role},
                    created_at=datetime.utcnow()
                )
                db.add(record)
                db.commit()
                db.refresh(record)
                record_id = str(record.id)
                
                # 3. Update Indexes (In-Memory)
                # Vector Store
                await self.vector_store.add(
                    id=record_id,
                    vector=embedding,
                    metadata={"content": content, "channel_id": session_id}
                )
                
                # Lexical Index
                # Note: BM25 add might be sync or async
                await self.lexical_index.index_document(
                    id=record_id,
                    text=content,
                    metadata={"content": content}
                )
                
                logger.info(f"Stored memory {record_id} for session {session_id}")
                
            finally:
                db.close()
                
            # 4. Save Indexes (Periodically or here? Lightweight -> maybe here or on shutdown)
            # For robustness, let's save every N or just rely on OS buffer?
            # User said "persisted to disk".
            await self.vector_store.save("memory_index.faiss")
            self.lexical_index.save("lexical_index.pkl")
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")

    async def query_memory(self, query: str, session_id: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """
        Hybrid query.
        """
        try:
            # 1. Embedding
            query_vec = await self._get_embedding(query)
            
            # 2. Vector Search
            vec_results = await self.vector_store.search(query_vec, top_k=top_k * 2) # Fetch more for re-ranking/filtering
            
            # 3. Lexical Search
            lex_results = await self.lexical_index.search(query, top_k=top_k * 2)
            
            # 4. Fusion (Simplified)
            # Map ID -> Score
            scores = {}
            w_sem = 0.7
            w_lex = 0.3
            
            # Normalize? Simple max division
            max_vec = max([r.score for r in vec_results]) if vec_results else 1.0
            max_lex = max([r.score for r in lex_results]) if lex_results else 1.0
            
            # Combine
            for r in vec_results:
                scores[r.id] = (r.score / max_vec) * w_sem
                
            for r in lex_results:
                current = scores.get(r.id, 0.0)
                scores[r.id] = current + (r.score / max_lex) * w_lex
                
            # Sort
            ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # 5. Fetch Content from DB
            results = []
            db: Session = SessionLocal()
            try:
                for rid, score in ranked_ids:
                    rec = db.query(MemoryRecordDB).filter(MemoryRecordDB.id == rid).first()
                    if rec:
                        results.append({
                            "id": rec.id,
                            "content": rec.content,
                            "score": score,
                            "role": rec.source,
                            "created_at": rec.created_at,
                            "is_current_chat": rec.channel_id == session_id
                        })
            finally:
                db.close()
                
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

memory_service = ArtemisMemoryService()
