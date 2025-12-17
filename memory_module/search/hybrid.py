"""
Hybrid search service implementation.

Combines semantic search (Vector/FAISS) and lexical search (BM25) 
with weighted score fusion and temporal boosting.
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..config.settings import Settings
from ..interfaces import ILexicalIndex, IVectorStore
from ..models import SearchResult

logger = logging.getLogger(__name__)


class HybridSearchService:
    """Orchestrator for hybrid search operations."""

    def __init__(
        self,
        settings: Settings,
        vector_store: IVectorStore,
        lexical_index: ILexicalIndex,
    ):
        """
        Initialize hybrid search service.

        Args:
            settings: Configuration settings
            vector_store: Vector store instance
            lexical_index: Lexical index instance
        """
        self.settings = settings
        self.vector_store = vector_store
        self.lexical_index = lexical_index
        
    async def search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        hybrid_weight: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search.

        Args:
            query_text: Raw query text
            query_vector: Embedding of query text
            top_k: Number of results
            filters: Metadata filters
            hybrid_weight: Alpha weight (0.0=lexical only, 1.0=vector only).
                           If None, uses default from settings.
        
        Returns:
            List of SearchResult
        """
        weight = hybrid_weight if hybrid_weight is not None else self.settings.default_hybrid_weight
        
        # Parallel execution of both searches
        # Note: We ask for slightly more results from each to ensure good intersection/fusion
        fetch_k = top_k * 2
        
        tasks = []
        
        # 1. Vector Search
        if weight > 0:
            tasks.append(self.vector_store.search(query_vector, top_k=fetch_k, filters=filters))
        else:
            tasks.append(asyncio.sleep(0, result=[])) # No-op
            
        # 2. Lexical Search
        if weight < 1:
            tasks.append(self.lexical_index.search(query_text, top_k=fetch_k))
        else:
            tasks.append(asyncio.sleep(0, result=[])) # No-op

        # Await results
        vector_results, lexical_results = await asyncio.gather(*tasks)
        
        # 3. Score Normalization & Fusion
        combined_results = self._fuse_results(
            vector_results, 
            lexical_results, 
            weight,
            top_k
        )
        
        return combined_results

    def _fuse_results(
        self,
        vector_results: List[SearchResult],
        lexical_results: List[SearchResult],
        weight: float,
        top_k: int
    ) -> List[SearchResult]:
        """
        Combine and rank results using Reciprocal Rank Fusion (RRF) or Linear Combination.
        Implementing Linear Combination with MinMax Normalization here as requested.
        """
        
        # Maps to hold scores
        # doc_id -> {sem_score, lex_score, result_obj}
        all_results: Dict[str, Dict[str, Any]] = {}
        
        # 1. Normalize Vector Scores (Cosine Similarity is -1 to 1, usually 0-1 for text)
        sem_scores = [r.score for r in vector_results]
        max_sem = max(sem_scores) if sem_scores else 1.0
        min_sem = min(sem_scores) if sem_scores else 0.0
        
        sem_range = max_sem - min_sem
        
        for r in vector_results:
            # Normalize to 0-1 range relative to this batch
            if sem_range > 0.001:
                norm_score = (r.score - min_sem) / sem_range
            else:
                 # If variance is low, fallback to raw score if it's <= 1, else 1.0?
                 # Cosine allows -1 to 1. If all are 0.8, using 0.8 is better than 0.
                 # But we need to ensure it doesn't break lexical scale.
                 norm_score = r.score if 0 <= r.score <= 1 else 0.5 
            
            all_results[str(r.id)] = {
                "result": r,
                "sem_score": norm_score,
                "lex_score": 0.0
            }
            # Populate fields
            r.semantic_score = norm_score

        # 2. Normalize Lexical Scores (BM25 is unbounded 0 to inf)
        lex_scores = [r.score for r in lexical_results]
        max_lex = max(lex_scores) if lex_scores else 1.0
        min_lex = min(lex_scores) if lex_scores else 0.0
        lex_range = max_lex - min_lex

        for r in lexical_results:
            rid = str(r.id)
            
            if lex_range > 0.001:
                norm_score = (r.score - min_lex) / lex_range
            else:
                # BM25 single result or identical.
                # If unbounded, we can't trust the raw value as "0-1" probability.
                # But fusion needs comparison.
                # Let's assign 1.0 if score > 0 else 0.0
                norm_score = 1.0 if r.score > 0 else 0.0
            
            if rid in all_results:
                # Merge
                all_results[rid]["lex_score"] = norm_score
                # Update the existing object
                all_results[rid]["result"].lexical_score = norm_score
            else:
                all_results[rid] = {
                    "result": r,
                    "sem_score": 0.0,
                    "lex_score": norm_score
                }
                r.lexical_score = norm_score
                r.semantic_score = 0.0

        # 3. Calculate Final Score + Temporal Boost
        final_list = []
        now = datetime.now(timezone.utc)
        
        for rid, data in all_results.items():
            r = data["result"]
            sem = data["sem_score"]
            lex = data["lex_score"]
            
            # Weighted average
            base_score = (weight * sem) + ((1.0 - weight) * lex)
            
            # Temporal Boost
            # Boost recent memories.
            # Decay factor: 1 / (1 + (age_hours / half_life)^2) 
            # or simple exponential e^(-lambda * t)
            
            age_hours = 0.0
            if r.created_at:
                # Ensure timezone awareness
                created_at = r.created_at
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                
                delta = now - created_at
                age_hours = max(0, delta.total_seconds() / 3600)
            
            half_life = self.settings.temporal_half_life_hours
            if half_life > 0:
                temporal_boost = 1.0 / (1.0 + (age_hours / half_life))
            else:
                temporal_boost = 1.0
                
            # Validating boost isn't too strong? 
            # Let's say we want to multiply score, or add to it.
            # Usually strict retrieval score shouldn't be overridden by recency too much.
            # Let's use it as a multiplier.
            
            final_score = base_score * temporal_boost
            
            result = r.model_copy()
            result.score = final_score
            result.temporal_score = temporal_boost
            
            final_list.append(result)
            
        # 4. Sort and Truncate
        final_list.sort(key=lambda x: x.score, reverse=True)
        return final_list[:top_k]
