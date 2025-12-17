"""
Arctic Embed 2 wrapper for sentence-transformers with MRL support.

This module provides a clean interface for loading and using
Snowflake Arctic Embed models with caching, retries, and MRL optimization.
"""

import asyncio
import hashlib
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import redis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sentence_transformers import SentenceTransformer

from ..interfaces import IEmbeddingService
from ..config.settings import get_settings

logger = logging.getLogger(__name__)

class ArcticEmbedService(IEmbeddingService):
    """
    Embedding service using Snowflake Arctic Embed models.
    
    Features:
    - Async support
    - Redis caching
    - MRL (Matryoshka Representation Learning) truncation
    - Automatic retries
    """

    def __init__(self):
        """Initialize the service."""
        settings = get_settings()
        self.model_name = settings.embedding_model_name
        self.device = settings.embedding_device
        self.default_dim = settings.vector_dim
        self.redis_url = settings.redis_url
        self.cache_ttl = settings.cache_ttl
        
        self._model: Optional[SentenceTransformer] = None
        self._redis: Optional[redis.Redis] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Initialize Redis if configured
        if self.redis_url:
            try:
                self._redis = redis.from_url(self.redis_url)
                self._redis.ping()
                logger.info(f"Connected to Redis at {self.redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Caching disabled.")
                self._redis = None

    def _load_model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Warmup
            _ = self._model.encode("warmup")
            logger.info("Model loaded successfully")
        return self._model

    @property
    def model(self) -> SentenceTransformer:
        """Get the model instance, loading if necessary."""
        return self._load_model()

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.default_dim

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"embed:{self.model_name}:{text_hash}"

    def _truncate_vector(self, vector: List[float], dim: Optional[int]) -> List[float]:
        """Truncate vector for MRL."""
        if dim and dim < len(vector):
            # For MRL, valid truncation is just slicing the first n components
            # and re-normalizing (though Arctic is trained such that slicing is enough, 
            # re-normalization usually recommended for cosine similarity)
            vec_np = np.array(vector[:dim])
            norm = np.linalg.norm(vec_np)
            if norm > 0:
                vec_np = vec_np / norm
            return vec_np.tolist()
        return vector

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _encode_sync(self, text: str) -> List[float]:
        """Synchronous encoding logic with caching."""
        if self._redis:
            key = self._get_cache_key(text)
            cached = self._redis.get(key)
            if cached:
                return pickle.loads(cached)

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        if self._redis:
            try:
                # Store full embedding
                self._redis.setex(
                    self._get_cache_key(text),
                    self.cache_ttl,
                    pickle.dumps(embedding)
                )
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {e}")

        return embedding

    async def embed(self, text: str, dimension: Optional[int] = None) -> List[float]:
        """
        Generate embedding vector for a single text.

        Args:
            text: Input text
            dimension: Optional output dimension (MRL truncation)

        Returns:
            Embedding vector
        """
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(self._executor, self._encode_sync, text)
        
        target_dim = dimension or self.default_dim
        return self._truncate_vector(vector, target_dim)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _encode_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch encoding logic with caching."""
        results: List[Optional[List[float]]] = [None] * len(texts)
        texts_to_encode = []
        indices_to_encode = []

        # Check cache first
        if self._redis:
            for i, text in enumerate(texts):
                key = self._get_cache_key(text)
                cached = self._redis.get(key)
                if cached:
                    results[i] = pickle.loads(cached)
                else:
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
        else:
            texts_to_encode = texts
            indices_to_encode = list(range(len(texts)))

        if texts_to_encode:
            embeddings = self.model.encode(
                texts_to_encode,
                batch_size=32, # Configurable?
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()

            # Fill results and cache
            for idx_in_batch, original_idx in enumerate(indices_to_encode):
                embedding = embeddings[idx_in_batch]
                results[original_idx] = embedding
                
                if self._redis:
                    try:
                        self._redis.setex(
                            self._get_cache_key(texts[original_idx]),
                            self.cache_ttl,
                            pickle.dumps(embedding)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to cache batch embedding: {e}")

        # results should be fully populated now (assuming no logic errors)
        assert all(r is not None for r in results)
        return results # type: ignore

    async def batch_embed(self, texts: List[str], dimension: Optional[int] = None) -> List[List[float]]:
        """
        Generate embedding vectors for multiple texts.

        Args:
            texts: List of input texts
            dimension: Optional output dimension (MRL truncation)

        Returns:
            List of embedding vectors
        """
        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(self._executor, self._encode_batch_sync, texts)
        
        target_dim = dimension or self.default_dim
        return [self._truncate_vector(v, target_dim) for v in vectors]


# Global instance cache
_service_instance: Optional[ArcticEmbedService] = None

def get_embedding_service() -> ArcticEmbedService:
    """Get or create scalar ArcticEmbedService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ArcticEmbedService()
    return _service_instance
