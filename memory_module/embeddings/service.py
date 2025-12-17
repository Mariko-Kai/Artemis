"""
Embedding service implementation using Arctic Embed.

Implements the IEmbeddingService interface with caching support.
"""

import logging
from functools import lru_cache
from typing import List

from ..config.settings import Settings
from ..interfaces import IEmbeddingService
from .arctic import get_arctic_model

logger = logging.getLogger(__name__)


class EmbeddingService(IEmbeddingService):
    """Embedding service using Snowflake Arctic Embed."""

    def __init__(self, settings: Settings):
        """
        Initialize the embedding service.

        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.model = get_arctic_model(
            model_name=settings.embedding_model_name,
            device=settings.embedding_device,
        )
        logger.info("Embedding service initialized")

    async def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        # Use cached version for duplicate texts
        return self._embed_cached(text)

    @lru_cache(maxsize=1000)
    def _embed_cached(self, text: str) -> List[float]:
        """
        Cached embedding generation.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return self.model.encode(text)

    async def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        return self.model.encode_batch(texts, batch_size=self.settings.batch_size)

    def get_dimension(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            Dimension of embedding vectors
        """
        return self.model.dimension
