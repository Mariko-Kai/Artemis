"""
Arctic Embed 2 wrapper for sentence-transformers.

This module provides a clean interface for loading and using
Snowflake Arctic Embed models.
"""

import logging
from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ArcticEmbedModel:
    """Wrapper for Snowflake Arctic Embed models."""

    def __init__(self, model_name: str = "Snowflake/snowflake-arctic-embed-m", device: str = "cpu"):
        """
        Initialize the Arctic Embed model.

        Args:
            model_name: Hugging Face model identifier
            device: Device to run the model on (cpu/cuda)
        """
        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

        logger.info(f"Initializing Arctic Embed model: {model_name} on {device}")

    def _load_model(self) -> SentenceTransformer:
        """
        Lazy load the model.

        Returns:
            SentenceTransformer instance
        """
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Get dimension from first encoding
            test_embedding = self._model.encode("test", convert_to_numpy=True)
            self._dimension = len(test_embedding)
            logger.info(f"Model loaded successfully. Embedding dimension: {self._dimension}")

        return self._model

    @property
    def model(self) -> SentenceTransformer:
        """Get the model instance, loading if necessary."""
        return self._load_model()

    @property
    def dimension(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            Dimension of embedding vectors
        """
        if self._dimension is None:
            _ = self.model  # Trigger model loading
        assert self._dimension is not None
        return self._dimension

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        """
        Encode a single text into an embedding vector.

        Args:
            text: Input text
            normalize: Whether to L2-normalize the embedding

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def encode_batch(
        self, texts: List[str], batch_size: int = 32, normalize: bool = True
    ) -> List[List[float]]:
        """
        Encode multiple texts into embedding vectors.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize the embeddings

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()


# Global model instance cache
_model_cache: dict[str, ArcticEmbedModel] = {}


def get_arctic_model(
    model_name: str = "Snowflake/snowflake-arctic-embed-m", device: str = "cpu"
) -> ArcticEmbedModel:
    """
    Get or create a cached Arctic Embed model instance.

    Args:
        model_name: Hugging Face model identifier
        device: Device to run the model on

    Returns:
        ArcticEmbedModel instance
    """
    cache_key = f"{model_name}:{device}"
    if cache_key not in _model_cache:
        _model_cache[cache_key] = ArcticEmbedModel(model_name, device)
    return _model_cache[cache_key]
