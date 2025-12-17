"""
Unit tests for ArcticEmbedService.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from memory_module.embeddings.arctic import ArcticEmbedService

@pytest.fixture
def mock_settings():
    with patch("memory_module.embeddings.arctic.get_settings") as mock:
        mock.return_value.embedding_model_name = "test-model"
        mock.return_value.embedding_device = "cpu"
        mock.return_value.vector_dim = 16
        mock.return_value.redis_url = None
        mock.return_value.cache_ttl = 3600
        yield mock

@pytest.fixture
def service(mock_settings):
    # Mock SentenceTransformer to avoid loading real model
    with patch("memory_module.embeddings.arctic.SentenceTransformer") as mock_st:
        # Configure mock model encoding
        mock_model = MagicMock()
        # Return random vectors of size 16
        def side_effect_encode(texts, **kwargs):
            if isinstance(texts, str):
                return np.random.rand(16)
            return np.random.rand(len(texts), 16)
            
        mock_model.encode.side_effect = side_effect_encode
        mock_st.return_value = mock_model
        
        service = ArcticEmbedService()
        yield service

@pytest.mark.asyncio
async def test_embed_dimension(service):
    text = "test"
    vector = await service.embed(text)
    assert len(vector) == 16
    
@pytest.mark.asyncio
async def test_embed_truncation(service):
    text = "test"
    # Truncate to 8
    vector = await service.embed(text, dimension=8)
    assert len(vector) == 8
    
    # Check normalization
    norm = np.linalg.norm(vector)
    assert np.isclose(norm, 1.0) or norm == 0

@pytest.mark.asyncio
async def test_batch_embed(service):
    texts = ["one", "two"]
    vectors = await service.batch_embed(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == 16

@pytest.mark.asyncio
async def test_batch_embed_truncation(service):
    texts = ["one", "two"]
    vectors = await service.batch_embed(texts, dimension=4)
    assert len(vectors) == 2
    assert len(vectors[0]) == 4
