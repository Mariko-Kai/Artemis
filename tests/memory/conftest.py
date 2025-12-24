"""
Pytest fixtures for memory unit tests.
"""

import pytest
import tempfile
import shutil
import os
from unittest.mock import AsyncMock, MagicMock

from memory_module.config.settings import Settings


@pytest.fixture
def temp_indices_dir():
    """Create a temporary directory for test indices."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_settings(temp_indices_dir):
    """Create test settings with temporary paths."""
    db_path = os.path.join(temp_indices_dir, "test.db").replace("\\", "/")
    return Settings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        faiss_index_path=temp_indices_dir,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        vector_dim=384,
        batch_size=8,
        redis_url=None,
        log_level="DEBUG",
        enable_mrl=False,
    )


@pytest.fixture
def mock_embedding_service():
    """Mock the embedding service."""
    mock = AsyncMock()
    mock.embed = AsyncMock(return_value=[0.1] * 384)
    mock.batch_embed = AsyncMock(side_effect=lambda texts: [[0.1] * 384 for _ in texts])
    return mock
