"""
Pytest configuration and fixtures.
"""

import asyncio
import os
import tempfile
from typing import AsyncGenerator, Generator

import pytest
from httpx import AsyncClient

from memory_module.api.app import create_app
from memory_module.config.settings import Settings
from memory_module.service import MemoryService
from memory_module.storage.metadata import MetadataStore


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def temp_index_path() -> Generator[str, None, None]:
    """Create a temporary index directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir

    # Cleanup
    import shutil

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def test_settings(temp_db_path: str, temp_index_path: str) -> Settings:
    """Create test settings with temporary paths."""
    return Settings(
        database_url=f"sqlite+aiosqlite:///{temp_db_path}",
        faiss_index_path=temp_index_path,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model for testing
        vector_dim=384,
        batch_size=8,
        redis_url=None,  # Disable Redis for tests
        log_level="DEBUG",
    )


@pytest.fixture
async def metadata_store(test_settings: Settings) -> AsyncGenerator[MetadataStore, None]:
    """Create a metadata store for testing."""
    store = MetadataStore(test_settings.database_url)
    await store.init_db()
    yield store


@pytest.fixture
async def memory_service(test_settings: Settings) -> AsyncGenerator[MemoryService, None]:
    """Create a memory service for testing."""
    service = MemoryService(test_settings)
    await service.initialize()
    yield service


@pytest.fixture
async def test_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    app = create_app()

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_memory_data() -> dict:
    """Sample memory data for testing."""
    return {
        "content": "Paris is the capital of France",
        "metadata": {"category": "geography"},
        "memory_type": "semantic",
        "tags": ["geography", "europe", "fact"],
        "source": "test",
        "confidence": 0.95,
    }


@pytest.fixture
def sample_memories() -> list[dict]:
    """Multiple sample memories for batch testing."""
    return [
        {
            "content": "The Earth orbits around the Sun",
            "tags": ["astronomy", "science"],
        },
        {
            "content": "Python is a programming language",
            "tags": ["programming", "technology"],
        },
        {
            "content": "Tokyo is the capital of Japan",
            "tags": ["geography", "asia"],
        },
    ]
