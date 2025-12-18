
import pytest
import pytest_asyncio
import os
import sys
import shutil
import asyncio
from unittest.mock import MagicMock, AsyncMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import memory_module
print(f"DEBUG: memory_module file={memory_module.__file__}")
from memory_module.storage.models import Base
from memory_module.config.settings import Settings

print(f"DEBUG: sys.path[0]={sys.path[0]}")
print(f"DEBUG: MM_DATABASE_URL env={os.environ.get('MM_DATABASE_URL')}")

@pytest_asyncio.fixture(scope="session")
async def test_settings():
    """Mock settings for testing"""
    return Settings(
        MM_DATABASE_URL="sqlite+aiosqlite:///", # In-memory SQLite
        MM_EMBEDDING_DEVICE="cpu",
        MM_FAISS_INDEX_PATH="./test_faiss_index",
        MM_BM25_INDEX_PATH="./test_bm25_index.pkl",
        MM_LOG_LEVEL="DEBUG"
    )

@pytest.fixture(scope="function")
def mock_embedding_service():
    """Mock embedding service to avoid loading model"""
    service = MagicMock()
    # Arctic Embed M is 768.
    service.get_dimension.return_value = 768
    
    # Use AsyncMock for async methods
    service.embed = AsyncMock(return_value=[0.1] * 768)
    service.batch_embed = AsyncMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    
    # Also support older names if any code still uses them
    service.generate_embedding = AsyncMock(return_value=[0.1] * 768)
    service.generate_embeddings = AsyncMock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    
    return service

@pytest_asyncio.fixture(scope="function")
async def db_session(test_settings):
    """Async in-memory SQLite session for testing"""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    
    engine = create_async_engine(
        "sqlite+aiosqlite:///",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
            # We don't necessarily need to drop_all for in-memory DB as it dies with engine
            await engine.dispose()

@pytest.fixture(scope="function")
def temp_indices_dir(tmp_path):
    """Use pytest's tmp_path for indices to avoid PermissionError on Windows"""
    return str(tmp_path)

@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
