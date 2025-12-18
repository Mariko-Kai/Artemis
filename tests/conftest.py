
import pytest
import pytest_asyncio
import os
import sys
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import httpx
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

# Mock missing heavy dependencies before importing app modules
import types
from unittest.mock import MagicMock

def mock_heavy_deps(names):
    for name in names:
        parts = name.split('.')
        for i in range(len(parts)):
            pkg_name = '.'.join(parts[:i+1])
            if pkg_name not in sys.modules:
                m = MagicMock()
                sys.modules[pkg_name] = m

mock_heavy_deps([
    "faster_whisper",
    "sentence_transformers",
    "redis",
    "faiss",
    "langchain",
    "langchain_core.tools",
    "langchain_core.language_models",
    "langchain_community.tools",
    "langchain_community.vectorstores",
    "langchain_openai",
    "playwright.async_api",
    "llama_cpp"
])

from app.main import app
from app.core.config import settings
from app.db.database import Base, get_db
from app.core.global_lock import gpu_lock
from app.core.llm_engine import llm_engine
from app.services.memory_service import memory_service

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_env():
    """Setup environment variables for testing"""
    os.environ["TESTING"] = "True"
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///"
    # Disable background tasks that might interfere
    settings.MEMORY_SUMMARIZATION_ENABLED = False
    settings.MEMORY_ARCHIVAL_ENABLED = False
    yield

@pytest_asyncio.fixture(scope="function")
async def test_db():
    """Create a fresh in-memory database for each test (Synchronous)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Use sync engine for compatibility with existing service code
    engine = create_engine(
        "sqlite:///",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Patch SessionLocal and get_db in the app
    # get_db might need to remain compatible with Depends()
    with patch("app.db.database.SessionLocal", session_factory), \
         patch("app.db.database.engine", engine):
        
        def override_get_db():
            db = session_factory()
            try:
                yield db
            finally:
                db.close()
        
        app.dependency_overrides[get_db] = override_get_db
        
        # Also patch services that use SessionLocal directly
        with patch("app.services.memory_service.SessionLocal", session_factory), \
             patch("app.services.deferred_processing_service.SessionLocal", session_factory):
            yield session_factory
            
        app.dependency_overrides.clear()

@pytest_asyncio.fixture(scope="function")
async def client(test_db):
    """Async client for integration testing."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

@pytest.fixture(scope="function")
def gpu_spy():
    """Spy on the GPU lock to verify acquisition and release."""
    with patch("app.core.global_lock.gpu_lock.request_priority_access", wraps=gpu_lock.request_priority_access) as priority_spy, \
         patch("app.core.global_lock.gpu_lock.background", wraps=gpu_lock.background) as background_spy:
        yield {"priority": priority_spy, "background": background_spy}

@pytest.fixture(scope="function", autouse=True)
def mock_llm():
    """Mock the LLM engine to avoid loading heavy models."""
    mock_model = MagicMock()
    mock_model.create_completion = MagicMock(return_value={
        "choices": [{"text": "This is a mock summary."}]
    })
    mock_model.create_chat_completion = MagicMock(return_value={
        "choices": [{"message": {"content": "This is a mock chat response."}}]
    })
    
    with patch.object(llm_engine, "get_model", return_value=mock_model), \
         patch.object(llm_engine, "load_model", return_value=None):
        yield mock_model

@pytest.fixture(scope="function")
def mock_embeddings():
    """Mock the Embedding Service."""
    with patch("app.services.memory_service.EmbeddingService") as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.embed = AsyncMock(return_value=[0.1] * settings.vector_dim)
        mock_instance.batch_embed = AsyncMock(side_effect=lambda texts: [[0.1] * settings.vector_dim for _ in texts])
        
        # Also patch ArcticEmbedService used in deferred_processing_service
        with patch("app.services.deferred_processing_service.EmbeddingService", return_value=mock_instance):
            yield mock_instance

@pytest.fixture(scope="function", autouse=True)
def mock_vector_store():
    """Mock FAISS and Lexical index to avoid disk IO."""
    with patch.object(memory_service.vector_store, "add", AsyncMock()), \
         patch.object(memory_service.vector_store, "search", AsyncMock(return_value=[])), \
         patch.object(memory_service.vector_store, "save", AsyncMock()), \
         patch.object(memory_service.lexical_index, "index_document", AsyncMock()), \
         patch.object(memory_service.lexical_index, "save", MagicMock()):
        yield

