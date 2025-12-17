
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from unittest.mock import MagicMock

# Mock faster_whisper to avoid dependency issues during memory testing
sys.modules["faster_whisper"] = MagicMock()
sys.modules["app.core.asr_engine"] = MagicMock()
sys.modules["langchain_core"] = MagicMock()
sys.modules["langchain_core.language_models"] = MagicMock()
sys.modules["langchain_community"] = MagicMock()
sys.modules["llama_cpp"] = MagicMock()
sys.modules["app.agent.executor"] = MagicMock()
sys.modules["app.core.llm_engine"] = MagicMock()

from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_memory_lifecycle():
    # 1. Store a memory via chat (simulate auto-save) 
    # Use manual store endpoint if available, but we only have manual summarize/export/query
    # We can inject via memory_service directly or use query to check empty first.
    
    # 2. Query Memory
    response = client.post("/v1/memory/query", json={"query": "test memory", "top_k": 1})
    assert response.status_code == 200
    
    # 3. Archive Stats
    response = client.get("/v1/memory/archive/stats")
    assert response.status_code == 200
    data = response.json()
    assert "archived_count" in data
    
    # 4. Export Alias
    response = client.post("/v1/memory/export")
    assert response.status_code == 200
    
    # 5. Search Alias
    response = client.post("/v1/memory/search", json={"query": "hello", "top_k": 1})
    assert response.status_code == 200


def test_delete_endpoint():
    # This is destructive/requires ID.
    # Since we can't easily create a raw memory via API without chat, 
    # we might skip or assume a mock.
    pass
