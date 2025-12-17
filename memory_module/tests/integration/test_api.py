"""
Integration tests for API endpoints.
"""

import pytest
from httpx import AsyncClient

from memory_module.models import MemoryType


@pytest.mark.integration
async def test_health_check(test_client: AsyncClient) -> None:
    """Test health check endpoint."""
    response = await test_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "memory-module"


@pytest.mark.integration
async def test_store_memory(test_client: AsyncClient, sample_memory_data: dict) -> None:
    """Test storing a memory via API."""
    response = await test_client.post(
        "/api/v1/memories",
        json=sample_memory_data,
    )

    assert response.status_code == 201

    data = response.json()
    assert data["content"] == sample_memory_data["content"]
    assert "id" in data
    assert data["memory_type"] == sample_memory_data["memory_type"]


@pytest.mark.integration
async def test_get_memory(test_client: AsyncClient, sample_memory_data: dict) -> None:
    """Test retrieving a memory by ID."""
    # First, store a memory
    store_response = await test_client.post(
        "/api/v1/memories",
        json=sample_memory_data,
    )
    assert store_response.status_code == 201
    memory_id = store_response.json()["id"]

    # Then retrieve it
    get_response = await test_client.get(f"/api/v1/memories/{memory_id}")
    assert get_response.status_code == 200

    data = get_response.json()
    assert data["id"] == memory_id
    assert data["content"] == sample_memory_data["content"]


@pytest.mark.integration
async def test_delete_memory(test_client: AsyncClient, sample_memory_data: dict) -> None:
    """Test deleting a memory."""
    # Store a memory
    store_response = await test_client.post(
        "/api/v1/memories",
        json=sample_memory_data,
    )
    memory_id = store_response.json()["id"]

    # Delete it
    delete_response = await test_client.delete(f"/api/v1/memories/{memory_id}")
    assert delete_response.status_code == 204

    # Verify it's gone
    get_response = await test_client.get(f"/api/v1/memories/{memory_id}")
    assert get_response.status_code == 404


@pytest.mark.integration
async def test_search_memories(test_client: AsyncClient, sample_memories: list) -> None:
    """Test searching memories."""
    # Store multiple memories
    for memory_data in sample_memories:
        await test_client.post("/api/v1/memories", json=memory_data)

    # Search for astronomy-related content
    search_response = await test_client.post(
        "/api/v1/search",
        json={
            "query": "astronomy space science",
            "top_k": 5,
            "hybrid_weight": 0.7,
        },
    )

    assert search_response.status_code == 200

    data = search_response.json()
    assert "results" in data
    assert "query_time_ms" in data
    assert data["query"] == "astronomy space science"


@pytest.mark.integration
async def test_batch_store(test_client: AsyncClient, sample_memories: list) -> None:
    """Test batch storing memories."""
    batch_request = {"memories": sample_memories}

    response = await test_client.post("/api/v1/memories/batch", json=batch_request)

    assert response.status_code == 201

    data = response.json()
    assert len(data) == len(sample_memories)
    assert all("id" in memory for memory in data)


@pytest.mark.integration
async def test_get_stats(test_client: AsyncClient, sample_memory_data: dict) -> None:
    """Test getting statistics."""
    # Store a memory first
    await test_client.post("/api/v1/memories", json=sample_memory_data)

    # Get stats
    response = await test_client.get("/api/v1/stats")
    assert response.status_code == 200

    data = response.json()
    assert "total_memories" in data
    assert "memory_types_count" in data
    assert "vector_dimension" in data
    assert data["total_memories"] >= 1
