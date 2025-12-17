"""
Unit tests for Pydantic models.
"""

import pytest
from pydantic import ValidationError

from memory_module.models import (
    MemoryRecord,
    MemoryType,
    QueryRequest,
    QueryResponse,
    SearchResult,
    StoreRequest,
)


def test_memory_record_creation() -> None:
    """Test basic MemoryRecord creation."""
    memory = MemoryRecord(
        content="Test content",
        tags=["test", "sample"],
    )

    assert memory.content == "Test content"
    assert "test" in memory.tags
    assert "sample" in memory.tags
    assert memory.memory_type == MemoryType.SEMANTIC
    assert memory.id is not None
    assert memory.created_at is not None


def test_memory_record_validation() -> None:
    """Test MemoryRecord validation."""
    # Empty content should fail
    with pytest.raises(ValidationError):
        MemoryRecord(content="")

    # Whitespace-only content should fail
    with pytest.raises(ValidationError):
        MemoryRecord(content="   ")


def test_memory_record_tags_deduplication() -> None:
    """Test that tags are deduplicated."""
    memory = MemoryRecord(
        content="Test",
        tags=["tag1", "tag2", "tag1", "tag2"],
    )

    assert len(memory.tags) == 2
    assert "tag1" in memory.tags
    assert "tag2" in memory.tags


def test_query_request_validation() -> None:
    """Test QueryRequest validation."""
    # Valid request
    request = QueryRequest(query="test query")
    assert request.query == "test query"
    assert request.top_k == 10
    assert request.hybrid_weight == 0.5

    # Custom values
    request = QueryRequest(
        query="custom query",
        top_k=20,
        hybrid_weight=0.7,
    )
    assert request.top_k == 20
    assert request.hybrid_weight == 0.7

    # Invalid top_k (too large)
    with pytest.raises(ValidationError):
        QueryRequest(query="test", top_k=200)

    # Invalid hybrid_weight (out of range)
    with pytest.raises(ValidationError):
        QueryRequest(query="test", hybrid_weight=1.5)


def test_store_request_creation() -> None:
    """Test StoreRequest creation."""
    request = StoreRequest(
        content="Test memory",
        metadata={"key": "value"},
        tags=["tag1", "tag2"],
        source="test_source",
        confidence=0.9,
    )

    assert request.content == "Test memory"
    assert request.metadata == {"key": "value"}
    assert "tag1" in request.tags
    assert request.source == "test_source"
    assert request.confidence == 0.9


def test_search_result_creation() -> None:
    """Test SearchResult creation."""
    from datetime import datetime
    from uuid import uuid4

    result = SearchResult(
        id=uuid4(),
        content="Test content",
        score=0.85,
        metadata={"category": "test"},
        memory_type=MemoryType.SEMANTIC,
        tags=["test"],
        created_at=datetime.utcnow(),
    )

    assert result.score == 0.85
    assert result.memory_type == MemoryType.SEMANTIC


def test_query_response_creation() -> None:
    """Test QueryResponse creation."""
    response = QueryResponse(
        results=[],
        query="test query",
        total_results=0,
        query_time_ms=42.5,
        hybrid_weight=0.5,
    )

    assert response.query == "test query"
    assert response.total_results == 0
    assert response.query_time_ms == 42.5
