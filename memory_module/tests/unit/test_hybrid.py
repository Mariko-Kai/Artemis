import pytest
import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from memory_module.search.hybrid import HybridSearchService
from memory_module.models import SearchResult, MemoryType
from memory_module.config.settings import Settings

@pytest.fixture
def mock_vector_store():
    store = AsyncMock()
    store.search.return_value = []
    return store

@pytest.fixture
def mock_lexical_index():
    index = AsyncMock()
    index.search.return_value = []
    return index

@pytest.fixture
def settings():
    return Settings(
        temporal_half_life_hours=2.0,
        default_hybrid_weight=0.5
    )

@pytest.mark.asyncio
async def test_hybrid_search_fusion(mock_vector_store, mock_lexical_index, settings):
    # Setup results
    id1 = uuid.uuid4()
    id2 = uuid.uuid4()
    
    vec_result = SearchResult(
        id=id1, content="vec", score=0.8, created_at=datetime.now(timezone.utc),
        memory_type=MemoryType.SEMANTIC, metadata={}
    )
    lex_result = SearchResult(
        id=id2, content="lex", score=10.0, created_at=datetime.now(timezone.utc),
        memory_type=MemoryType.SEMANTIC, metadata={}
    )
    
    mock_vector_store.search.return_value = [vec_result]
    mock_lexical_index.search.return_value = [lex_result]

    service = HybridSearchService(settings, mock_vector_store, mock_lexical_index)
    
    results = await service.search("query", [0.1], top_k=5)
    
    assert len(results) == 2
    # Verify both IDs are present
    ids = {r.id for r in results}
    assert id1 in ids
    assert id2 in ids

@pytest.mark.asyncio
async def test_normalization_logic(mock_vector_store, mock_lexical_index, settings):
    id1 = uuid.uuid4()
    id2 = uuid.uuid4()
    id3 = uuid.uuid4()

    # Vector results: 0.5 and 1.0 (range 0.5)
    v1 = SearchResult(id=id1, content="v1", score=1.0, created_at=datetime.now(timezone.utc), memory_type=MemoryType.SEMANTIC, metadata={})
    v2 = SearchResult(id=id2, content="v2", score=0.5, created_at=datetime.now(timezone.utc), memory_type=MemoryType.SEMANTIC, metadata={})
    
    # Lexical results: 10 and 20 (range 10)
    l1 = SearchResult(id=id1, content="v1", score=10.0, created_at=datetime.now(timezone.utc), memory_type=MemoryType.SEMANTIC, metadata={}) # Same ID as v1
    l2 = SearchResult(id=id3, content="l2", score=20.0, created_at=datetime.now(timezone.utc), memory_type=MemoryType.SEMANTIC, metadata={})
    
    mock_vector_store.search.return_value = [v1, v2]
    mock_lexical_index.search.return_value = [l1, l2]
    
    service = HybridSearchService(settings, mock_vector_store, mock_lexical_index)
    results = await service.search("q", [], 5, hybrid_weight=0.5)
    
    # Result 1 (id1): Matches both
    # v1 norm = (1.0-0.5)/0.5 = 1.0
    # l1 norm = (10-10)/10 = 0.0
    # Score = 0.5*1.0 + 0.5*0.0 = 0.5
    
    r_dict = {str(r.id): r for r in results}
    
    # Use pytest.approx because of float precision
    assert r_dict[str(id1)].score == pytest.approx(0.5, abs=0.01)
    
@pytest.mark.asyncio
async def test_temporal_boosting(mock_vector_store, mock_lexical_index, settings):
    id1 = uuid.uuid4()
    id2 = uuid.uuid4()
    
    now = datetime.now(timezone.utc)
    # Item 1: New (0 hours old)
    r1 = SearchResult(id=id1, content="new", score=0.9, created_at=now, memory_type=MemoryType.SEMANTIC, metadata={})
    # Item 2: Old (2 hours old = 1 half-life)
    r2 = SearchResult(id=id2, content="old", score=0.9, created_at=now - timedelta(hours=2), memory_type=MemoryType.SEMANTIC, metadata={})
    
    mock_vector_store.search.return_value = [r1, r2]
    
    service = HybridSearchService(settings, mock_vector_store, mock_lexical_index)
    results = await service.search("q", [], 5, hybrid_weight=1.0) # Pure vector
    
    r_dict = {str(r.id): r for r in results}
    
    # Both items have identical raw score 0.9.
    # Logic fallback to 0.5 if range < 0.001 (which is true here).
    # So base score = 0.5 for both.
    # r1 boost: age=0 -> boost=1.0. Final = 0.5 * 1.0 = 0.5.
    # r2 boost: age=2, half-life=2 -> boost=0.5. Final = 0.5 * 0.5 = 0.25.
    
    assert r_dict[str(id1)].score > r_dict[str(id2)].score
