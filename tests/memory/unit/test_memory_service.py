
import pytest
from unittest.mock import MagicMock, AsyncMock
from memory_module.service import MemoryService
from uuid import uuid4
from datetime import datetime
import memory_module.models as mmodels

@pytest.fixture
def memory_service(test_settings, mock_embedding_service):
    """Create MemoryService with mocked components"""
    service = MemoryService(settings=test_settings)
    service.embedding_service = mock_embedding_service
    
    # Mock all internal components
    service.vector_store = AsyncMock()
    service.lexical_index = AsyncMock()
    service.metadata_store = AsyncMock()
    # We mock hybrid_service too but make sure it works for our tests
    service.hybrid_service = AsyncMock()
    
    return service

@pytest.mark.asyncio
async def test_store_memory(memory_service):
    """Test storing a memory"""
    # Configure metadata_store.save to return a real MemoryRecord
    target_id = uuid4()
    mock_record = mmodels.MemoryRecord(
        id=target_id,
        content="Test memory",
        metadata={"role": "user"},
        status=mmodels.MemoryStatus.PENDING_SUMMARY
    )
    memory_service.metadata_store.save = AsyncMock(return_value=mock_record)
    
    result = await memory_service.store_memory(
        content="Test memory",
        metadata={"role": "user"}
    )
    
    assert result.id == target_id
    assert isinstance(result, mmodels.MemoryRecord)
    assert not memory_service.embedding_service.embed.called
    assert result.status == mmodels.MemoryStatus.PENDING_SUMMARY
    assert memory_service.metadata_store.save.called

@pytest.mark.asyncio
async def test_search_memory(memory_service):
    """Test hybrid search"""
    from uuid import uuid4
    from datetime import datetime
    
    # Mock return value to be SearchResult objects
    mock_results = [
        mmodels.SearchResult(
            id=uuid4(), 
            content="test", 
            score=0.9, 
            memory_type=mmodels.MemoryType.SEMANTIC,
            created_at=datetime.now(),
            metadata={}
        )
    ]
    
    # The service calls hybrid_service.search
    memory_service.hybrid_service.search = AsyncMock(return_value=mock_results)
    
    request = mmodels.QueryRequest(query="test", top_k=1)
    response = await memory_service.search_memories(request)
    
    assert len(response.results) > 0
    assert response.results[0].content == "test"
    assert memory_service.hybrid_service.search.called
