
import pytest
from app.db.memory_models import MemoryRecordDB
from app.services.deferred_processing_service import deferred_processing_service
from app.core.config import settings
from datetime import datetime
import asyncio

@pytest.mark.asyncio
async def test_memory_workflow_lifecycle(client, test_db, mock_llm, mock_embeddings):
    """
    Verify the full data lifecycle: Hot Store -> Idle Summarization -> Cold embedding/indexing.
    """
    # Step 1: Hot Store via API
    memory_data = {
        "content": "This is a test memory about integration testing.",
        "session_id": "test-session-123",
        "role": "user",
        "importance": 0.8
    }
    
    response = await client.post("/v1/memory/store", json=memory_data)
    assert response.status_code == 200
    
    # Verify status in DB
    with test_db() as session:
        from sqlalchemy import select
        record = session.query(MemoryRecordDB).filter_by(channel_id="test-session-123").one()
        assert record.status == "pending_summary"
        assert record.content == memory_data["content"]
        record_id = record.id

    # Step 2: Idle Processing - Summarization
    # Manually trigger the summarization task
    await deferred_processing_service.task_summarize_pending()
    
    # Verify status changed to pending_embedding and summary is present
    with test_db() as session:
        record = session.query(MemoryRecordDB).filter_by(id=record_id).one()
        assert record.status == "pending_embedding"
        assert record.summary == "This is a mock summary." # From mock_llm in conftest

    # Step 3: Cold Store - Embedding and Indexing
    # Manually trigger the indexing task
    await deferred_processing_service.task_index_pending()
    
    # Verify status changed to completed
    with test_db() as session:
        record = session.query(MemoryRecordDB).filter_by(id=record_id).one()
        assert record.status == "completed"
        assert record.embedding_blob is not None
        
    # Verify mock calls
    mock_embeddings.embed.assert_called()
