
import pytest
from datetime import datetime
from memory_module.storage.models import MemoryRecordDB, VectorMapping

@pytest.mark.asyncio
async def test_memory_record_creation(db_session):
    """Test creating a memory record"""
    from sqlalchemy import select
    record = MemoryRecordDB(
        content="Test content",
        memory_type="semantic",
        source="user",
        importance=0.8,
        record_metadata={"tag": "test"}
    )
    db_session.add(record)
    await db_session.commit()
    
    assert record.id is not None
    assert record.created_at is not None
    assert record.content == "Test content"
    
    # Retrieve
    stmt = select(MemoryRecordDB).filter_by(id=record.id)
    result = await db_session.execute(stmt)
    saved = result.scalar_one_or_none()
    assert saved is not None
    assert saved.record_metadata["tag"] == "test"

@pytest.mark.asyncio
async def test_vector_mapping(db_session):
    """Test vector mapping relationship"""
    record = MemoryRecordDB(content="Vector test")
    db_session.add(record)
    await db_session.commit()
    
    mapping = VectorMapping(
        memory_id=record.id,
        faiss_id=123,
        embedding_model_version="test-1.0"
    )
    db_session.add(mapping)
    await db_session.commit()
    
    assert mapping.memory_id == record.id
