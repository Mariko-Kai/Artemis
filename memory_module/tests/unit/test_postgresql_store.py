
import pytest
from datetime import datetime
from uuid import uuid4
from unittest.mock import MagicMock, AsyncMock

from memory_module.models import MemoryRecord, MemoryType
from memory_module.storage.postgresql import PostgreSQLMetadataStore
from memory_module.config.settings import Settings

# Use sqlite for fast testing of logic
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture
async def store():
    store = PostgreSQLMetadataStore(TEST_DB_URL)
    await store.init_db()
    return store

@pytest.mark.asyncio
async def test_crud_operations(store):
    # Create
    record = MemoryRecord(content="Test content", memory_type=MemoryType.EPISODIC)
    saved = await store.save(record)
    assert saved.id == record.id
    assert saved.content_hash is not None

    # Read
    fetched = await store.get(str(saved.id))
    assert fetched.content == "Test content"
    
    # Update
    updated = await store.update(str(saved.id), {"content": "Updated content"})
    assert updated.content == "Updated content"
    assert updated.version == 2
    
    # Delete (soft)
    deleted = await store.delete(str(saved.id), soft=True)
    assert deleted is True
    
    # Verify soft delete
    # Query should exclude archived by default
    results = await store.query()
    assert len(results) == 0
    
    # Verify manual fetch still works for ID
    archived = await store.get(str(saved.id))
    assert archived.archived is True

@pytest.mark.asyncio
async def test_deduplication(store):
    record1 = MemoryRecord(content="Unique content")
    saved1 = await store.save(record1)
    
    # Save same content twice
    record2 = MemoryRecord(content="Unique content")
    saved2 = await store.save(record2)
    
    # Should be the same record (deduplicated by hash)
    assert saved1.id == saved2.id
    assert saved1.created_at == saved2.created_at

@pytest.mark.asyncio
async def test_optimistic_concurrency(store):
    record = MemoryRecord(content="Concurrency test")
    saved = await store.save(record)
    
    # Simulate concurrent update
    # Pass correct version
    await store.update(str(saved.id), {"content": "Update 1", "version": 1})
    
    # Pass WRONG version (expecting 1, but DB is now 2)
    with pytest.raises(ValueError, match="Version conflict"):
        await store.update(str(saved.id), {"content": "Update 2", "version": 1})

@pytest.mark.asyncio
async def test_reference_management(store):
    parent = MemoryRecord(content="Parent")
    child = MemoryRecord(content="Child")
    
    s_parent = await store.save(parent)
    s_child = await store.save(child)
    
    # Link
    await store.link_records(str(s_parent.id), str(s_child.id))
    
    # Verify related
    related = await store.get_related(str(s_parent.id))
    # Note: get_related in our implementation fetches records referenced BY the ID
    # Simpler link implementation just appended to ref_id of parent.
    # So retrieving parent should show child in ref_id?
    # get_related(id) -> fetches records that are in db_record.ref_id
    
    # Let's verify ref_id in parent
    p_fetched = await store.get(str(s_parent.id))
    assert str(s_child.id) in p_fetched.ref_id
    
    # Verify get_related returns the child record
    related_recs = await store.get_related(str(s_parent.id))
    assert len(related_recs) == 1
    assert related_recs[0].id == s_child.id

