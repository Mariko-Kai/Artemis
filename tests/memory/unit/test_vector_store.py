
import pytest
import pytest_asyncio
import os
import numpy as np
from memory_module.search.vector_store import FAISSVectorStore
from memory_module.storage.models import MemoryRecordDB
from memory_module.models import MemoryType
from uuid import uuid4
from datetime import datetime

@pytest_asyncio.fixture
async def store_instance(temp_indices_dir):
    """Create a vector store instance"""
    index_path = os.path.join(temp_indices_dir, "test_index")
    # For testing mappings, we use a temporary SQLite DB
    db_path = os.path.join(temp_indices_dir, "test_mappings.db")
    database_url = f"sqlite+aiosqlite:///{db_path.replace(os.sep, '/')}"
    
    store = FAISSVectorStore(
        dimension=768,
        index_path=index_path,
        database_url=database_url,
        storage_dimension=768,
        mrl_enabled=False # Simplify for unit tests
    )
    # Initialize index and DB
    await store.init_db()
    return store

@pytest.mark.asyncio
async def test_add_vectors(store_instance):
    """Test adding vectors"""
    # 2 vectors of dim 768
    vectors = [[0.1] * 768] * 2
    ids = ["id1", "id2"]
    metadatas = [{"source": "test"}] * 2
    
    # Implementation uses batch_add
    await store_instance.batch_add(ids, vectors, metadatas)
    
    assert len(ids) == 2
    assert store_instance.index.ntotal == 2

@pytest.mark.asyncio
async def test_search_vectors(store_instance):
    """Test searching vectors"""
    # Add a specific vector
    vec = [0.0] * 768
    vec[0] = 1.0 # Strong signal in first dim
    
    target_id = str(uuid4())
    # Manually insert into MemoryRecordDB so hydration works
    async with store_instance.async_session() as session:
        record = MemoryRecordDB(
            id=target_id,
            content="target content",
            memory_type=MemoryType.SEMANTIC,
            record_metadata={"source": "target"},
            created_at=datetime.now()
        )
        session.add(record)
        await session.commit()

    await store_instance.batch_add([target_id], [vec], [{"source": "target"}])
    
    # Search for it
    query = [0.0] * 768
    query[0] = 1.0
    
    results = await store_instance.search(query, top_k=1)
    
    assert str(results[0].id) == target_id
    assert results[0].score >= 0

@pytest.mark.asyncio
async def test_persistence(store_instance, temp_indices_dir):
    """Test saving and loading index"""
    vectors = [[0.1] * 768] * 5
    ids = [f"id{i}" for i in range(5)]
    metadatas = [{"source": "test"}] * 5
    
    await store_instance.batch_add(ids, vectors, metadatas)
    await store_instance.save(str(store_instance.index_path))
    
    # Load into new instance
    new_store = FAISSVectorStore(
        dimension=768,
        index_path=store_instance.index_path,
        database_url=store_instance.database_url,
        storage_dimension=768,
        mrl_enabled=False
    )
    await new_store.load(str(store_instance.index_path))
    
    assert new_store.index.ntotal == 5
