"""
Unit tests for MRL search functionality.
"""

import pytest
import numpy as np
import uuid
from memory_module.search.vector_store import FAISSVectorStore

@pytest.mark.asyncio
async def test_mrl_addition_and_search(test_settings, temp_index_path):
    # Generate unique IDs for test isolation
    id1 = str(uuid.uuid4())
    id2 = str(uuid.uuid4())
    
    # Initialize store with MRL (dim=128, storage=384 from conftest)
    store = FAISSVectorStore(
        dimension=test_settings.mrl_index_dim,
        index_path=temp_index_path,
        database_url=test_settings.database_url,
        index_type="Flat", # Use Flat for exact checks in test
        mrl_enabled=True,
        storage_dimension=test_settings.mrl_storage_dim
    )
    
    # Check if mappings table exists (created by MetadataStore fixture usually)
    # But here we are using store directly. We need to init DB tables.
    from memory_module.storage.models import Base
    async with store.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # Create dummy vectors
    vec_dim = test_settings.mrl_storage_dim
    vec1 = [0.1] * vec_dim
    vec2 = [0.2] * vec_dim # Higher magnitude/different direction if normalized?
    # If normalized, [0.1]*d and [0.2]*d are identical (angle 0).
    # Let's make them orthogonal-ish in first few dims
    vec1 = [1.0 if i < 10 else 0.0 for i in range(vec_dim)]
    vec2 = [1.0 if i >= 10 and i < 20 else 0.0 for i in range(vec_dim)]
    
    # Add vectors
    ids = [id1, id2]
    vectors = [vec1, vec2]
    metadatas = [{"content": "c1"}, {"content": "c2"}]
    
    # Seed MemoryRecordDB (search requires these to exist)
    from memory_module.storage.models import MemoryRecordDB
    async with store.async_session() as session:
        session.add(MemoryRecordDB(
            id=id1, content="c1", memory_type="semantic", 
            record_metadata={"content": "c1"}
        ))
        session.add(MemoryRecordDB(
            id=id2, content="c2", memory_type="semantic",
            record_metadata={"content": "c2"}
        ))
        await session.commit()

    await store.batch_add(ids, vectors, metadatas)
    
    assert store.index.ntotal == 2
    
    # Verify DB content
    async with store.async_session() as session:
        from memory_module.storage.models import MemoryRecordDB, VectorMapping
        from sqlalchemy import select
        
        # Check MemoryRecords
        res = await session.execute(select(MemoryRecordDB))
        recs = res.scalars().all()
        print(f"MemoryRecords: {[r.id for r in recs]}")
        assert len(recs) == 2
        
        # Check VectorMappings
        res = await session.execute(select(VectorMapping))
        maps = res.scalars().all()
        print(f"VectorMappings: {[(m.faiss_id, m.memory_id) for m in maps]}")
        assert len(maps) == 2

    # Search for vec1 (should retrieve id1)
    results = await store.search(vec1, top_k=1)
    print(f"Search results for vec1: {results}")
    
    assert len(results) == 1
    assert results[0].id == id1
    
    # Search for vec2
    results = await store.search(vec2, top_k=1)
    print(f"Search results for vec2: {results}")
    assert len(results) == 1
    assert results[0].id == id2
