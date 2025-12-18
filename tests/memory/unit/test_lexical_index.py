
import pytest
import os
from memory_module.search.lexical import BM25LexicalIndex

@pytest.fixture
def lexical_index(temp_indices_dir):
    """Create a lexical index instance"""
    path = os.path.join(temp_indices_dir, "test_bm25.pkl")
    # For unit testing, we can use a simpler tokenizer if needed,
    # but let's see if we can make the real one work.
    index = BM25LexicalIndex(index_path=path)
    return index

@pytest.mark.asyncio
async def test_add_documents(lexical_index):
    """Test adding text documents"""
    docs = ["hello world", "hello python", "testing code"]
    ids = ["550e8400-e29b-41d4-a716-446655440000", "550e8400-e29b-41d4-a716-446655440001", "550e8400-e29b-41d4-a716-446655440002"] 
    metadatas = [{"source": "test"}] * 3
    
    await lexical_index.batch_index(ids, docs, metadatas)
    assert len(lexical_index.corpus) == 3

@pytest.mark.asyncio
async def test_search(lexical_index):
    """Test keyword search"""
    docs = [
        "apple banana", 
        "apple orange", 
        "banana grape", 
        "cherry date", 
        "elderberry fig", 
        "guava honeydew"
    ]
    ids = [
        "550e8400-e29b-41d4-a716-446655440010", 
        "550e8400-e29b-41d4-a716-446655440020", 
        "550e8400-e29b-41d4-a716-446655440030",
        "550e8400-e29b-41d4-a716-446655440040",
        "550e8400-e29b-41d4-a716-446655440050",
        "550e8400-e29b-41d4-a716-446655440060"
    ]
    metadatas = [{"source": "test"}] * 6
    await lexical_index.batch_index(ids, docs, metadatas)
    
    # Search for 'apple'
    results = await lexical_index.search("apple", top_k=5)
    
    # Debug: if results are empty, let's see what's in corpus
    if not results:
        print(f"Corpus: {lexical_index.corpus}")
        print(f"BM25 initialized: {lexical_index.bm25 is not None}")
        
    assert len(results) >= 2
    found_ids = [str(r.id) for r in results]
    assert "550e8400-e29b-41d4-a716-446655440010" in found_ids
    assert "550e8400-e29b-41d4-a716-446655440020" in found_ids
    assert "550e8400-e29b-41d4-a716-446655440030" not in found_ids 

@pytest.mark.asyncio
async def test_persistence(lexical_index, temp_indices_dir):
    """Test save and load"""
    path = os.path.join(temp_indices_dir, "test_bm25_save.pkl")
    docs = ["save me"]
    ids = ["550e8400-e29b-41d4-a716-446655440099"]
    metadatas = [{"source": "test"}]
    await lexical_index.batch_index(ids, docs, metadatas)
    lexical_index.save(path)
    
    new_index = BM25LexicalIndex(index_path=path)
    new_index.load(path)
    
    assert len(new_index.corpus) == 1
    results = await new_index.search("save")
    assert str(results[0].id) == "550e8400-e29b-41d4-a716-446655440099"
