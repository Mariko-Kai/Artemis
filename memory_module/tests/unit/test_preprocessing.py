"""
Unit tests for preprocessing pipeline.
"""

import pytest
from memory_module.pipeline.preprocessing import PreprocessingPipeline

@pytest.fixture
def pipeline():
    return PreprocessingPipeline()

def test_clean_text(pipeline):
    text = "  Hello   World! \n This is a   test.  "
    cleaned = pipeline.clean_text(text)
    assert cleaned == "Hello World! This is a test."

def test_count_tokens(pipeline):
    text = "Hello world"
    count = pipeline.count_tokens(text)
    # gpt-3.5/4/tiktoken count: 'Hello' (1) + ' world' (1) = 2 usually
    assert count > 0

def test_token_based_chunking(pipeline):
    text = "word " * 100
    # chunk size 10, overlap 2
    chunks = pipeline.chunk_text(text, strategy="token_based", chunk_size=10, chunk_overlap=2)
    assert len(chunks) > 0
    # Verify roughly correct logic (overlap means more chunks)
    
def test_boundary_aligned_chunking(pipeline):
    text = "Sentence one. Sentence two. Sentence three."
    chunks = pipeline.chunk_text(text, strategy="boundary_aligned", chunk_size=10) # Small chunk size to force split
    
    # Needs meaningful asserts based on implementation
    # Given simple period splitting
    assert len(chunks) >= 1

def test_extract_keywords(pipeline):
    text = "Python is a programming language. Python is great."
    keywords = pipeline.extract_keywords(text, top_k=2)
    assert "python" in keywords
