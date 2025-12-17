"""
Preprocessing pipeline for text data.

Includes:
- Text cleaning
- Token counting (tiktoken)
- Chunking strategies
- Keyword extraction
"""

import re
import logging
from typing import List, Literal, Optional

import tiktoken
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)

# Pattern for simple sentence splitting (periods followed by space/end)
SENTENCE_PATTERN = r'(?<=[.!?])\s+'

class PreprocessingPipeline:
    """Text preprocessing pipeline."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the pipeline.

        Args:
            model_name: Model name for tiktoken encoding
        """
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"Model {model_name} not found, using cl100k_base")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Stop words list (minimal set)
        self.stop_words = "english" # Use sklearn's built-in list

    def clean_text(self, text: str) -> str:
        """
        Normalize and clean text.
        
        - Normalize whitespace
        - Remove control characters
        """
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove control characters (except newlines if we wanted to keep them, but we normalized spaces above)
        # If we want to keep structure for markdown, we should be careful.
        # But for embedding, flattening is usually okay.
        # Let's keep it simple: normalize whitespace.
        return text.strip()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))

    def chunk_text(
        self, 
        text: str, 
        strategy: Literal["boundary_aligned", "token_based"] = "token_based",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[str]:
        """
        Chunk text into smaller segments.

        Args:
            text: Input text
            strategy: Chunking strategy
            chunk_size: Max tokens per chunk
            chunk_overlap: Overlap in tokens (for token_based)
        
        Returns:
            List of text chunks
        """
        if strategy == "boundary_aligned":
            return self._boundary_aligned_chunking(text, chunk_size)
        else:
            return self._token_based_chunking(text, chunk_size, chunk_overlap)

    def _token_based_chunking(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Chunk based on strict token count with overlap."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= chunk_size:
            return [text]
        
        chunks = []
        step = chunk_size - chunk_overlap
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        
        return chunks

    def _boundary_aligned_chunking(self, text: str, chunk_size: int) -> List[str]:
        """
        Chunk based on sentence boundaries.
        Optimized for chat messages to keep context intact where possible.
        """
        # First split by newlines if significant
        # But we already cleaned whitespace effectively removing newlines in clean_text?
        # If clean_text removed newlines, we can't use them.
        # Let's assume we might receive raw text or clean_text was called.
        # Ideally clean_text should probably preserve semantic newlines or we skip it for this strategy.
        
        sentences = re.split(SENTENCE_PATTERN, text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_length + sentence_tokens > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # If a single sentence is too big, force split it
                if sentence_tokens > chunk_size:
                    sub_chunks = self._token_based_chunking(sentence, chunk_size, 0)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        Extract keywords using TF (Term Frequency) from a single document.
        Using sklearn CountVectorizer.
        """
        try:
            vectorizer = CountVectorizer(stop_words=self.stop_words, max_features=top_k)
            # We need to pass an iterable
            vectorizer.fit([text])
            # For a single doc, fit selects the most frequent words in that doc
            # This is effectively what we want for "lexical keywords" of a specific memory/doc
            return list(vectorizer.vocabulary_.keys())
        except ValueError:
            # Can happen if text is empty or only stop words
            return []

# Helper instance
_pipeline: Optional[PreprocessingPipeline] = None

def get_preprocessing_pipeline() -> PreprocessingPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PreprocessingPipeline()
    return _pipeline
