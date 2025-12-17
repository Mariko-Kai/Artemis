"""
BM25-based lexical search implementation.

Provides traditional keyword-based search using BM25 algorithm.
"""

import logging
import pickle
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..interfaces import ILexicalIndex
from ..models import SearchResult
from ..pipeline.preprocessing import get_preprocessing_pipeline

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """
    Tokenize text using the standardized pipeline.
    
    Args:
        text: Input text

    Returns:
        List of tokens
    """
    pipeline = get_preprocessing_pipeline()
    # Normalize first
    text = pipeline.clean_text(text)
    # Tokenize. The pipeline focuses on counting or chunking.
    # We need keywords/tokens for BM25.
    # Let's use the tokenizer from pipeline (tiktoken) or better yet,
    # the same logic used for keywords? 
    # Tiktoken is BPE, might be too granular for BM25 which prefers words.
    # Let's use a simple word tokenizer but with better cleaning.
    # Actually, pipeline.extract_keywords uses CountVectorizer which does tokenization.
    # Let's use a simple regex tokenizer consistent with standard NLP.
    # Simple whitespace + lowercase is what we had.
    # Let's stick to simple for now but ensure we use clean_text.
    return text.lower().split()


class BM25LexicalIndex(ILexicalIndex):
    """Lexical search using BM25 algorithm."""

    def __init__(self, index_path: Optional[str] = None) -> None:
        """
        Initialize BM25 lexical index.
        
        Args:
            index_path: Path to load/save the index
        """
        self.corpus: List[List[str]] = []
        self.documents: Dict[str, Dict[str, Any]] = {}  # id -> {text, metadata}
        self.id_list: List[str] = []  # Ordered list of IDs matching corpus
        self.bm25: BM25Okapi | None = None
        self.index_path = index_path
        self._dirty = False # Track if index needs rebuilding

        if self.index_path and os.path.exists(self.index_path):
            self.load(self.index_path)
            
        logger.info("BM25 lexical index initialized")

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from current corpus."""
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            self._dirty = False
            logger.debug(f"Rebuilt BM25 index with {len(self.corpus)} documents")
        else:
            self.bm25 = None

    async def index_document(self, id: str, text: str, metadata: Dict[str, Any]) -> None:
        """
        Index a single document.
        """
        if id in self.documents:
            logger.warning(f"Document {id} already indexed, skipping")
            return

        # Tokenize and add to corpus
        tokens = tokenize(text)
        self.corpus.append(tokens)
        self.id_list.append(id)
        self.documents[id] = {"text": text, "metadata": metadata}

        # Mark as dirty. Rebuild immediately for consistency in single-add filtering
        # or defer?
        # For real-time chat, immediate availability is preferred.
        self._rebuild_index()
        logger.debug(f"Indexed document {id}")

    async def batch_index(
        self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Index multiple documents.
        """
        if not ids or not texts or not metadatas:
            return

        if len(ids) != len(texts) != len(metadatas):
            raise ValueError("Length mismatch between ids, texts, and metadatas")

        # Filter out existing documents
        new_entries = [
            (id, text, meta)
            for id, text, meta in zip(ids, texts, metadatas)
            if id not in self.documents
        ]

        if not new_entries:
            logger.warning("All provided IDs already indexed, skipping batch index")
            return

        # Add new documents
        for id, text, metadata in new_entries:
            tokens = tokenize(text)
            self.corpus.append(tokens)
            self.id_list.append(id)
            self.documents[id] = {"text": text, "metadata": metadata}

        # Rebuild index once
        self._rebuild_index()
        logger.info(f"Batch indexed {len(new_entries)} documents")
        
        # Auto-save if path configured?
        if self.index_path:
            self.save(self.index_path)

    async def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search documents using BM25.
        """
        if not self.bm25:
             if self._dirty:
                 self._rebuild_index()
             if not self.bm25:
                 return []

        # Tokenize query
        query_tokens = tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        # Convert to SearchResult objects
        results = []
        
        # Determine max score for normalization if needed later, but here we return raw BM25
        # The Orchestrator will handle normalization.
        
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue

            doc_id = self.id_list[idx]
            doc = self.documents[doc_id]

            results.append(
                SearchResult(
                    id=doc_id,
                    content=doc["text"],
                    score=float(score),
                    lexical_score=float(score), # Populate lexical score
                    metadata=doc["metadata"].get("metadata", {}),
                    memory_type=doc["metadata"].get("memory_type", "semantic"),
                    tags=doc["metadata"].get("tags", []),
                    created_at=doc["metadata"].get("created_at"),
                )
            )

        return results

    async def delete(self, id: str) -> bool:
        """
        Remove a document from the index.
        """
        if id not in self.documents:
            return False

        # Find index in id_list
        idx = self.id_list.index(id)

        # Remove from all structures
        del self.corpus[idx]
        del self.id_list[idx]
        del self.documents[id]

        # Rebuild index
        self._rebuild_index()
        logger.debug(f"Deleted document {id}")
        
        if self.index_path:
            self.save(self.index_path)
            
        return True

    def save(self, path: str) -> None:
        """Save index to disk."""
        data = {
            "corpus": self.corpus,
            "documents": self.documents,
            "id_list": self.id_list
        }
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Saved lexical index to {path}")
        except Exception as e:
            logger.error(f"Failed to save lexical index: {e}")

    def load(self, path: str) -> None:
        """Load index from disk."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.corpus = data["corpus"]
            self.documents = data["documents"]
            self.id_list = data["id_list"]
            self._rebuild_index()
            logger.info(f"Loaded lexical index from {path}")
        except Exception as e:
            logger.error(f"Failed to load lexical index: {e}")
