"""
BM25-based lexical search implementation.

Provides traditional keyword-based search using BM25 algorithm.
"""

import logging
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from ..interfaces import ILexicalIndex
from ..models import SearchResult

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """
    Simple tokenization function.

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    # Simple whitespace tokenization with lowercasing
    # In production, consider using nltk or spacy
    return text.lower().split()


class BM25LexicalIndex(ILexicalIndex):
    """Lexical search using BM25 algorithm."""

    def __init__(self) -> None:
        """Initialize BM25 lexical index."""
        self.corpus: List[List[str]] = []
        self.documents: Dict[str, Dict[str, Any]] = {}  # id -> {text, metadata}
        self.id_list: List[str] = []  # Ordered list of IDs matching corpus
        self.bm25: BM25Okapi | None = None

        logger.info("BM25 lexical index initialized")

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from current corpus."""
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            logger.debug(f"Rebuilt BM25 index with {len(self.corpus)} documents")
        else:
            self.bm25 = None

    async def index_document(self, id: str, text: str, metadata: Dict[str, Any]) -> None:
        """
        Index a single document.

        Args:
            id: Document ID
            text: Document text
            metadata: Associated metadata
        """
        if id in self.documents:
            logger.warning(f"Document {id} already indexed, skipping")
            return

        # Tokenize and add to corpus
        tokens = tokenize(text)
        self.corpus.append(tokens)
        self.id_list.append(id)
        self.documents[id] = {"text": text, "metadata": metadata}

        # Rebuild index
        self._rebuild_index()
        logger.debug(f"Indexed document {id}")

    async def batch_index(
        self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Index multiple documents.

        Args:
            ids: List of document IDs
            texts: List of document texts
            metadatas: List of metadata dictionaries
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

    async def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Search documents using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results sorted by BM25 score
        """
        if not self.bm25 or not self.corpus:
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

        Args:
            id: Document ID

        Returns:
            True if deleted, False if not found
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
        return True
