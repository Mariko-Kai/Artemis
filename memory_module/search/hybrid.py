"""
Hybrid search combining vector and lexical search.

Uses Reciprocal Rank Fusion (RRF) to combine results from multiple sources.
"""

import logging
from typing import Dict, List

from ..models import SearchResult

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]], k: int = 60
) -> List[SearchResult]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF formula: RRF(d) = Σ(1 / (k + rank(d)))
    where k is a constant (typically 60) and rank is the position in the list.

    Args:
        result_lists: List of ranked result lists
        k: RRF constant (higher values reduce rank impact)

    Returns:
        Fused and re-ranked list of search results
    """
    # Calculate RRF scores for each document
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, SearchResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            doc_id = str(result.id)

            # Add RRF score contribution
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                doc_map[doc_id] = result

            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Sort by RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Create new SearchResult list with RRF scores
    fused_results = []
    for doc_id, rrf_score in sorted_docs:
        result = doc_map[doc_id]
        # Update score to be the RRF score
        fused_results.append(
            SearchResult(
                id=result.id,
                content=result.content,
                score=rrf_score,
                metadata=result.metadata,
                memory_type=result.memory_type,
                tags=result.tags,
                created_at=result.created_at,
            )
        )

    return fused_results


def weighted_score_fusion(
    vector_results: List[SearchResult],
    lexical_results: List[SearchResult],
    vector_weight: float = 0.5,
) -> List[SearchResult]:
    """
    Combine results using weighted score fusion.

    Final score = (vector_score * vector_weight) + (lexical_score * (1 - vector_weight))

    Args:
        vector_results: Results from vector search
        lexical_results: Results from lexical search
        vector_weight: Weight for vector scores (0-1)

    Returns:
        Fused and re-ranked list of search results
    """
    lexical_weight = 1.0 - vector_weight

    # Normalize scores to 0-1 range for each result set
    def normalize_scores(results: List[SearchResult]) -> Dict[str, float]:
        if not results:
            return {}

        max_score = max(r.score for r in results)
        min_score = min(r.score for r in results)
        score_range = max_score - min_score

        if score_range == 0:
            return {str(r.id): 1.0 for r in results}

        return {
            str(r.id): (r.score - min_score) / score_range for r in results
        }

    vector_scores = normalize_scores(vector_results)
    lexical_scores = normalize_scores(lexical_results)

    # Combine scores
    combined_scores: Dict[str, float] = {}
    doc_map: Dict[str, SearchResult] = {}

    # Add vector scores
    for result in vector_results:
        doc_id = str(result.id)
        combined_scores[doc_id] = vector_scores[doc_id] * vector_weight
        doc_map[doc_id] = result

    # Add lexical scores
    for result in lexical_results:
        doc_id = str(result.id)
        if doc_id in combined_scores:
            combined_scores[doc_id] += lexical_scores[doc_id] * lexical_weight
        else:
            combined_scores[doc_id] = lexical_scores[doc_id] * lexical_weight
            doc_map[doc_id] = result

    # Sort by combined score
    sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    # Create fused results
    fused_results = []
    for doc_id, score in sorted_docs:
        result = doc_map[doc_id]
        fused_results.append(
            SearchResult(
                id=result.id,
                content=result.content,
                score=score,
                metadata=result.metadata,
                memory_type=result.memory_type,
                tags=result.tags,
                created_at=result.created_at,
            )
        )

    return fused_results


async def hybrid_search(
    query_vector: List[float],
    query_text: str,
    vector_store,
    lexical_index,
    top_k: int = 10,
    hybrid_weight: float = 0.5,
    use_rrf: bool = True,
) -> List[SearchResult]:
    """
    Perform hybrid search combining vector and lexical approaches.

    Args:
        query_vector: Query embedding vector
        query_text: Query text
        vector_store: Vector store instance
        lexical_index: Lexical index instance
        top_k: Number of final results
        hybrid_weight: Weight for vector search (only used if use_rrf=False)
        use_rrf: Whether to use RRF (True) or weighted fusion (False)

    Returns:
        Fused search results
    """
    # Retrieve more results from each source for better fusion
    retrieval_k = min(top_k * 3, 100)

    # Perform both searches in parallel
    vector_results = await vector_store.search(query_vector, top_k=retrieval_k)
    lexical_results = await lexical_index.search(query_text, top_k=retrieval_k)

    logger.debug(
        f"Hybrid search: {len(vector_results)} vector results, "
        f"{len(lexical_results)} lexical results"
    )

    # Fuse results
    if use_rrf:
        fused_results = reciprocal_rank_fusion([vector_results, lexical_results])
    else:
        fused_results = weighted_score_fusion(
            vector_results, lexical_results, vector_weight=hybrid_weight
        )

    # Return top-k
    return fused_results[:top_k]
