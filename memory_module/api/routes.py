"""
FastAPI routes for memory operations.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from ..models import (
    BatchStoreRequest,
    MemoryRecord,
    QueryRequest,
    QueryResponse,
    StatsResponse,
    StoreRequest,
)
from ..service import MemoryService
from .dependencies import get_memory_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/memories", response_model=MemoryRecord, status_code=status.HTTP_201_CREATED)
async def store_memory(
    request: StoreRequest,
    service: MemoryService = Depends(get_memory_service),
) -> MemoryRecord:
    """
    Store a new memory.

    Args:
        request: Store request with memory data
        service: Memory service instance

    Returns:
        Stored memory record
    """
    try:
        memory = await service.store_memory_from_request(request)
        return memory
    except Exception as e:
        logger.error(f"Error storing memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store memory: {str(e)}",
        )


@router.post(
    "/memories/batch",
    response_model=List[MemoryRecord],
    status_code=status.HTTP_201_CREATED,
)
async def store_memories_batch(
    request: BatchStoreRequest,
    service: MemoryService = Depends(get_memory_service),
) -> List[MemoryRecord]:
    """
    Store multiple memories in batch.

    Args:
        request: Batch store request
        service: Memory service instance

    Returns:
        List of stored memory records
    """
    try:
        memories = []
        for store_req in request.memories:
            memory = await service.store_memory_from_request(store_req)
            memories.append(memory)
        return memories
    except Exception as e:
        logger.error(f"Error storing batch memories: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store batch memories: {str(e)}",
        )


@router.get("/memories/{memory_id}", response_model=MemoryRecord)
async def get_memory(
    memory_id: str,
    service: MemoryService = Depends(get_memory_service),
) -> MemoryRecord:
    """
    Retrieve a memory by ID.

    Args:
        memory_id: Memory UUID
        service: Memory service instance

    Returns:
        Memory record

    Raises:
        HTTPException: If memory not found
    """
    memory = await service.get_memory(memory_id)
    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found",
        )
    return memory


@router.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: str,
    service: MemoryService = Depends(get_memory_service),
) -> None:
    """
    Delete a memory.

    Args:
        memory_id: Memory UUID
        service: Memory service instance

    Raises:
        HTTPException: If memory not found
    """
    deleted = await service.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found",
        )


@router.post("/search", response_model=QueryResponse)
async def search_memories(
    request: QueryRequest,
    service: MemoryService = Depends(get_memory_service),
) -> QueryResponse:
    """
    Search for memories using hybrid search.

    Args:
        request: Query request
        service: Memory service instance

    Returns:
        Query response with results
    """
    try:
        return await service.search_memories(request)
    except Exception as e:
        logger.error(f"Error searching memories: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    service: MemoryService = Depends(get_memory_service),
) -> StatsResponse:
    """
    Get service statistics.

    Args:
        service: Memory service instance

    Returns:
        Statistics response
    """
    try:
        stats = await service.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        )


@router.post("/admin/save-indexes", status_code=status.HTTP_200_OK)
async def save_indexes(
    service: MemoryService = Depends(get_memory_service),
) -> dict:
    """
    Save indexes to disk (admin endpoint).

    Args:
        service: Memory service instance

    Returns:
        Success message
    """
    try:
        await service.save_indexes()
        return {"message": "Indexes saved successfully"}
    except Exception as e:
        logger.error(f"Error saving indexes: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save indexes: {str(e)}",
        )
