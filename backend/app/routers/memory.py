
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from pydantic import BaseModel

from app.db.database import get_db
from app.services.memory_service import memory_service
from app.db.memory_models import MemoryRecordDB

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: int = 5

class StoreRequest(BaseModel):
    content: str
    session_id: str
    role: str = "user"
    importance: float = 0.5

class MemoryResponse(BaseModel):
    id: str
    content: str
    score: float
    role: Optional[str]
    created_at: str

@router.post("/query", response_model=List[MemoryResponse])
async def query_memory(request: QueryRequest):
    """
    Search memories manually.
    """
    try:
        results = await memory_service.query_memory(
            query=request.query,
            session_id=request.session_id,
            top_k=request.top_k
        )
        return [
            MemoryResponse(
                id=r["id"],
                content=r["content"],
                score=float(r["score"]),
                role=r["role"],
                created_at=str(r["created_at"])
            ) for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/store")
async def store_memory(request: StoreRequest):
    """
    Manually store a memory (Hot storage).
    """
    try:
        await memory_service.store_memory(
            session_id=request.session_id,
            content=request.content,
            role=request.role,
            importance=request.importance
        )
        return {"status": "stored"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/{chat_id}/context")
async def get_chat_context(chat_id: str, db: Session = Depends(get_db)):
    """
    Get all memories associated with a chat session (basic retrieval).
    """
    memories = db.query(MemoryRecordDB).filter(MemoryRecordDB.channel_id == chat_id).all()
    return [
        {
            "id": m.id,
            "content": m.content,
            "role": m.source,
            "created_at": m.created_at
        } for m in memories
    ]

@router.post("/chat/{chat_id}/export")
async def export_chat_memories(chat_id: str, db: Session = Depends(get_db)):
    """
    Export memories for a chat.
    """
    memories = db.query(MemoryRecordDB).filter(MemoryRecordDB.channel_id == chat_id).all()
    return {
        "chat_id": chat_id,
        "count": len(memories),
        "records": [
            {
                "id": m.id,
                "content": m.content,
                "metadata": m.metadata_json
            } for m in memories
        ]
    }

@router.post("/chat/{chat_id}/summarize")
async def manual_summarize(chat_id: str):
    """
    Trigger manual summarization for a chat.
    """
    from app.services.summarization_worker import summarization_worker
    
    # Direct trigger (bypass queue check, but respect queue)
    # Ideally queue it to avoid race.
    from app.db.database import SessionLocal
    from app.db.memory_models import JobQueue
    
    db = SessionLocal()
    try:
        job = JobQueue(target_id=chat_id, job_type="summarization", status="pending")
        db.add(job)
        db.commit()
    finally:
        db.close()
    
    # Wake up worker
    if summarization_worker._task:
        # We can't easily interrupt sleep without cancelling.
        # Just let it pick up next cycle (60s). Or expose a 'wake' method.
        pass
        
    # Wake up worker
    if summarization_worker._task:
        # We can't easily interrupt sleep without cancelling.
        # Just let it pick up next cycle (60s). Or expose a 'wake' method.
        pass
        
    return {"status": "queued"}

# Archival Endpoints

@router.post("/archive/cleanup")
async def trigger_archival_cleanup():
    """
    Trigger manual archival cleanup.
    """
    from app.services.archival_service import archival_service
    await archival_service.run_cleanup()
    return {"status": "cleanup_started"}

@router.post("/archive/restore/{archive_id}")
async def restore_archived_memory(archive_id: str):
    """
    Restore a memory from archive.
    """
    from app.services.archival_service import archival_service
    success = await archival_service.restore_record(archive_id)
    if not success:
        raise HTTPException(status_code=404, detail="Archived record not found")
    return {"status": "restored"}

@router.get("/archive/export")
async def export_all_memories():
    """
    Export all active memories as JSONL.
    """
    from app.services.archival_service import archival_service
    # Streaming response might be better for large exports, but string dump ok for <10k.
    data = await archival_service.export_all()
    from fastapi.responses import Response
    return Response(content=data, media_type="application/jsonl", headers={"Content-Disposition": "attachment; filename=memories.jsonl"})

@router.post("/export")
async def export_all_memories_alias():
     """Alias for /archive/export per user request."""
     return await export_all_memories()

@router.delete("/{memory_id}")
async def delete_memory(memory_id: str, db: Session = Depends(get_db)):
    """
    Delete a memory record (GDPR compliance).
    Removes from DB and Indices.
    """
    try:
        # 1. Delete from DB
        record = db.query(MemoryRecordDB).filter(MemoryRecordDB.id == memory_id).first()
        if not record:
             raise HTTPException(status_code=404, detail="Memory not found")
        
        db.delete(record)
        db.commit()
        
        # 2. Remove from Indices
        # We need to access memory_service manually or add delete method to it
        # memory_service was imported
        if memory_service.vector_store:
            await memory_service.vector_store.delete(memory_id)
        if memory_service.lexical_index:
            await memory_service.lexical_index.delete(memory_id)
            
        return {"status": "deleted", "id": memory_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/archive/stats")
async def get_archive_stats(db: Session = Depends(get_db)):
    """
    Get statistics about archived memories.
    """
    from app.db.memory_models import ArchivedMemoryRecordDB
    from sqlalchemy import func
    
    count = db.query(func.count(ArchivedMemoryRecordDB.id)).scalar()
    total_size = db.query(func.sum(ArchivedMemoryRecordDB.decompressed_size)).scalar() or 0
    compressed_size = 0 # Cannot easily sum BLOB size in SQLite query without function, assume ratio
    
    # Estimate compressed size or just return count
    return {
        "archived_count": count,
        "total_decompressed_size_bytes": total_size
    }

@router.post("/search", response_model=List[MemoryResponse])
async def search_memory_alias(request: QueryRequest):
    """Alias for /query per user request."""
    return await query_memory(request)
