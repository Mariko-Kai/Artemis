from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.asr_engine import asr_engine
from app.agent.executor import agent_executor
from app.core.llm_engine import llm_engine
from app.core.global_lock import gpu_lock
from app.db.database import engine, Base, get_db, SessionLocal
from app.db.models import ChatSession, ChatMessage as DbMessage
from app.db.models import ChatSession, ChatMessage as DbMessage
from app.db.memory_models import MemoryRecordDB, VectorMapping, AuditLog, ConfigVersion, JobQueue, ArchivedMemoryRecordDB
from app.routers import sessions

import shutil
import os
import time
import logging
import asyncio
import uuid

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")

# Include Routers
# Include Routers
app.include_router(sessions.router, prefix=settings.API_V1_STR, tags=["sessions"])
from app.routers import memory
app.include_router(memory.router, prefix=f"{settings.API_V1_STR}/memory", tags=["memory"])

class AgentRunRequest(BaseModel):
    goal: str
    session_id: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "local-model"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    session_id: Optional[str] = None

def get_context_window(db: Session, session_id: str, new_messages: List[Dict], max_tokens: int = 4096):
    # Retrieve last N messages
    # Simple heuristic: 1 token ~= 4 chars
    # We reserve 1000 tokens for generation and new messages
    # So we want roughly 3000 tokens of history
    
    # Get all messages sorted by timestamp DESC (newest first)
    # Then take as many as fit
    
    history_msgs = db.query(DbMessage).filter(DbMessage.session_id == session_id).order_by(DbMessage.timestamp.desc()).limit(20).all()
    history_msgs.reverse() # Oldest first
    
    context_msgs = []
    
    # Convert DB messages to format
    # {"role": "...", "content": "..."}
    for m in history_msgs:
        context_msgs.append({"role": m.role, "content": m.content})
        
    return context_msgs

def save_message(db: Session, session_id: str, role: str, content: str):
    msg = DbMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()

async def generate_title(session_id: str, first_msg: str):
    logger.info(f"Generating title for session {session_id}...")
    try:
        # Prompt
        prompt = f"Summarize the following message into a short 3-5 word title:\n\nMessage: {first_msg}\n\nTitle:"
        
        async with gpu_lock:
             response = await asyncio.to_thread(
                llm_engine.get_model().create_completion,
                prompt=prompt,
                max_tokens=20,
                stop=["\n"]
            )
        title = response["choices"][0]["text"].strip().strip('"')
        
        # Update DB
        db = SessionLocal()
        try:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session:
                session.title = title
                db.commit()
                logger.info(f"Set title for {session_id} to: {title}")
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Title generation failed: {e}")

@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)
    llm_engine.load_model()
    # Initialize Memory Service
    from app.services.memory_service import memory_service
    await memory_service.initialize()
    # Start Summarization Worker
    await summarization_worker.start()
    # Start Archival Service
    from app.services.archival_service import archival_service
    await archival_service.start_cleanup_task()

@app.on_event("shutdown")
async def shutdown_event():
    await summarization_worker.stop()
    from app.services.archival_service import archival_service
    await archival_service.stop_cleanup_task()

@app.post(f"{settings.API_V1_STR}/chat/completions")
async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    model = llm_engine.get_model()
    from app.services.memory_service import memory_service
    
    try:
        # Update Activity Timestamp
        if request.session_id:
            try:
                session = db.query(ChatSession).filter(ChatSession.id == request.session_id).first()
                if session:
                    session.last_activity = datetime.utcnow()
                    db.commit()
            except Exception as e:
                logger.error(f"Failed to update activity: {e}")

        current_messages = [msg.dict(exclude_none=True) for msg in request.messages]
        messages_for_inference = []
        
        # Memory Context
        memory_context = ""
        user_msg_content = ""
        for msg in current_messages:
            if msg.get("role") == "user":
                user_msg_content = msg.get("content")

        if user_msg_content:
            # Query memories (async)
            memories = await memory_service.query_memory(user_msg_content, session_id=request.session_id)
            if memories:
                memory_block = "\n".join([f"- {m['content']} (from {m['created_at'].strftime('%Y-%m-%d')})" for m in memories])
                memory_context = f"\nRelevant previous context:\n{memory_block}\n"
                logger.info(f"Injected {len(memories)} memories")

        if request.session_id:
            # 1. Load History
            history = get_context_window(db, request.session_id, current_messages)
            
            # Prepend memory to the very first system message or create one?
            # Or just prepend to the whole list as a 'system' message?
            # Ideally, detailed system prompt -> memories -> history -> new message
            
            # Simple strategy: Add a system message with memories at the start
            final_messages = []
            if memory_context:
                final_messages.append({"role": "system", "content": memory_context})
                
            final_messages.extend(history)
            final_messages.extend(current_messages)
            
            messages_for_inference = final_messages
            
            # 2. Save User Message(s)
            for msg in current_messages:
                if msg.get("role") == "user":
                    save_message(db, request.session_id, "user", msg.get("content"))
                    # Auto-save to memory
                    background_tasks.add_task(
                        memory_service.store_memory,
                        session_id=request.session_id,
                        content=msg.get("content"),
                        role="user",
                        importance=0.7
                    )
                    
            # 3. Check for Auto-Titling
            if not history and user_msg_content:
                background_tasks.add_task(generate_title, request.session_id, user_msg_content)

        else:
            messages_for_inference = current_messages

        logger.info("Acquiring GPU lock for LLM inference...")
        async with gpu_lock:
            response = await asyncio.to_thread(
                model.create_chat_completion,
                messages=messages_for_inference,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=request.stream,
                stop=request.stop,
                max_tokens=request.max_tokens,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                model=request.model 
            )
            
        # 4. Save Assistant Response
        if request.session_id and not request.stream:
            content = response["choices"][0]["message"]["content"]
            save_message(db, request.session_id, "assistant", content)
            
            # Auto-save to memory
            background_tasks.add_task(
                memory_service.store_memory,
                session_id=request.session_id,
                content=content,
                role="assistant",
                importance=0.5 # Default, tools might boost this
            )
            
            # Inject related memories into response for Frontend UI
            if memories:
                # Add a custom field to the first choice's message or top level
                # OpenAI spec is strict, but extra fields usually ignored by strict clients, useful for us.
                # Let's add to top level 'context'
                response["related_memories"] = [
                    {"id": m["id"], "content": m["content"], "score": m["score"]} 
                    for m in memories
                ]
            
        return response
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_STR}/audio/transcriptions")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        transcript = await asr_engine.transcribe(temp_file_path)
        return {"text": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post(f"{settings.API_V1_STR}/agent/run")
async def run_agent(request: AgentRunRequest, db: Session = Depends(get_db)):
    logger.info(f"Received agent goal: {request.goal}")
    try:
        if request.session_id:
            save_message(db, request.session_id, "user", request.goal)
            
        # Agent execution involves heavy thinking and tool usage (IO)
        # We run it in a thread to verify async/sync interop
        result = await asyncio.to_thread(agent_executor.run, request.goal)
        
        if request.session_id:
            save_message(db, request.session_id, "assistant", result)
            
        return {"answer": result}
    except Exception as e:
        logger.error(f"Agent endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
