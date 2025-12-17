from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from app.core.config import settings
from app.core.asr_engine import asr_engine
from app.agent.executor import agent_executor
import shutil
import os

class AgentRunRequest(BaseModel):
    goal: str
from app.core.llm_engine import llm_engine
from app.core.global_lock import gpu_lock
import time
import logging
import asyncio

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")

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

@app.on_event("startup")
def startup_event():
    llm_engine.load_model()

@app.post(f"{settings.API_V1_STR}/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    model = llm_engine.get_model()
    
    try:
        # Convert Pydantic models to dicts for llama-cpp-python
        messages = [msg.dict(exclude_none=True) for msg in request.messages]
        
        logger.info("Acquiring GPU lock for LLM inference...")
        async with gpu_lock:
            # We run the synchronous Llama inference in a thread to verify async lock works
            # and to not block the event loop (though Llama release GIL often)
            response = await asyncio.to_thread(
                model.create_chat_completion,
                messages=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=request.stream,
                stop=request.stop,
                max_tokens=request.max_tokens,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                model=request.model 
            )
        return response
    except Exception as e:
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
async def run_agent(request: AgentRunRequest):
    logger.info(f"Received agent goal: {request.goal}")
    try:
        # Agent execution involves heavy thinking and tool usage (IO)
        # We run it in a thread to verify async/sync interop
        result = await asyncio.to_thread(agent_executor.run, request.goal)
        return {"answer": result}
    except Exception as e:
        logger.error(f"Agent endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
