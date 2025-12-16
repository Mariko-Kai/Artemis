from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from app.core.config import settings
from app.core.llm_engine import llm_engine
import time

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
        
        response = model.create_chat_completion(
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
            stop=request.stop,
            max_tokens=request.max_tokens,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            model=request.model # Echo back the model name or use internal
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
