
import random
import uuid
import datetime
import numpy as np
from typing import List, Dict

def generate_embedding(dim: int = 768) -> List[float]:
    """Generate a random normalized embedding vector"""
    vec = np.random.rand(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist()

def generate_chat_messages(count: int = 100) -> List[Dict]:
    """Generate sample chat messages"""
    roles = ["user", "assistant"]
    topics = [
        "python programming", "artificial intelligence", "space exploration", 
        "cooking recipes", "history of rome", "quantum physics",
        "kubernetes deployment", "react hooks", "sql optimization"
    ]
    
    messages = []
    base_time = datetime.datetime.now() - datetime.timedelta(days=30)
    
    for i in range(count):
        role = random.choice(roles)
        topic = random.choice(topics)
        content = f"Tell me about {topic} specific detail {i}" if role == "user" else f"Here is information about {topic} for query {i}"
        
        msg = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "created_at": base_time + datetime.timedelta(hours=i),
            "usage": {"total_tokens": random.randint(50, 500)}
        }
        messages.append(msg)
        
    return messages

def generate_memory_records(count: int = 100, dim: int = 768) -> List[Dict]:
    """Generate complete memory records with embeddings"""
    msgs = generate_chat_messages(count)
    records = []
    
    for msg in msgs:
        record = {
            "id": str(uuid.uuid4()),
            "content": msg["content"],
            "embedding": generate_embedding(dim),
            "metadata": {
                "role": msg["role"],
                "source": "chat",
                "timestamp": msg["created_at"].isoformat()
            },
            "created_at": msg["created_at"],
            "importance": random.uniform(0.1, 1.0)
        }
        records.append(record)
        
    return records
