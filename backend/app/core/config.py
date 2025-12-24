from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Artemis Backend"
    API_V1_STR: str = "/v1"
    
    # Embedding Configuration
    embedding_model_name: str = "Snowflake/snowflake-arctic-embed-m"
    embedding_device: str = "cpu"
    vector_dim: int = 768
    redis_url: str = ""
    cache_ttl: int = 3600
    DATABASE_URL: str = f"sqlite+aiosqlite:///{os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'sessions.db').replace(os.sep, '/')}"
    
    # Model Configuration
    # Model Configuration
    # Path relative to backend directory
    MODEL_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/Phi-3-mini-4k-instruct-q4.gguf"))
    N_GPU_LAYERS: int = 32  # Default to 0 (CPU) to avoid CUDA errors. Increase via env var if GPU is available.
    N_CTX: int = 4096000 # Using a large context window cap, but Phi-3 mini is 4k. Let's stick to 4096.
    
    # We should probably set N_CTX to 4096 to match model card, or 0 to use model's default.
    # The prompt says "Phi-3-mini-4k-instruct". 
    CONTEXT_WINDOW: int = 4096

    # System Prompt
    # System Prompt
    SYSTEM_PROMPT: str = """You are Artemis, a helpful and concise AI assistant.
1. Answer directly and relevantly to the user's request.
2. Do not lecture the user or provide unsolicited advice on etiquette.
3. Be concise. Avoid unnecessary introductions or conclusions. 
4. If the user speaks Russian, reply in Russian naturally.
5. If provided with multimodal input (images/audio), use them to answer but do not explicitly meta-reference the input type unless necessary for clarity.
"""

    # Summarization Configuration
    MEMORY_SUMMARIZATION_ENABLED: bool = True
    IDLE_TIME_MINUTES: int = 5
    SUMMARY_BATCH_SIZE: int = 20
    MEMORY_IDLE_THRESHOLD_SECONDS: int = 300 # This will be calculated from IDLE_TIME_MINUTES in __init__ if needed, or kept as is.
    # We can use a property or validator. 
    @property
    def idle_threshold_seconds(self) -> int:
        return self.IDLE_TIME_MINUTES * 60

    MEMORY_SUMMARY_MIN_MESSAGES: int = 10
    MEMORY_FACT_EXTRACTION_ENABLED: bool = False
    
    # Archival Configuration
    MEMORY_ARCHIVAL_ENABLED: bool = True
    MEMORY_RETENTION_DAYS: int = 30
    MEMORY_MAX_RECORDS: int = 10000
    MEMORY_ARCHIVE_BATCH_SIZE: int = 100

    # Multi-Agent Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    SEARXNG_URL: str = "http://localhost:8080"
    AGENT_DECISION_LOG_PATH: str = "./logs/agent_decisions.jsonl"

    class Config:
        # Use absolute path for .env file to work regardless of where the app is run from
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
        env_file_encoding = 'utf-8'

settings = Settings()
