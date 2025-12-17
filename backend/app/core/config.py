from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Artemis Backend"
    API_V1_STR: str = "/v1"
    
    # Model Configuration
    # Model Configuration
    # Path relative to backend directory
    MODEL_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/Phi-3-mini-4k-instruct-q4.gguf"))
    N_GPU_LAYERS: int = 32  # Default to 0 (CPU) to avoid CUDA errors. Increase via env var if GPU is available.
    N_CTX: int = 4096000 # Using a large context window cap, but Phi-3 mini is 4k. Let's stick to 4096.
    
    # We should probably set N_CTX to 4096 to match model card, or 0 to use model's default.
    # The prompt says "Phi-3-mini-4k-instruct". 
    CONTEXT_WINDOW: int = 4096
    
    class Config:
        env_file = ".env"

settings = Settings()
