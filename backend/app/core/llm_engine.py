from llama_cpp import Llama
from app.core.config import settings
from app.core.global_lock import gpu_lock
import logging
import asyncio

logger = logging.getLogger(__name__)

class LLMEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMEngine, cls).__new__(cls)
            cls._instance.model = None
        return cls._instance

    def load_model(self):
        if self.model:
            return

        logger.info(f"Loading model from {settings.MODEL_PATH} with {settings.N_GPU_LAYERS} GPU layers...")
        try:
            self.model = Llama(
                model_path=settings.MODEL_PATH,
                n_gpu_layers=settings.N_GPU_LAYERS,
                n_ctx=settings.CONTEXT_WINDOW,
                verbose=True
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def get_model(self) -> Llama:
        if not self.model:
            self.load_model()
        return self.model

llm_engine = LLMEngine()
