from faster_whisper import WhisperModel
from app.core.config import settings
from app.core.global_lock import gpu_lock
import logging
import asyncio
import os

logger = logging.getLogger(__name__)

class ASREngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ASREngine, cls).__new__(cls)
            cls._instance.model = None
        return cls._instance

    def load_model(self):
        if self.model:
            return

        # Model size: tiny, base, small, medium, large-v2
        # 'small' is good balance for 4GB VRAM
        model_size = "small" 
        
        logger.info(f"Loading Whisper model '{model_size}' on GPU...")
        try:
            self.model = WhisperModel(
                model_size, 
                device="cuda", 
                compute_type="int8_float16" # Save memory
            )
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise e

    async def transcribe(self, file_path: str) -> str:
        if not self.model:
            self.load_model()
        
        logger.info(f"Acquiring GPU lock for transcription of {file_path}...")
        async with gpu_lock:
            logger.info("GPU lock acquired. Starting transcription.")
            segments, info = await asyncio.to_thread(
                self.model.transcribe, 
                file_path, 
                beam_size=5
            )
            
            # segments is a generator, so we must iterate to get the result
            # Using list() consumes the generator and runs inference
            # We run this in a thread to avoid blocking the event loop
            segment_list = await asyncio.to_thread(list, segments)
            
            text = " ".join([segment.text for segment in segment_list]).strip()
            logger.info("Transcription complete.")
            return text

asr_engine = ASREngine()
