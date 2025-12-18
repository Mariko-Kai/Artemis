import asyncio
import logging
import time
import gc
from datetime import datetime, timedelta, timezone
from typing import List, Optional

try:
    import torch
except ImportError:
    torch = None

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.database import SessionLocal
from app.db.memory_models import MemoryRecordDB
from app.core.config import settings
from app.core.global_lock import gpu_lock
from app.core.llm_engine import llm_engine
from app.services.memory_service import memory_service
from memory_module.embeddings.service import EmbeddingService
from memory_module.models import MemoryStatus

logger = logging.getLogger(__name__)

class DeferredProcessingService:
    """
    Background service for deferred processing of memory records.
    Coordinates summarization and indexing tasks during GPU idle time.
    """
    def __init__(self):
        self.running = False
        self._task = None
        self.idle_threshold = settings.MEMORY_IDLE_THRESHOLD_SECONDS
        self.check_interval = 30 # Check every 30 seconds
        self.embedding_service: Optional[EmbeddingService] = None

    async def start(self):
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self.run_loop())
            logger.info("Deferred Processing Service started.")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Deferred Processing Service stopped.")

    async def run_loop(self):
        while self.running:
            try:
                # Check if GPU is idle
                idle_time = time.time() - gpu_lock.last_activity
                if idle_time > settings.idle_threshold_seconds and not gpu_lock.locked:
                    logger.info(f"GPU idle for {idle_time:.1f}s. Starting deferred tasks.")
                    await self.process_tasks()
                else:
                    logger.debug(f"GPU not idle enough ({idle_time:.1f}s / {settings.idle_threshold_seconds}s)")
            except Exception as e:
                logger.error(f"Error in deferred processing loop: {e}")
            
            await asyncio.sleep(self.check_interval)

    async def process_tasks(self):
        """Execute summarization and indexing tasks."""
        # 1. Summarization Task
        await self.task_summarize_pending()
        
        # 2. Indexing Task
        await self.task_index_pending()

    async def task_summarize_pending(self):
        """Summarize records with status 'pending_summary'."""
        db: Session = SessionLocal()
        try:
            records = db.query(MemoryRecordDB).filter(
                MemoryRecordDB.status == MemoryStatus.PENDING_SUMMARY.value
            ).limit(settings.SUMMARY_BATCH_SIZE).all()

            if not records:
                return

            logger.info(f"Found {len(records)} records pending summary (batching {settings.SUMMARY_BATCH_SIZE}).")

            for record in records:
                async with gpu_lock.background() as preempt_trigger:
                    if preempt_trigger is None: # Lock already taken or preemption active
                        break
                    
                    if preempt_trigger.is_set():
                        # Signal we are yielding
                        gpu_lock.cleanup_confirmed.set()
                        break

                    success = await self._summarize_record(record, preempt_trigger)
                    if not success:
                        if preempt_trigger.is_set():
                            gpu_lock.cleanup_confirmed.set()
                        break
                    
                    db.commit() # Save progress per record

        finally:
            db.close()

    async def _summarize_record(self, record: MemoryRecordDB, preempt_trigger: asyncio.Event) -> bool:
        """Perform actual summarization via LLM."""
        try:
            model = llm_engine.get_model()
            if not model:
                return False

            prompt = f"Summarize the following text concisely (max 2 sentences):\n\n{record.content}\n\nSummary:"
            
            # Check preemption before heavy call
            if preempt_trigger.is_set():
                return False

            # Llama-cpp doesn't support easy cancellation mid-inference via standard Python wrapper.
            response = await asyncio.to_thread(
                model.create_completion,
                prompt=prompt,
                max_tokens=100,
                temperature=0.3
            )

            if preempt_trigger.is_set():
                return False

            summary = response["choices"][0]["text"].strip()
            record.summary = summary
            record.status = MemoryStatus.PENDING_EMBEDDING.value
            record.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            
            logger.info(f"Summarized record {record.id}")
            return True

        except Exception as e:
            logger.error(f"Summarization failed for record {record.id}: {e}")
            return False

    async def task_index_pending(self):
        """Generate embeddings and index records with status 'pending_embedding'."""
        db: Session = SessionLocal()
        try:
            records = db.query(MemoryRecordDB).filter(
               MemoryRecordDB.status == MemoryStatus.PENDING_EMBEDDING.value
            ).limit(settings.SUMMARY_BATCH_SIZE).all()

            if not records:
                return

            logger.info(f"Found {len(records)} records pending embedding (batching {settings.SUMMARY_BATCH_SIZE}).")

            async with gpu_lock.background() as preempt_trigger:
                if preempt_trigger is None or preempt_trigger.is_set():
                    if preempt_trigger and preempt_trigger.is_set():
                        gpu_lock.cleanup_confirmed.set()
                    return

                # Load Embedding Model only when needed
                if self.embedding_service is None:
                    logger.info("Loading Embedding Model into VRAM...")
                    self.embedding_service = EmbeddingService(settings)

                for record in records:
                    if preempt_trigger.is_set():
                        self._cleanup_vram()
                        gpu_lock.cleanup_confirmed.set()
                        return

                    success = await self._index_record(record, preempt_trigger)
                    if not success:
                        if preempt_trigger.is_set():
                            self._cleanup_vram()
                            gpu_lock.cleanup_confirmed.set()
                        return
                    
                    db.commit()

        finally:
            db.close()

    async def _index_record(self, record: MemoryRecordDB, preempt_trigger: asyncio.Event) -> bool:
        """Generate embedding and add to FAISS."""
        try:
            if preempt_trigger.is_set():
                return False

            embedding = await self.embedding_service.embed(record.content)

            if preempt_trigger.is_set():
                return False

            # Update DB
            import pickle
            record.embedding_blob = pickle.dumps(embedding)
            record.status = MemoryStatus.COMPLETED.value
            record.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

            from app.services.memory_service import memory_service as ams
            await ams.vector_store.add(
                id=str(record.id),
                vector=embedding,
                metadata={"content": record.content, "channel_id": record.channel_id}
            )

            await ams.lexical_index.index_document(
                id=str(record.id),
                text=record.content,
                metadata={"content": record.content}
            )

            logger.info(f"Indexed record {record.id}")
            return True

        except Exception as e:
            logger.error(f"Indexing failed for record {record.id}: {e}")
            return False

    def _cleanup_vram(self):
        """Unload embedding model and clear CUDA cache."""
        logger.info("Preemption triggered: Cleaning up VRAM...")
        if self.embedding_service:
            if hasattr(self.embedding_service, '_model'):
                self.embedding_service._model = None
            self.embedding_service = None
        
        # Trigger garbage collection to free memory
        gc.collect()
        
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VRAM cleanup completed.")

deferred_processing_service = DeferredProcessingService()
