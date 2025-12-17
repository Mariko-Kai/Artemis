
import asyncio
import logging
import psutil
from datetime import datetime, timedelta
from typing import Optional, List

from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.db.database import SessionLocal
from app.db.models import ChatSession, ChatMessage
from app.db.memory_models import JobQueue, MemoryRecordDB
from app.core.config import settings
from app.core.global_lock import gpu_lock
from app.core.llm_engine import llm_engine
from app.services.memory_service import memory_service

logger = logging.getLogger(__name__)

class SummarizationWorker:
    """
    Background worker for chat summarization.
    Respects GPU lock and system resources.
    """
    
    def __init__(self):
        self.running = False
        self._task = None
        self.check_interval = 60 # Check every 60 seconds
        self.idle_threshold = settings.MEMORY_IDLE_THRESHOLD_SECONDS

    async def start(self):
        if not settings.MEMORY_SUMMARIZATION_ENABLED:
            logger.info("Summarization disabled via config.")
            return

        self.running = True
        self._task = asyncio.create_task(self.run_loop())
        logger.info("Summarization Worker started.")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Summarization Worker stopped.")

    async def run_loop(self):
        while self.running:
            try:
                if self._check_resources():
                    await self.check_idle_sessions()
                    await self.process_queue()
                else:
                    logger.debug("Summarization skipped due to low resources.")
            except Exception as e:
                logger.error(f"Summarization loop error: {e}")
            
            await asyncio.sleep(self.check_interval)

    def _check_resources(self) -> bool:
        """Check if system has enough RAM (skip if < 2GB free)."""
        mem = psutil.virtual_memory()
        if mem.available < 2 * 1024 * 1024 * 1024: # 2GB
            return False
        return True

    async def check_idle_sessions(self):
        """Identify idle sessions and queue them."""
        db: Session = SessionLocal()
        try:
            cutoff = datetime.utcnow() - timedelta(seconds=self.idle_threshold)
            
            # Find sessions with last_activity < cutoff AND unprocessed messages
            # Optimization: Just check time first, then message count in worker or here.
            # We don't have a "summarized_up_to" pointer easily yet, assume we check if last message > last summary time?
            # Or just check if job already exists.
            
            sessions = db.query(ChatSession).filter(
                ChatSession.last_activity < cutoff
            ).all()

            for session in sessions:
                # Check if already pending
                existing_job = db.query(JobQueue).filter(
                    JobQueue.target_id == session.id,
                    or_(JobQueue.status == "pending", JobQueue.status == "processing")
                ).first()
                
                if not existing_job:
                    # Check message count
                    msg_count = len(session.messages)
                    if msg_count >= settings.MEMORY_SUMMARY_MIN_MESSAGES:
                        # Queue it
                        job = JobQueue(target_id=session.id, job_type="summarization", status="pending")
                        db.add(job)
                        logger.info(f"Queued summarization for session {session.id}")
            
            db.commit()
        except Exception as e:
            logger.error(f"Error checking idle sessions: {e}")
        finally:
            db.close()

    async def process_queue(self):
        """Process one pending job."""
        db: Session = SessionLocal()
        job = None
        try:
            job = db.query(JobQueue).filter(JobQueue.status == "pending").first()
            if not job:
                return

            # Lock job
            job.status = "processing"
            db.commit()
            
            logger.info(f"Processing summarization job {job.id} for session {job.target_id}")
            
            # Do work
            success = await self._summarize_session(job.target_id)
            
            if success:
                job.status = "completed"
                # Remove job or keep log? Keep for audit, maybe clean up old ones later.
            else:
                job.status = "failed"
                job.error = "Summarization logic returned False"
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Job processing failed: {e}")
            if job:
                job.status = "failed"
                job.error = str(e)
                job.retries += 1
                if job.retries < 3:
                    job.status = "pending" # Retry
                db.commit()
        finally:
            db.close()

    async def _summarize_session(self, session_id: str) -> bool:
        """
        Summarize the session.
        """
        db: Session = SessionLocal()
        try:
            # 1. Get messages
            # Logic: We should only summarize unsummarized messages? 
            # Or just summarize the tail? Prompt says "last 10-15 messages".
            messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id)\
                .order_by(ChatMessage.timestamp.desc()).limit(15).all()
            
            if not messages:
                return False
                
            messages.reverse() # Oldest to newest
            
            text_block = "\n".join([f"{m.role}: {m.content}" for m in messages])
            
            # 2. Generate Summary using LLM
            prompt = f"""Summarize the following conversation in Russian. Focus on key facts and decisions. Concise (max 3 sentences).

Conversation:
{text_block}

Summary:"""

            # Acquire Lock!
            async with gpu_lock:
                 # Check if LLM engine is loaded
                 model = llm_engine.get_model()
                 if not model:
                     return False
                 
                 response = await asyncio.to_thread(
                    model.create_completion,
                    prompt=prompt,
                    max_tokens=200,
                    stop=["\n\n"],
                    temperature=0.5
                 )
                 
            summary_text = response["choices"][0]["text"].strip()
            if not summary_text:
                return False

            # 3. Save as MemoryRecord
            # Mark it as 'summary' source
            await memory_service.store_memory(
                session_id=session_id,
                content=summary_text,
                role="summary",
                importance=0.8
            )
            
            # 4. Optional: Extract facts on CPU (regex/spacy)
            if settings.MEMORY_FACT_EXTRACTION_ENABLED:
                # Placeholder for lightweight entity extraction
                pass
                
            return True

        except Exception as e:
            logger.error(f"Summarization execution failed: {e}")
            raise e
        finally:
            db.close()

summarization_worker = SummarizationWorker()
