import asyncio
import time
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class PreemptibleGPULock:
    """
    Enhanced GPU Lock that prioritizes user requests over background tasks.
    Background tasks can be signaled to stop immediately.
    """
    def __init__(self):
        self._lock = asyncio.Lock()
        self.preempt_requested = asyncio.Event()
        self.cleanup_confirmed = asyncio.Event()
        self.last_activity = time.time()
        self._background_active = False

    def _update_activity(self):
        self.last_activity = time.time()

    @asynccontextmanager
    async def request_priority_access(self):
        """
        Explicitly request GPU access, triggering preemption and waiting for cleanup.
        """
        self.preempt_requested.set()
        self.cleanup_confirmed.clear()
        
        logger.debug("High priority GPU request initiated")
        
        # If background is active, wait for it to signal cleanup completion
        if self._background_active:
            logger.info("Background task active, waiting for VRAM cleanup...")
            try:
                # Wait for cleanup with a strict timeout (1-2s as requested)
                await asyncio.wait_for(self.cleanup_confirmed.wait(), timeout=2.0)
                logger.debug("Cleanup confirmed by background service")
            except asyncio.TimeoutError:
                logger.warning("Cleanup timeout reached, proceeding with lock acquisition anyway")

        async with self._lock:
            self.preempt_requested.clear()
            self.cleanup_confirmed.clear()
            self._update_activity()
            try:
                yield
            finally:
                self._update_activity()

    @asynccontextmanager
    async def high_priority(self):
        """Standard acquisition for user requests. Uses priority handshake."""
        async with self.request_priority_access():
            yield

    @asynccontextmanager
    async def background(self):
        """Acquisition for background tasks. Can be preempted."""
        if self.preempt_requested.is_set():
            return # Don't even try if preemption is active

        async with self._lock:
            self._background_active = True
            try:
                yield self.preempt_requested
            finally:
                self._background_active = False

    def __aenter__(self):
        """Default behavior is high priority for backward compatibility."""
        self._high_priority_ctx = self.high_priority()
        return self._high_priority_ctx.__aenter__()

    def __aexit__(self, exc_type, exc_val, exc_tb):
        return self._high_priority_ctx.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def locked(self):
        return self._lock.locked()

gpu_lock = PreemptibleGPULock()
