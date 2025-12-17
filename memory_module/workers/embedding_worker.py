"""
Background worker for embedding generation.

Provides async queue-based processing for batch embedding tasks.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class EmbeddingWorker:
    """
    Background worker for async embedding processing.

    This can be extended to use a proper task queue like Celery or RQ.
    """

    def __init__(self, queue_size: int = 100):
        """
        Initialize embedding worker.

        Args:
            queue_size: Maximum queue size
        """
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.running = False
        self.worker_task: asyncio.Task | None = None
        logger.info("Embedding worker initialized")

    async def start(self, processor: Callable) -> None:
        """
        Start the worker.

        Args:
            processor: Async function to process queue items
        """
        if self.running:
            logger.warning("Worker already running")
            return

        self.running = True
        self.worker_task = asyncio.create_task(self._worker_loop(processor))
        logger.info("Embedding worker started")

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        if not self.running:
            return

        self.running = False
        if self.worker_task:
            await self.worker_task
        logger.info("Embedding worker stopped")

    async def enqueue(self, item: Dict[str, Any]) -> None:
        """
        Add an item to the queue.

        Args:
            item: Item to process
        """
        await self.queue.put(item)
        logger.debug(f"Enqueued item, queue size: {self.queue.qsize()}")

    async def _worker_loop(self, processor: Callable) -> None:
        """
        Main worker loop.

        Args:
            processor: Function to process items
        """
        while self.running:
            try:
                # Wait for item with timeout
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)

                # Process item
                try:
                    await processor(item)
                except Exception as e:
                    logger.error(f"Error processing item: {e}", exc_info=True)
                finally:
                    self.queue.task_done()

            except asyncio.TimeoutError:
                # No items in queue, continue
                continue
            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)

        logger.info("Worker loop exited")
