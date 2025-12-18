
import pytest
import asyncio
from app.core.global_lock import gpu_lock
from unittest.mock import MagicMock, patch

@pytest.mark.asyncio
async def test_gpu_lock_preemption(client, gpu_spy):
    """
    Verify that a high-priority request (Chat) preempts a background task (Indexing).
    """
    background_task_started = asyncio.Event()
    background_task_preempted = asyncio.Event()
    
    async def simulate_background_work():
        async with gpu_lock.background() as preempt_trigger:
            background_task_started.set()
            # Simulate waiting for preemption
            try:
                # Wait for the trigger to be set by the priority request
                await asyncio.wait_for(preempt_trigger.wait(), timeout=5.0)
                background_task_preempted.set()
                # Simulate yielding
                gpu_lock.cleanup_confirmed.set()
            except asyncio.TimeoutError:
                pass

    # Start background task
    bg_task = asyncio.create_task(simulate_background_work())
    await background_task_started.wait()
    
    # Background task is now holding the lock (shared-ish or actual lock)
    assert gpu_lock._background_active is True
    
    # Fire high-priority chat request
    chat_request = {
        "messages": [{"role": "user", "content": "Hello, this is a priority request."}],
        "session_id": "session-456"
    }
    
    # This call should trigger preemption logic in gpu_lock.request_priority_access()
    response = await client.post("/v1/chat/completions", json=chat_request)
    
    assert response.status_code == 200
    assert background_task_preempted.is_set()
    
    # Verify the spies
    gpu_spy["priority"].assert_called()
    gpu_spy["background"].assert_called()
    
    await bg_task
