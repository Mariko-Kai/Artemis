import asyncio

class GlobalLock:
    _instance = None
    _lock = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalLock, cls).__new__(cls)
            cls._instance._lock = asyncio.Lock()
        return cls._instance

    @property
    def lock(self):
        return self._lock

gpu_lock = GlobalLock().lock
