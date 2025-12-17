"""
Dependency injection for FastAPI.

Provides shared instances and database sessions.
"""

from typing import AsyncGenerator

from ..config.settings import Settings, get_settings
from ..service import MemoryService

# Global service instance
_memory_service: MemoryService | None = None


async def get_memory_service() -> MemoryService:
    """
    Get the global memory service instance.

    Returns:
        MemoryService instance
    """
    global _memory_service

    if _memory_service is None:
        settings = get_settings()
        _memory_service = MemoryService(settings)
        await _memory_service.initialize()

    return _memory_service


def get_settings_dependency() -> Settings:
    """
    Get settings for dependency injection.

    Returns:
        Settings instance
    """
    return get_settings()
