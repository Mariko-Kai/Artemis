"""
FastAPI application for Memory Module.

Provides RESTful API for memory storage and search.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config.settings import get_settings
from .dependencies import get_memory_service
from .routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for startup and shutdown events.

    Args:
        app: FastAPI application

    Yields:
        None
    """
    # Startup
    logger.info("Starting Memory Module API...")
    settings = get_settings()

    # Initialize memory service (creates DB tables, loads indexes)
    try:
        service = await get_memory_service()
        logger.info("Memory service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize memory service: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down Memory Module API...")
    # Save indexes on shutdown
    try:
        await service.save_indexes()
        logger.info("Indexes saved successfully")
    except Exception as e:
        logger.error(f"Failed to save indexes on shutdown: {e}", exc_info=True)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI application instance
    """
    settings = get_settings()

    app = FastAPI(
        title="Memory Module API",
        description="Semantic memory and hybrid search API for Artemis",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(router, prefix="/api/v1", tags=["memories"])

    # Health check endpoint
    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "memory-module",
            "version": "0.1.0",
        }

    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "error": str(exc),
            },
        )

    return app


# Create app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
