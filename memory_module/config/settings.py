"""
Configuration management using Pydantic Settings.

Supports loading configuration from:
1. YAML configuration file
2. Environment variables (with MM_ prefix)
3. .env file
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Main configuration settings for the memory module."""

    # Embedding configuration
    embedding_model_name: str = Field(
        default="Snowflake/snowflake-arctic-embed-m",
        description="Hugging Face model name for embeddings",
    )
    embedding_device: str = Field(
        default="cpu", description="Device for embedding model (cpu/cuda)"
    )
    vector_dim: int = Field(default=768, description="Dimension of embedding vectors")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")

    # Database configuration
    database_url: str = Field(
        default="sqlite+aiosqlite:///./memory_module/memory.db",
        description="Database connection URL",
    )

    # Vector store configuration
    faiss_index_path: str = Field(
        default="./memory_module/faiss_index", description="Path to FAISS index directory"
    )
    faiss_index_type: str = Field(
        default="Flat", description="FAISS index type (Flat, IVF, HNSW)"
    )

    # Redis configuration
    redis_url: Optional[str] = Field(
        default=None, description="Redis connection URL for caching"
    )
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # Search configuration
    default_top_k: int = Field(default=10, description="Default number of search results")
    max_top_k: int = Field(default=100, description="Maximum allowed top_k value")
    default_hybrid_weight: float = Field(
        default=0.5, description="Default weight for hybrid search"
    )

    # API configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8001, description="API port")
    api_reload: bool = Field(default=False, description="Enable auto-reload for development")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins",
    )

    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )

    # Worker configuration
    enable_background_workers: bool = Field(
        default=True, description="Enable background workers"
    )
    worker_queue_size: int = Field(default=100, description="Worker queue size")

    # Performance configuration
    max_concurrent_requests: int = Field(
        default=10, description="Maximum concurrent API requests"
    )

    model_config = SettingsConfigDict(
        env_prefix="MM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Settings":
        """
        Load settings from a YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Settings instance
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Loads settings from:
    1. YAML file if MM_CONFIG_PATH is set
    2. Environment variables
    3. .env file
    4. Default values

    Returns:
        Settings instance
    """
    global _settings

    if _settings is None:
        config_path = os.getenv("MM_CONFIG_PATH")
        if config_path and os.path.exists(config_path):
            _settings = Settings.from_yaml(config_path)
        else:
            _settings = Settings()

    return _settings


def reload_settings() -> Settings:
    """
    Reload settings, useful for testing or configuration updates.

    Returns:
        New settings instance
    """
    global _settings
    _settings = None
    return get_settings()
