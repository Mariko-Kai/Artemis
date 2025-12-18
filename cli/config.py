"""
CLI configuration settings.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class CLISettings(BaseSettings):
    """CLI configuration."""
    
    # Server settings for HTTP mode
    api_base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for FastAPI server"
    )
    api_version: str = Field(
        default="v1",
        description="API version prefix"
    )
    
    # Display settings
    color_enabled: bool = Field(
        default=True,
        description="Enable colored output"
    )
    markdown_enabled: bool = Field(
        default=True,
        description="Render markdown in responses"
    )
    
    # HTTP client settings
    timeout: float = Field(
        default=120.0,
        description="Request timeout in seconds"
    )
    
    # Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Project root directory"
    )
    
    class Config:
        env_prefix = "ARTEMIS_CLI_"
        env_file = ".env"


# Global settings instance
_settings: Optional[CLISettings] = None


def get_settings() -> CLISettings:
    """Get CLI settings singleton."""
    global _settings
    if _settings is None:
        _settings = CLISettings()
    return _settings
