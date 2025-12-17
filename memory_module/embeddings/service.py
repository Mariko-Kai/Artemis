"""
Embedding service implementation.
"""

import logging
from typing import List, Optional, Any

from ..config.settings import Settings
from .arctic import ArcticEmbedService

logger = logging.getLogger(__name__)

class EmbeddingService(ArcticEmbedService):
    """
    Embedding service implementation.
    Inherits from ArcticEmbedService which implements IEmbeddingService.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the embedding service.
        args are accepted for compatibility but configuration is loaded from global settings
        in the parent class unless we refactor parent.
        """
        super().__init__()
        # If settings were passed, we could theoretically override self.* properties 
        # but ArcticEmbedService loads from get_settings(). 
        # Ideally we should respect passed settings.
        if settings:
             self.model_name = settings.embedding_model_name
             self.device = settings.embedding_device
             self.default_dim = settings.vector_dim
             # ... other settings if needed
        
        logger.info("Embedding service initialized (Wrapper)")
