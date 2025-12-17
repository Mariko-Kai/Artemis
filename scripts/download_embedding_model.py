
import os
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    model_name = "Snowflake/snowflake-arctic-embed-m"
    logger.info(f"Downloading embedding model: {model_name}")
    
    # This will cache the model in ~/.cache/huggingface/hub or local directory if specified
    # We let sentence-transformers handle the default cache
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
        logger.info(f"Successfully downloaded and loaded {model_name}")
        
        # Verify basic functionality
        embedding = model.encode("Test sentence")
        logger.info(f"Verification successful. Embedding shape: {embedding.shape}")
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        exit(1)

if __name__ == "__main__":
    download_model()
