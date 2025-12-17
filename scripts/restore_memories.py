
import sys
import os
import shutil
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def restore_memories(backup_zip_path):
    if not os.path.exists(backup_zip_path):
        logger.error(f"Backup file not found: {backup_zip_path}")
        exit(1)
        
    logger.info(f"Restoring from {backup_zip_path}...")
    
    # Warning
    print("WARNING: This will overwrite current database and indices. Ensure Artemis is STOPPED.")
    confirm = input("Continue? (y/n): ")
    if confirm.lower() != 'y':
        logger.info("Aborted.")
        exit(0)
        
    try:
        with zipfile.ZipFile(backup_zip_path, 'r') as zipf:
            zipf.extractall(PROJECT_ROOT)
            logger.info("Restore complete.")
            
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python restore_memories.py <path_to_zip>")
        exit(1)
    restore_memories(sys.argv[1])
