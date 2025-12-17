
import shutil
import os
import datetime
import sqlite3
import logging
import zipfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "sessions.db")
BACKUP_DIR = os.path.join(PROJECT_ROOT, "backups")
# Index paths (Assuming defaults from config, usually in root or memory_module)
# We need to find where FAISS/BM25 are stored. Based on previous analysis: 
# FAISS: ./faiss_index (relative to run dir?) or configured.
# Let's assume they are in PROJECT_ROOT or PROJECT_ROOT/storage.
# The code uses `index_path` passed to constructor.
# For safety, we will backup common locations or read config.
# Simplified: Backup specific files if they exist.

FILES_TO_BACKUP = [
    "sessions.db",
    "faiss_index.bin", # If file based
    "faiss_index/index.faiss", # If dir based
    "bm25_index.pkl",
    "memory_stats.json"
]

DIRS_TO_BACKUP = [
    "faiss_index" # Often a directory
]

def backup_memories():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"memory_backup_{timestamp}.zip")
    
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    logger.info(f"Starting backup to {backup_path}")
    
    with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 1. Backup Database (Safe copy using VACUUM into new file or just file copy if app stopped)
        # Using SQLite API to backup safely while running
        try:
            if os.path.exists(DB_PATH):
                logger.info("Backing up SQLite database...")
                # Create temp copy via API
                temp_db = DB_PATH + ".backup"
                src = sqlite3.connect(DB_PATH)
                dst = sqlite3.connect(temp_db)
                with src:
                    src.backup(dst)
                dst.close()
                src.close()
                
                zipf.write(temp_db, "sessions.db")
                os.remove(temp_db)
            else:
                logger.warning(f"Database not found at {DB_PATH}")
                
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            
        # 2. Backup Indices
        for item in FILES_TO_BACKUP:
            if item == "sessions.db": continue # Handled above
            
            path = os.path.join(PROJECT_ROOT, item)
            if os.path.exists(path):
                logger.info(f"Backing up file: {item}")
                zipf.write(path, item)
                
        for item in DIRS_TO_BACKUP:
            path = os.path.join(PROJECT_ROOT, item)
            if os.path.exists(path):
                logger.info(f"Backing up directory: {item}")
                for root, dirs, files in os.walk(path):
                    for file in files:
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, PROJECT_ROOT)
                        zipf.write(abs_path, rel_path)

    logger.info("Backup complete.")

if __name__ == "__main__":
    backup_memories()
