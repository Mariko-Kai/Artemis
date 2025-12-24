import os
import sys

# Add backend to path so we can import app.core.config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app.core.config import settings
    print(f"DATABASE_URL: {settings.DATABASE_URL}")
    print(f"ENV_FILE: {settings.Config.env_file}")
    
    # Check if directory exists
    db_path = settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")
    print(f"Extracted DB Path: {db_path}")
    
    db_dir = os.path.dirname(db_path)
    print(f"DB Dir: {db_dir}, Exists: {os.path.exists(db_dir)}, Writable: {os.access(db_dir, os.W_OK)}")
    
    # Check if DB file exists
    if os.path.exists(db_path):
        print(f"DB File exists. Writable: {os.access(db_path, os.W_OK)}")
        # Try opening it
        try:
            fd = os.open(db_path, os.O_RDWR)
            os.close(fd)
            print("DB File valid and accessible via os.open")
        except Exception as e:
            print(f"Failed to open DB file: {e}")
    else:
        print("DB File does not exist (will be created)")
        
except Exception as e:
    print(f"Error loading settings: {e}")
