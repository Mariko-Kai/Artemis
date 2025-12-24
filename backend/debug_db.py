import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Absolute path based on current file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ABS_PATH = os.path.join(BASE_DIR, "sessions.db")

URLS_TO_TEST = [
    f"sqlite+aiosqlite:///{ABS_PATH}", # Absolute with 4 slashes (unix style)
    "sqlite+aiosqlite:///sessions.db", # Relative simple
    "sqlite+aiosqlite:///./sessions.db", # Relative with dot
    "sqlite+aiosqlite:////tmp/sessions.db" # Tmp directory
]

async def test_url(url):
    print(f"\nTesting URL: {url}")
    try:
        engine = create_async_engine(url)
        async with engine.begin() as conn:
            # Try to execute simple query
            await conn.execute(text("SELECT 1"))
            print(f"SUCCESS: {url}")
            return True
    except Exception as e:
        print(f"FAILED: {url}")
        print(f"Error: {e}")
        return False
    finally:
        await engine.dispose()

async def main():
    print(f"Base Dir: {BASE_DIR}")
    for url in URLS_TO_TEST:
        if await test_url(url):
            print("Found working configuration!")
            # don't break, see what else works
            
if __name__ == "__main__":
    asyncio.run(main())
