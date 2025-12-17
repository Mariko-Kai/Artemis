from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from playwright.async_api import async_playwright
import logging
import asyncio

logger = logging.getLogger(__name__)

# Search Tool
search_run = DuckDuckGoSearchRun()
search_tool = Tool(
    name="Search",
    func=search_run.run,
    description="Useful for when you need to answer questions about current events or find information on the internet."
)

# Browser Tool
async def browse_page(url: str) -> str:
    """Visits a URL and returns the text content of the page."""
    logger.info(f"Browsing URL: {url}")
    try:
        async with async_playwright() as p:
            # Launch headless browser
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Go to URL with timeout
            await page.goto(url, timeout=10000)
            
            # Get text content
            text = await page.inner_text("body")
            
            await browser.close()
            
            # Truncate if too long (simple heuristic)
            return text[:4000] 
    except Exception as e:
        logger.error(f"Error browsing {url}: {e}")
        return f"Error reading page: {str(e)}"

# Wrapper for LangChain
def browser_func(url: str) -> str:
    # Since LangChain tools are often synchronous or expect specific run patterns,
    # and Playwright is async, we need to manage the loop.
    # However, running async code from sync tool can be tricky.
    # We will use asyncio.run for simplicity if the event loop isn't already running,
    # but inside FastAPI, an event loop IS running.
    # So we should probably expose this as an AsyncTool or handle loop safely.
    
    # For this MVP, we'll try to use a simple run_until_complete if possible,
    # or better, define it as an async tool if the executor supports it.
    # But standard Tool is sync. Let's try basic asyncio.run() and see if it clashes.
    # WARNING: nesting asyncio.run inside a running loop fails.
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        # If we are here, we are likely in the FastAPI loop.
        # We can't block it with loop.run_until_complete.
        # Ideally, we should use an async agent. 
        # For now, let's just return a message saying "Async Execution Required" 
        # or implement a thread workaround.
        
        # ACTUALLY: The easiest fix for MVP is to run playwright in a separate thread 
        # where we create a NEW loop.
        import threading
        
        result = []
        def run_in_thread(u, res):
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            res.append(new_loop.run_until_complete(browse_page(u)))
            new_loop.close()
            
        t = threading.Thread(target=run_in_thread, args=(url, result))
        t.start()
        t.join()
        return result[0]
            
    else:
        return asyncio.run(browse_page(url))

browser_tool = Tool(
    name="Browser",
    func=browser_func,
    description="Useful for reading the content of a specific webpage. Input should be a valid URL."
)

ALL_TOOLS = [search_tool, browser_tool]
