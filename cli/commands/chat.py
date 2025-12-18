"""
Chat command - interactive dialogue with LLM.
"""
import typer
from typing import Optional
from cli.utils.output import (
    console, print_message, print_thinking, print_error, 
    print_welcome, print_info
)
from cli.utils.http_client import get_client
import sys
import io

# Fix for Unicode issues in some terminals
if sys.stdin.encoding != 'utf-8':
    try:
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    except Exception:
        pass

app = typer.Typer(help="💬 Chat with Artemis LLM")


def _direct_chat(messages: list, session_id: Optional[str] = None) -> str:
    """Chat using direct Python service calls."""
    import sys
    import os
    
    # Add project paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "backend"))
    
    import asyncio
    from app.core.llm_engine import llm_engine
    from app.core.global_lock import gpu_lock
    
    # Ensure model is loaded
    if llm_engine.model is None:
        print_info("Loading LLM model...")
        llm_engine.load_model()
    
    async def run_inference():
        async with gpu_lock:
            response = await asyncio.to_thread(
                llm_engine.model.create_chat_completion,
                messages=messages,
                temperature=0.7
            )
            return response["choices"][0]["message"]["content"]
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(run_inference())


@app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context,
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Single message to send"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID to use"),
    http: bool = typer.Option(False, "--http", help="Use HTTP mode (requires running server)"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Generation temperature")
):
    """Start interactive chat or send a single message."""
    
    if message:
        # Single message mode
        messages = [{"role": "user", "content": message}]
        
        with print_thinking("Generating response..."):
            try:
                if http:
                    client = get_client()
                    response = client.chat(messages, session_id=session, temperature=temperature)
                    reply = response["choices"][0]["message"]["content"]
                else:
                    reply = _direct_chat(messages, session)
            except Exception as e:
                print_error(str(e))
                raise typer.Exit(1)
        
        print_message("user", message)
        print_message("assistant", reply)
    else:
        # Interactive mode
        print_welcome()
        history = []
        
        while True:
            try:
                user_input = console.input("\n[bold blue]You:[/] ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ("exit", "quit", "/q"):
                    print_info("Goodbye! 👋")
                    break
                
                history.append({"role": "user", "content": user_input})
                
                with print_thinking("Thinking..."):
                    try:
                        if http:
                            client = get_client()
                            response = client.chat(history, session_id=session, temperature=temperature)
                            reply = response["choices"][0]["message"]["content"]
                        else:
                            reply = _direct_chat(history, session)
                        
                        history.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        print_error(str(e))
                        continue
                
                print_message("assistant", reply)
                
            except KeyboardInterrupt:
                print_info("\nGoodbye! 👋")
                break
