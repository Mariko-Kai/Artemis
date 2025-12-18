"""
Memory command - interact with semantic memory.
"""
import typer
from typing import Optional
from pathlib import Path
from cli.utils.output import (
    console, print_memory_results, print_thinking, print_error,
    print_success, print_info, print_table
)
from cli.utils.http_client import get_client

app = typer.Typer(help="🧠 Manage semantic memory")


@app.command("query")
def query_memory(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Filter by session"),
    http: bool = typer.Option(False, "--http", help="Use HTTP mode")
):
    """
    Search memories semantically.
    
    Example: artemis memory query "previous discussion about AI"
    """
    with print_thinking("Searching memories..."):
        try:
            if http:
                client = get_client()
                results = client.query_memory(query, session_id=session, top_k=top_k)
            else:
                # Direct mode
                import sys
                import os
                import asyncio
                
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                sys.path.insert(0, project_root)
                sys.path.insert(0, os.path.join(project_root, "backend"))
                
                from app.services.memory_service import memory_service
                
                async def run_query():
                    await memory_service.initialize()
                    return await memory_service.query_memory(query, session_id=session, top_k=top_k)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(run_query())
        except Exception as e:
            print_error(str(e))
            raise typer.Exit(1)
    
    print_memory_results(results)


@app.command("stats")
def memory_stats(
    http: bool = typer.Option(False, "--http", help="Use HTTP mode")
):
    """Show memory statistics."""
    with print_thinking("Loading stats..."):
        try:
            if http:
                client = get_client()
                stats = client.get_archive_stats()
            else:
                import sys
                import os
                
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                sys.path.insert(0, project_root)
                sys.path.insert(0, os.path.join(project_root, "backend"))
                
                from app.db.database import SessionLocal
                from app.db.memory_models import MemoryRecordDB, ArchivedMemoryRecordDB
                from sqlalchemy import func
                
                db = SessionLocal()
                try:
                    active_count = db.query(func.count(MemoryRecordDB.id)).scalar() or 0
                    archived_count = db.query(func.count(ArchivedMemoryRecordDB.id)).scalar() or 0
                    stats = {
                        "active_memories": active_count,
                        "archived_memories": archived_count,
                        "total": active_count + archived_count
                    }
                finally:
                    db.close()
        except Exception as e:
            print_error(str(e))
            raise typer.Exit(1)
    
    console.print("\n[bold]📊 Memory Statistics[/]\n")
    for key, value in stats.items():
        console.print(f"  [cyan]{key.replace('_', ' ').title()}:[/] {value}")


@app.command("export")
def export_memories(
    output: Path = typer.Option(
        Path("memories_export.jsonl"), 
        "--output", "-o", 
        help="Output file path"
    ),
    http: bool = typer.Option(False, "--http", help="Use HTTP mode")
):
    """Export all memories to JSONL file."""
    with print_thinking("Exporting memories..."):
        try:
            if http:
                client = get_client()
                data = client.export_memories()
            else:
                import sys
                import os
                import asyncio
                
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                sys.path.insert(0, project_root)
                sys.path.insert(0, os.path.join(project_root, "backend"))
                
                from app.services.archival_service import archival_service
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                data = loop.run_until_complete(archival_service.export_all())
            
            output.write_text(data, encoding="utf-8")
        except Exception as e:
            print_error(str(e))
            raise typer.Exit(1)
    
    print_success(f"Exported to {output}")


@app.command("store")
def store_memory(
    content: str = typer.Argument(..., help="Content to store"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID"),
    http: bool = typer.Option(False, "--http", help="Use HTTP mode")
):
    """
    Store a new memory.
    
    Example: artemis memory store "Important fact to remember"
    """
    with print_thinking("Storing memory..."):
        try:
            if http:
                # HTTP store not directly exposed, use memory service directly
                print_info("HTTP store not implemented yet, using direct mode")
            
            import sys
            import os
            import asyncio
            
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            sys.path.insert(0, project_root)
            sys.path.insert(0, os.path.join(project_root, "backend"))
            
            from app.services.memory_service import memory_service
            
            async def run_store():
                await memory_service.initialize()
                await memory_service.store_memory(
                    session_id=session or "cli-session",
                    content=content,
                    role="user",
                    importance=0.8
                )
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_store())
        except Exception as e:
            print_error(str(e))
            raise typer.Exit(1)
    
    print_success("Memory stored successfully")
