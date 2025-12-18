"""
Session command - manage chat sessions.
"""
import typer
from typing import Optional
from cli.utils.output import (
    console, print_table, print_thinking, print_error,
    print_success, print_info, print_message
)
from cli.utils.http_client import get_client

app = typer.Typer(help="📋 Manage chat sessions")


@app.command("list")
def list_sessions(
    http: bool = typer.Option(False, "--http", help="Use HTTP mode")
):
    """List all chat sessions."""
    with print_thinking("Loading sessions..."):
        try:
            if http:
                client = get_client()
                sessions = client.list_sessions()
            else:
                import sys
                import os
                
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                sys.path.insert(0, project_root)
                sys.path.insert(0, os.path.join(project_root, "backend"))
                
                from app.db.database import SessionLocal
                from app.db.models import ChatSession
                
                db = SessionLocal()
                try:
                    db_sessions = db.query(ChatSession).order_by(ChatSession.last_activity.desc()).all()
                    sessions = [
                        {
                            "id": s.id[:8] + "...",
                            "title": s.title or "Untitled",
                            "created": str(s.created_at)[:16] if s.created_at else "N/A",
                            "last_activity": str(s.last_activity)[:16] if s.last_activity else "N/A"
                        }
                        for s in db_sessions
                    ]
                finally:
                    db.close()
        except Exception as e:
            print_error(str(e))
            raise typer.Exit(1)
    
    if sessions:
        print_table(sessions, title="📋 Chat Sessions", columns=["id", "title", "created", "last_activity"])
    else:
        print_info("No sessions found")


@app.command("create")
def create_session(
    title: Optional[str] = typer.Option(None, "--title", "-t", help="Session title"),
    http: bool = typer.Option(False, "--http", help="Use HTTP mode")
):
    """Create a new chat session."""
    with print_thinking("Creating session..."):
        try:
            if http:
                client = get_client()
                result = client.create_session(title)
            else:
                import sys
                import os
                import uuid
                from datetime import datetime
                
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                sys.path.insert(0, project_root)
                sys.path.insert(0, os.path.join(project_root, "backend"))
                
                from app.db.database import SessionLocal
                from app.db.models import ChatSession
                
                db = SessionLocal()
                try:
                    new_session = ChatSession(
                        id=str(uuid.uuid4()),
                        title=title or "CLI Session",
                        created_at=datetime.utcnow(),
                        last_activity=datetime.utcnow()
                    )
                    db.add(new_session)
                    db.commit()
                    result = {"id": new_session.id, "title": new_session.title}
                finally:
                    db.close()
        except Exception as e:
            print_error(str(e))
            raise typer.Exit(1)
    
    print_success(f"Created session: {result.get('id', 'unknown')}")
    console.print(f"  Title: {result.get('title', 'N/A')}")


@app.command("delete")
def delete_session(
    session_id: str = typer.Argument(..., help="Session ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    http: bool = typer.Option(False, "--http", help="Use HTTP mode")
):
    """Delete a chat session."""
    if not force:
        confirm = typer.confirm(f"Delete session {session_id}?")
        if not confirm:
            print_info("Cancelled")
            raise typer.Exit(0)
    
    with print_thinking("Deleting session..."):
        try:
            if http:
                client = get_client()
                client.delete_session(session_id)
            else:
                import sys
                import os
                
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                sys.path.insert(0, project_root)
                sys.path.insert(0, os.path.join(project_root, "backend"))
                
                from app.db.database import SessionLocal
                from app.db.models import ChatSession, ChatMessage
                
                db = SessionLocal()
                try:
                    # Delete messages first
                    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
                    # Delete session
                    deleted = db.query(ChatSession).filter(ChatSession.id == session_id).delete()
                    db.commit()
                    
                    if not deleted:
                        print_error("Session not found")
                        raise typer.Exit(1)
                finally:
                    db.close()
        except Exception as e:
            print_error(str(e))
            raise typer.Exit(1)
    
    print_success(f"Deleted session {session_id}")


@app.command("history")
def session_history(
    session_id: str = typer.Argument(..., help="Session ID"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of messages"),
    http: bool = typer.Option(False, "--http", help="Use HTTP mode")
):
    """Show message history for a session."""
    with print_thinking("Loading history..."):
        try:
            if http:
                client = get_client()
                messages = client.get_session_messages(session_id)
            else:
                import sys
                import os
                
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                sys.path.insert(0, project_root)
                sys.path.insert(0, os.path.join(project_root, "backend"))
                
                from app.db.database import SessionLocal
                from app.db.models import ChatMessage
                
                db = SessionLocal()
                try:
                    db_messages = db.query(ChatMessage)\
                        .filter(ChatMessage.session_id == session_id)\
                        .order_by(ChatMessage.timestamp.desc())\
                        .limit(limit)\
                        .all()
                    db_messages.reverse()  # Oldest first
                    messages = [
                        {"role": m.role, "content": m.content, "timestamp": str(m.timestamp)}
                        for m in db_messages
                    ]
                finally:
                    db.close()
        except Exception as e:
            print_error(str(e))
            raise typer.Exit(1)
    
    if messages:
        console.print(f"\n[bold]📜 Session History[/] (last {len(messages)} messages)\n")
        for msg in messages:
            print_message(msg["role"], msg["content"], markdown=False)
    else:
        print_info("No messages in this session")
