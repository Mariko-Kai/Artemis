"""
Artemis CLI - Main entry point.

Usage:
    artemis chat              - Interactive chat with LLM
    artemis agent run "task"  - Execute agent task
    artemis memory query "q"  - Search memories
    artemis session list      - List chat sessions
"""
import typer
from typing import Optional

from cli.commands import chat, agent, memory, session
from cli.utils.output import console, print_info
from cli.config import get_settings

app = typer.Typer(
    name="artemis",
    help="🏛️ Artemis - Local AI Agent CLI",
    add_completion=True,
    no_args_is_help=True
)

# Register command groups
app.add_typer(chat.app, name="chat", help="💬 Chat with LLM")
app.add_typer(agent.app, name="agent", help="🤖 Agent tasks")
app.add_typer(memory.app, name="memory", help="🧠 Memory operations")
app.add_typer(session.app, name="session", help="📋 Session management")


@app.command()
def version():
    """Show version information."""
    from cli import __version__
    console.print(f"[bold blue]Artemis CLI[/] v{__version__}")
    console.print(f"[dim]API URL: {get_settings().api_base_url}[/]")


@app.command()
def health(
    http: bool = typer.Option(True, help="Check HTTP server health")
):
    """Check if the Artemis server is running."""
    if http:
        from cli.utils.http_client import get_client
        client = get_client()
        if client.health():
            console.print("[bold green]✓[/] Server is healthy")
        else:
            console.print("[bold red]✗[/] Server is not responding")
            raise typer.Exit(1)
    else:
        print_info("Direct mode - no health check needed")


def main():
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
