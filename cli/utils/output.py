"""
Rich output utilities for CLI formatting.
"""
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
error_console = Console(stderr=True)


def print_message(role: str, content: str, markdown: bool = True) -> None:
    """Print a chat message with formatting."""
    if role == "user":
        emoji = "👤"
        style = "bold blue"
        title = "You"
    elif role == "assistant":
        emoji = "🤖"
        style = "bold green"
        title = "Artemis"
    else:
        emoji = "ℹ️"
        style = "dim"
        title = role.capitalize()
    
    if markdown and role == "assistant":
        panel_content = Markdown(content)
    else:
        panel_content = content
    
    console.print(Panel(
        panel_content,
        title=f"{emoji} {title}",
        title_align="left",
        border_style=style,
        padding=(0, 1)
    ))


def print_thinking(message: str = "Thinking...") -> Progress:
    """Show a thinking spinner."""
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[bold cyan]{message}[/]"),
        console=console,
        transient=True
    )


def print_error(message: str) -> None:
    """Print an error message."""
    error_console.print(f"[bold red]❌ Error:[/] {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]✓[/] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]⚠[/] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue]ℹ[/] {message}")


def print_table(
    data: List[Dict[str, Any]], 
    title: Optional[str] = None,
    columns: Optional[List[str]] = None
) -> None:
    """Print data as a rich table."""
    if not data:
        print_warning("No data to display")
        return
    
    table = Table(title=title, show_header=True, header_style="bold magenta")
    
    # Auto-detect columns if not specified
    if columns is None:
        columns = list(data[0].keys())
    
    for col in columns:
        table.add_column(col.replace("_", " ").title())
    
    for row in data:
        table.add_row(*[str(row.get(col, "")) for col in columns])
    
    console.print(table)


def print_memory_results(results: List[Dict[str, Any]]) -> None:
    """Print memory search results."""
    if not results:
        print_warning("No memories found")
        return
    
    console.print(f"\n[bold]Found {len(results)} memories:[/]\n")
    
    for i, mem in enumerate(results, 1):
        score = mem.get("score", 0)
        score_color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
        
        console.print(Panel(
            mem.get("content", ""),
            title=f"[{score_color}]Score: {score:.2f}[/] | {mem.get('created_at', 'Unknown')}",
            subtitle=f"ID: {mem.get('id', 'N/A')[:8]}...",
            border_style="dim"
        ))


def print_agent_step(step: int, action: str, result: str) -> None:
    """Print an agent execution step."""
    console.print(f"\n[bold cyan]Step {step}:[/] {action}")
    if result:
        console.print(Panel(result, title="Observation", border_style="dim"))


def print_welcome() -> None:
    """Print CLI welcome banner."""
    console.print(Panel.fit(
        "[bold blue]🏛️ Artemis[/] - Local AI Agent\n"
        "[dim]Type 'exit' or Ctrl+C to quit[/]",
        border_style="blue"
    ))
