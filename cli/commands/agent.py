"""
Agent command - execute ReAct agent tasks.
"""
import typer
from typing import Optional
from cli.utils.output import (
    console, print_message, print_thinking, print_error,
    print_success, print_info, print_table
)
from cli.utils.http_client import get_client

app = typer.Typer(help="🤖 Execute agent tasks")


def _direct_agent_run(goal: str) -> str:
    """Run agent using direct Python service calls."""
    import sys
    import os
    
    # Add project paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "backend"))
    
    from app.agent.executor import agent_executor
    return agent_executor.run(goal)


@app.command("run")
def run_agent(
    goal: str = typer.Argument(..., help="Task/goal for the agent to complete"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID"),
    http: bool = typer.Option(False, "--http", help="Use HTTP mode")
):
    """
    Run an agent task.
    
    Example: artemis agent run "What is the weather in Tokyo?"
    """
    print_info(f"🎯 Goal: {goal}")
    
    with print_thinking("Agent thinking..."):
        try:
            if http:
                client = get_client()
                response = client.run_agent(goal, session_id=session)
                result = response.get("answer", str(response))
            else:
                result = _direct_agent_run(goal)
        except Exception as e:
            print_error(str(e))
            raise typer.Exit(1)
    
    print_message("assistant", result)
    print_success("Agent task completed")


@app.command("tools")
def list_tools():
    """Show available agent tools."""
    import sys
    import os
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "backend"))
    
    try:
        from app.agent.tools import ALL_TOOLS
        
        tools_data = []
        for tool in ALL_TOOLS:
            tools_data.append({
                "name": tool.name,
                "description": tool.description[:60] + "..." if len(tool.description) > 60 else tool.description
            })
        
        print_table(tools_data, title="🔧 Available Agent Tools", columns=["name", "description"])
    except Exception as e:
        print_error(f"Could not load tools: {e}")
        raise typer.Exit(1)
