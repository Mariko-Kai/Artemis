"""
HTTP client for CLI in HTTP mode.
"""
import httpx
from typing import Any, Dict, List, Optional
from cli.config import get_settings


class ArtemisClient:
    """HTTP client for Artemis API."""
    
    def __init__(self, base_url: Optional[str] = None):
        settings = get_settings()
        self.base_url = base_url or settings.api_base_url
        self.api_url = f"{self.base_url}/{settings.api_version}"
        self.timeout = settings.timeout
    
    def _get_client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout)
    
    # Chat endpoints
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        session_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Send chat completion request."""
        with self._get_client() as client:
            response = client.post(
                f"{self.api_url}/chat/completions",
                json={
                    "messages": messages,
                    "session_id": session_id,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            return response.json()
    
    # Agent endpoints
    def run_agent(self, goal: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Run agent task."""
        with self._get_client() as client:
            response = client.post(
                f"{self.api_url}/agent/run",
                json={"goal": goal, "session_id": session_id}
            )
            response.raise_for_status()
            return response.json()
    
    # Memory endpoints
    def query_memory(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Query memories."""
        with self._get_client() as client:
            response = client.post(
                f"{self.api_url}/memory/query",
                json={"query": query, "session_id": session_id, "top_k": top_k}
            )
            response.raise_for_status()
            return response.json()
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """Get memory archive stats."""
        with self._get_client() as client:
            response = client.get(f"{self.api_url}/memory/archive/stats")
            response.raise_for_status()
            return response.json()
    
    def export_memories(self) -> str:
        """Export all memories as JSONL."""
        with self._get_client() as client:
            response = client.get(f"{self.api_url}/memory/archive/export")
            response.raise_for_status()
            return response.text
    
    # Session endpoints
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        with self._get_client() as client:
            response = client.get(f"{self.api_url}/sessions")
            response.raise_for_status()
            return response.json()
    
    def create_session(self, title: Optional[str] = None) -> Dict[str, Any]:
        """Create a new session."""
        with self._get_client() as client:
            response = client.post(
                f"{self.api_url}/sessions",
                json={"title": title} if title else {}
            )
            response.raise_for_status()
            return response.json()
    
    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session."""
        with self._get_client() as client:
            response = client.delete(f"{self.api_url}/sessions/{session_id}")
            response.raise_for_status()
            return response.json()
    
    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get messages for a session."""
        with self._get_client() as client:
            response = client.get(f"{self.api_url}/sessions/{session_id}/messages")
            response.raise_for_status()
            return response.json()
    
    # Health check
    def health(self) -> bool:
        """Check if server is healthy."""
        try:
            with self._get_client() as client:
                response = client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False


# Global client instance
_client: Optional[ArtemisClient] = None


def get_client(base_url: Optional[str] = None) -> ArtemisClient:
    """Get HTTP client singleton."""
    global _client
    if _client is None or base_url:
        _client = ArtemisClient(base_url)
    return _client
