"""
Synthesizer - Response formatting and synthesis.

This agent converts reasoning results into user-friendly responses.
It does NOT make decisions - only formats output.

DESIGN PRINCIPLES:
1. No decision making
2. Format-only operations
3. May use local model for simple formatting (optional)
4. Combines results with sources
"""
import logging
from typing import List, Optional

from app.analyzer.contracts import ReasoningResult, WebSearchResult

logger = logging.getLogger(__name__)


class Synthesizer:
    """
    Converts reasoning results into user-friendly response.
    
    Does NOT make decisions. Only formats and combines outputs.
    Can optionally use local LLM for formatting polish.
    """
    
    def __init__(self, use_local_llm: bool = False):
        """
        Initialize the Synthesizer.
        
        Args:
            use_local_llm: Whether to use local LLM for formatting polish
                          (disabled by default to save GPU)
        """
        self.use_local_llm = use_local_llm
    
    async def synthesize(
        self,
        reasoning_result: Optional[ReasoningResult],
        original_query: str,
        web_result: Optional[WebSearchResult] = None,
        include_sources: bool = True,
        include_limitations: bool = True
    ) -> str:
        """
        Format the final response.
        
        Args:
            reasoning_result: Result from reasoning agent (may be None)
            original_query: The original user query
            web_result: Result from web search (may be None)
            include_sources: Whether to include source URLs
            include_limitations: Whether to include limitations/assumptions
            
        Returns:
            Formatted user-friendly response string
        """
        parts = []
        
        # Handle case where no reasoning was needed
        if reasoning_result is None:
            return await self._synthesize_simple(original_query, web_result)
        
        # Main conclusion
        if reasoning_result.conclusion:
            parts.append(reasoning_result.conclusion)
        
        # Add reasoning chain if substantial
        if reasoning_result.reasoning_chain and len(reasoning_result.reasoning_chain) > 1:
            reasoning_text = self._format_reasoning_chain(reasoning_result.reasoning_chain)
            parts.append(f"\n\n**Reasoning:**\n{reasoning_text}")
        
        # Add limitations if requested and present
        if include_limitations and reasoning_result.limitations:
            limitations_text = self._format_limitations(reasoning_result.limitations)
            parts.append(f"\n\n**Limitations:**\n{limitations_text}")
        
        # Add sources if requested and present
        if include_sources and web_result and web_result.sources:
            sources_text = self._format_sources(web_result.sources)
            parts.append(f"\n\n**Sources:**\n{sources_text}")
        
        response = "\n".join(parts)
        
        logger.info(f"Synthesized response: {len(response)} chars")
        
        return response
    
    async def _synthesize_simple(
        self, 
        query: str, 
        web_result: Optional[WebSearchResult]
    ) -> str:
        """
        Synthesize a simple response when no deep reasoning was needed.
        
        This handles cases where we only did web search or neither.
        """
        if web_result and web_result.facts:
            # Combine web facts into a response
            facts_text = "\n".join(f"• {fact}" for fact in web_result.facts[:5])
            
            response = f"Based on current information:\n\n{facts_text}"
            
            if web_result.sources:
                sources_text = self._format_sources(web_result.sources[:3])
                response += f"\n\n**Sources:**\n{sources_text}"
            
            return response
        
        # Fallback - couldn't find relevant information
        return f"I couldn't find sufficient information to answer: \"{query}\"\n\nPlease try rephrasing your question or being more specific."
    
    def _format_reasoning_chain(self, chain: List[str]) -> str:
        """Format reasoning steps as numbered list."""
        return "\n".join(f"{i+1}. {step}" for i, step in enumerate(chain))
    
    def _format_limitations(self, limitations: List[str]) -> str:
        """Format limitations as bullet points."""
        return "\n".join(f"• {lim}" for lim in limitations)
    
    def _format_sources(self, sources: List[str]) -> str:
        """Format sources as numbered links."""
        formatted = []
        for i, source in enumerate(sources[:5], 1):  # Limit to 5 sources
            # Try to extract domain for cleaner display
            domain = self._extract_domain(source)
            formatted.append(f"{i}. [{domain}]({source})")
        return "\n".join(formatted)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for display."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return url[:30] + "..." if len(url) > 30 else url
    
    async def synthesize_error(self, error_message: str, query: str) -> str:
        """
        Format an error response.
        
        Args:
            error_message: The error that occurred
            query: The original query
            
        Returns:
            User-friendly error message
        """
        return (
            f"I encountered an issue while processing your request.\n\n"
            f"**Error:** {error_message}\n\n"
            f"Please try again or rephrase your question."
        )


# Module-level instance
synthesizer = Synthesizer()
