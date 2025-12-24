"""
SearXNG Web Agent - Web search via SearXNG.

This agent executes web searches but does NOT make decisions.
It only executes commands:
- search(query)
- extract(snippets)
- deduplicate()

DESIGN PRINCIPLES:
1. No decision making
2. Structured output only
3. Confidence scoring based on result quality
"""
import httpx
import logging
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode

from app.analyzer.contracts import WebSearchResult
from app.core.config import settings

logger = logging.getLogger(__name__)


class SearXNGAgent:
    """
    Executes web search via SearXNG.
    
    Does NOT make decisions. Only executes commands.
    Returns structured facts with sources and confidence.
    """
    
    def __init__(self, searxng_url: Optional[str] = None):
        """
        Initialize the SearXNG Agent.
        
        Args:
            searxng_url: Base URL of SearXNG instance (default from config)
        """
        self.base_url = searxng_url or getattr(settings, "SEARXNG_URL", "http://localhost:8080")
        self.timeout = 30.0
    
    async def search(self, query: str, num_results: int = 5, categories: str = "general") -> List[Dict[str, Any]]:
        """
        Execute search and return raw results.
        
        Args:
            query: Search query
            num_results: Maximum number of results
            categories: Search categories (general, news, images, etc.)
            
        Returns:
            List of raw search result dictionaries
        """
        params = {
            "q": query,
            "format": "json",
            "categories": categories,
        }
        
        url = f"{self.base_url}/search?{urlencode(params)}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])[:num_results]
                logger.info(f"SearXNG returned {len(results)} results for: {query[:50]}...")
                
                return results
                
        except httpx.HTTPError as e:
            logger.error(f"SearXNG HTTP error: {e}")
            return []
        except Exception as e:
            logger.error(f"SearXNG search failed: {e}")
            return []
    
    async def extract_snippets(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract relevant text snippets from search results.
        
        Args:
            results: Raw search results from SearXNG
            
        Returns:
            List of text snippets
        """
        snippets = []
        
        for result in results:
            # Try different fields that might contain text
            content = result.get("content", "")
            title = result.get("title", "")
            
            if content:
                # Clean HTML tags if present
                clean_content = self._clean_html(content)
                if clean_content:
                    snippets.append(clean_content)
            
            if title and title not in snippets:
                snippets.append(title)
        
        return snippets
    
    async def deduplicate(self, facts: List[str]) -> List[str]:
        """
        Remove duplicate or very similar facts.
        
        Args:
            facts: List of extracted facts
            
        Returns:
            Deduplicated list of facts
        """
        if not facts:
            return []
        
        unique_facts = []
        seen_normalized = set()
        
        for fact in facts:
            # Normalize for comparison
            normalized = self._normalize_text(fact)
            
            # Skip if too similar to existing
            if normalized in seen_normalized:
                continue
            
            # Check for substring matches
            is_duplicate = False
            for seen in seen_normalized:
                if normalized in seen or seen in normalized:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_facts.append(fact)
                seen_normalized.add(normalized)
        
        return unique_facts
    
    async def execute(self, query: str, num_results: int = 5) -> WebSearchResult:
        """
        Full pipeline: search -> extract -> deduplicate.
        
        Args:
            query: The search query
            num_results: Maximum number of results
            
        Returns:
            WebSearchResult with facts, sources, and confidence
        """
        # 1. Search
        raw_results = await self.search(query, num_results)
        
        if not raw_results:
            logger.warning(f"No search results for: {query[:50]}...")
            return WebSearchResult(
                facts=[],
                sources=[],
                raw_snippets=[],
                confidence=0.0,
                query_used=query
            )
        
        # 2. Extract snippets
        snippets = await self.extract_snippets(raw_results)
        
        # 3. Extract sources
        sources = [r.get("url", "") for r in raw_results if r.get("url")]
        
        # 4. Deduplicate
        facts = await self.deduplicate(snippets)
        
        # 5. Calculate confidence
        confidence = self._calculate_confidence(raw_results, facts)
        
        logger.info(f"Web search complete: {len(facts)} facts from {len(sources)} sources")
        
        return WebSearchResult(
            facts=facts,
            sources=sources,
            raw_snippets=snippets,
            confidence=confidence,
            query_used=query
        )
    
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        clean = re.sub(r'<[^>]+>', '', text)
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase, remove punctuation, collapse whitespace
        normalized = text.lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()
    
    def _calculate_confidence(self, raw_results: List[Dict], facts: List[str]) -> float:
        """
        Calculate confidence score based on result quality.
        
        Factors:
        - Number of results
        - Number of unique sources
        - Presence of authoritative domains
        """
        if not raw_results:
            return 0.0
        
        score = 0.5  # Base score
        
        # More results = higher confidence
        if len(raw_results) >= 5:
            score += 0.1
        elif len(raw_results) >= 3:
            score += 0.05
        
        # More facts = higher confidence
        if len(facts) >= 5:
            score += 0.15
        elif len(facts) >= 3:
            score += 0.1
        
        # Check for authoritative sources
        authoritative_domains = [
            "wikipedia.org", "gov.", ".edu", "reuters.com",
            "bbc.com", "nytimes.com", "nature.com", "sciencedirect.com"
        ]
        
        sources = [r.get("url", "") for r in raw_results]
        for source in sources:
            if any(domain in source for domain in authoritative_domains):
                score += 0.05
                break
        
        return min(score, 1.0)


# Module-level instance
searxng_agent = SearXNGAgent()
