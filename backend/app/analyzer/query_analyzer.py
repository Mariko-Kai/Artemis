"""
Query Analyzer - Rules-first query classification.

This module analyzes user queries to determine routing:
- Whether web search is needed
- Whether heavy reasoning is required
- Complexity assessment

DESIGN PRINCIPLES:
1. Rules-first approach (deterministic, testable)
2. Cheap local LLM only as fallback
3. NO chain-of-thought
4. All decisions are logged
"""
import re
import logging
from typing import Optional, Set
from .contracts import AnalysisResult

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Deterministic query classifier.
    
    Uses rules first, cheap local LLM only as fallback.
    NO chain-of-thought. NO reasoning on local model.
    """
    
    # Keywords indicating web search is needed (current/real-time data)
    WEB_KEYWORDS: Set[str] = {
        # Temporal indicators
        "current", "latest", "today", "now", "recent", "2024", "2025",
        "this week", "this month", "this year", "yesterday", "tomorrow",
        # Real-time data
        "price", "stock", "weather", "news", "score", "results",
        "happening", "trending", "live", "update", "market", "economy",
        # Verification needs
        "is it true", "fact check", "verify", "confirm",
        # Specific entities that change
        "population", "statistics", "rate", "index", "election",
    }
    
    # Keywords indicating complex reasoning is needed
    REASONING_KEYWORDS: Set[str] = {
        # Analysis tasks
        "analyze", "analyse", "analysis", "evaluate", "assess",
        "compare", "contrast", "differentiate", "impact",
        # Explanation tasks
        "explain why", "explain how", "reason", "reasoning",
        "cause", "effect", "consequence", "implication",
        # Synthesis tasks
        "synthesize", "summarize complex", "integrate",
        # Problem-solving
        "solve", "calculate", "derive", "prove", "demonstrate",
        "optimize", "recommend", "suggest strategy",
        # Multi-factor
        "considering", "taking into account", "weighing",
        "pros and cons", "trade-off", "trade off",
    }
    
    # Keywords indicating simple queries (no external help needed)
    SIMPLE_KEYWORDS: Set[str] = {
        "what is", "define", "meaning of", "definition",
        "who is", "when was", "where is",
        "hello", "hi", "thanks", "thank you",
    }
    
    # Question patterns that suggest complexity
    COMPLEX_PATTERNS = [
        r"how (can|could|would|should|might) .+ (affect|impact|change|influence)",
        r"what (are|is) the (implications?|consequences?|effects?) of",
        r"why (does|do|did|is|are|was|were) .+ (important|significant|matter)",
        r"compare .+ (with|to|and) .+",
        r"(analyze|evaluate|assess) .+ (in terms of|based on|considering)",
    ]
    
    def __init__(self, enable_llm_fallback: bool = True):
        """
        Initialize the Query Analyzer.
        
        Args:
            enable_llm_fallback: Whether to use LLM for ambiguous queries
        """
        self.enable_llm_fallback = enable_llm_fallback
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.COMPLEX_PATTERNS]
    
    def analyze(self, query: str) -> AnalysisResult:
        """
        Analyze a query to determine routing.
        
        Args:
            query: The user's query string
            
        Returns:
            AnalysisResult with routing decisions
        """
        query_lower = query.lower().strip()
        
        # 1. Rule-based classification (check keywords first)
        needs_web = self._check_web_keywords(query_lower)
        needs_reasoning = self._check_reasoning_keywords(query_lower)
        
        # 2. Check for simple queries (if not already marked for web/reasoning)
        if not needs_web and not needs_reasoning and self._is_simple_query(query_lower):
            logger.debug(f"Query classified as SIMPLE: {query[:50]}...")
            return AnalysisResult(
                needs_web=False,
                needs_deep_reasoning=False,
                complexity_score=0.1,
                confidence=0.95,
                reason="Simple factual or greeting query"
            )
        
        # 3. Estimate complexity
        complexity = self._estimate_complexity(query_lower)
        
        # 4. Pattern-based complexity check
        pattern_match = self._check_complex_patterns(query_lower)
        if pattern_match:
            needs_reasoning = True
            complexity = max(complexity, 0.7)
        
        # 5. If rules matched with high confidence, return
        if needs_web or needs_reasoning:
            reason = self._generate_reason(needs_web, needs_reasoning, query_lower)
            confidence = 0.85 if (needs_web and needs_reasoning) else 0.9
            
            logger.info(f"Query classified by RULES: web={needs_web}, reasoning={needs_reasoning}")
            return AnalysisResult(
                needs_web=needs_web,
                needs_deep_reasoning=needs_reasoning,
                complexity_score=complexity,
                confidence=confidence,
                reason=reason
            )
        
        # 5. Ambiguous query - use heuristics or fallback
        # Length-based heuristic: longer queries tend to be more complex
        if len(query.split()) > 15:
            complexity = max(complexity, 0.5)
            needs_reasoning = True
            
        # Question word analysis
        if query_lower.startswith(("how", "why")):
            complexity = max(complexity, 0.4)
        
        # 6. Final decision for ambiguous queries
        if complexity > 0.5 and not needs_web and not needs_reasoning:
            # Ambiguous but complex - default to reasoning
            needs_reasoning = True
            
        logger.info(f"Query classified by HEURISTICS: web={needs_web}, reasoning={needs_reasoning}")
        return AnalysisResult(
            needs_web=needs_web,
            needs_deep_reasoning=needs_reasoning,
            complexity_score=complexity,
            confidence=0.7,  # Lower confidence for heuristic-based decisions
            reason=self._generate_reason(needs_web, needs_reasoning, query_lower)
        )
    
    def _is_simple_query(self, query: str) -> bool:
        """Check if query is simple (greeting, basic definition, etc.)."""
        for keyword in self.SIMPLE_KEYWORDS:
            if query.startswith(keyword) or keyword in query:
                # But not if it also contains complex indicators
                if not any(k in query for k in self.REASONING_KEYWORDS):
                    return True
        return False
    
    def _check_web_keywords(self, query: str) -> bool:
        """Check if query contains web search indicators."""
        for keyword in self.WEB_KEYWORDS:
            if keyword in query:
                return True
        return False
    
    def _check_reasoning_keywords(self, query: str) -> bool:
        """Check if query contains complex reasoning indicators."""
        for keyword in self.REASONING_KEYWORDS:
            if keyword in query:
                return True
        return False
    
    def _check_complex_patterns(self, query: str) -> bool:
        """Check if query matches complex question patterns."""
        for pattern in self._compiled_patterns:
            if pattern.search(query):
                return True
        return False
    
    def _estimate_complexity(self, query: str) -> float:
        """
        Estimate query complexity score.
        
        Returns:
            Float between 0.0 (trivial) and 1.0 (very complex)
        """
        score = 0.2  # Base score
        
        # Word count contributes to complexity
        word_count = len(query.split())
        if word_count > 20:
            score += 0.3
        elif word_count > 10:
            score += 0.2
        elif word_count > 5:
            score += 0.1
        
        # Multiple questions increase complexity
        question_marks = query.count("?")
        if question_marks > 1:
            score += 0.2
        
        # Conjunctions suggest multi-part queries
        conjunctions = ["and", "but", "however", "also", "moreover", "furthermore"]
        for conj in conjunctions:
            if f" {conj} " in query:
                score += 0.1
                break
        
        # Technical terms (simple heuristic: words > 10 chars)
        long_words = [w for w in query.split() if len(w) > 10]
        if len(long_words) > 2:
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_reason(self, needs_web: bool, needs_reasoning: bool, query: str) -> str:
        """Generate human-readable reason for the routing decision."""
        reasons = []
        
        if needs_web:
            # Find which keyword triggered web search
            for keyword in self.WEB_KEYWORDS:
                if keyword in query:
                    reasons.append(f"Requires current/real-time data (detected: '{keyword}')")
                    break
            else:
                reasons.append("May require current data")
        
        if needs_reasoning:
            # Find which keyword triggered reasoning
            for keyword in self.REASONING_KEYWORDS:
                if keyword in query:
                    reasons.append(f"Requires complex analysis (detected: '{keyword}')")
                    break
            else:
                reasons.append("Query complexity suggests deep reasoning needed")
        
        if not reasons:
            reasons.append("Standard query - can be handled locally")
        
        return "; ".join(reasons)


# Module-level instance for convenience
query_analyzer = QueryAnalyzer()
