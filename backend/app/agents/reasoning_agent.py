"""
Groq Reasoning Agent - Heavy reasoning via Groq API.

This agent uses external API for complex reasoning tasks.
It does NOT use local GPU resources.

DESIGN PRINCIPLES:
1. Uses only Groq API (openai/gpt-oss-120b)
2. Structured input/output
3. No web browsing
4. No chit-chat
5. No repeating input data

PROHIBITED:
- Web browsing
- Chit-chat
- Repeating input data verbatim
- Reasoning on local model
"""
import os
import logging
from typing import List, Optional
from openai import OpenAI, AsyncOpenAI

from app.analyzer.contracts import ReasoningResult
from app.core.config import settings

logger = logging.getLogger(__name__)


# Reasoning system prompt
REASONING_SYSTEM_PROMPT = """You are a reasoning specialist. Your task is to analyze information and provide structured, logical conclusions.

RULES:
1. Focus ONLY on the given task
2. Use provided context and facts
3. Do NOT browse the web
4. Do NOT engage in chit-chat
5. Do NOT repeat input data verbatim
6. Provide step-by-step reasoning
7. Acknowledge limitations and assumptions
8. Be precise and verifiable

OUTPUT FORMAT:
1. First, provide your conclusion (1-2 sentences)
2. Then, list your reasoning steps (numbered)
3. Finally, list any limitations or assumptions

Do NOT use markdown formatting in your response."""


class GroqReasoningAgent:
    """
    Heavy reasoning via Groq API.
    
    Uses openai/gpt-oss-120b model for complex analysis.
    Does NOT use local GPU resources.
    """
    
    def __init__(self):
        """Initialize the Groq Reasoning Agent."""
        api_key = os.environ.get("GROQ_API_KEY") or getattr(settings, "GROQ_API_KEY", "")
        
        if not api_key:
            logger.warning("GROQ_API_KEY not set - reasoning agent will fail")
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        ) if api_key else None
        
        self.model = getattr(settings, "GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        self.max_tokens = 2048
        self.temperature = 0.3  # Lower temperature for more deterministic reasoning
    
    async def reason(
        self, 
        task: str, 
        context: str, 
        web_facts: Optional[List[str]] = None
    ) -> ReasoningResult:
        """
        Solve a reasoning task with provided context.
        
        Args:
            task: The reasoning task (usually the original query)
            context: Additional context
            web_facts: Optional list of facts from web search
            
        Returns:
            ReasoningResult with conclusion and reasoning chain
        """
        if not self.client:
            logger.error("Groq client not initialized - missing API key")
            return ReasoningResult(
                conclusion="Error: Reasoning agent not configured (missing GROQ_API_KEY)",
                reasoning_chain=[],
                confidence=0.0,
                limitations=["API key not configured"]
            )
        
        # Build the user message
        user_message = self._build_user_message(task, context, web_facts)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            raw_response = response.choices[0].message.content
            logger.info(f"Groq reasoning complete: {len(raw_response)} chars")
            
            # Parse the response
            return self._parse_response(raw_response)
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return ReasoningResult(
                conclusion=f"Error during reasoning: {str(e)}",
                reasoning_chain=[],
                confidence=0.0,
                limitations=[f"API error: {str(e)}"]
            )
    
    def _build_user_message(
        self, 
        task: str, 
        context: str, 
        web_facts: Optional[List[str]]
    ) -> str:
        """Build the user message for the reasoning task."""
        parts = []
        
        parts.append(f"TASK: {task}")
        
        if context:
            parts.append(f"\nCONTEXT:\n{context}")
        
        if web_facts:
            facts_text = "\n".join(f"- {fact}" for fact in web_facts[:10])  # Limit facts
            parts.append(f"\nRELEVANT FACTS:\n{facts_text}")
        
        parts.append("\nProvide your analysis following the output format.")
        
        return "\n".join(parts)
    
    def _parse_response(self, raw_response: str) -> ReasoningResult:
        """
        Parse the raw LLM response into structured ReasoningResult.
        
        Expected format:
        - Conclusion first
        - Numbered reasoning steps
        - Limitations at the end
        """
        lines = raw_response.strip().split("\n")
        
        conclusion = ""
        reasoning_chain = []
        limitations = []
        
        current_section = "conclusion"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section changes
            lower_line = line.lower()
            if "reasoning" in lower_line and "step" in lower_line:
                current_section = "reasoning"
                continue
            elif "limitation" in lower_line or "assumption" in lower_line:
                current_section = "limitations"
                continue
            
            # Check for numbered steps
            if line[0].isdigit() and "." in line[:4]:
                if current_section == "conclusion":
                    current_section = "reasoning"
                
                if current_section == "reasoning":
                    # Remove the number prefix
                    step_text = line.split(".", 1)[-1].strip()
                    reasoning_chain.append(step_text)
                elif current_section == "limitations":
                    lim_text = line.split(".", 1)[-1].strip()
                    limitations.append(lim_text)
            elif line.startswith("-"):
                # Bullet points for limitations
                if current_section == "limitations":
                    limitations.append(line[1:].strip())
                elif current_section == "reasoning":
                    reasoning_chain.append(line[1:].strip())
            else:
                # Regular text
                if current_section == "conclusion":
                    conclusion += " " + line if conclusion else line
                elif current_section == "reasoning" and not reasoning_chain:
                    reasoning_chain.append(line)
        
        # Calculate confidence based on response quality
        confidence = self._calculate_confidence(conclusion, reasoning_chain)
        
        return ReasoningResult(
            conclusion=conclusion.strip() or "Unable to formulate conclusion",
            reasoning_chain=reasoning_chain,
            confidence=confidence,
            limitations=limitations
        )
    
    def _calculate_confidence(self, conclusion: str, reasoning_chain: List[str]) -> float:
        """Calculate confidence based on response quality."""
        score = 0.5  # Base score
        
        # Has conclusion
        if conclusion and len(conclusion) > 20:
            score += 0.15
        
        # Has reasoning steps
        if len(reasoning_chain) >= 3:
            score += 0.2
        elif len(reasoning_chain) >= 1:
            score += 0.1
        
        # Longer reasoning is often more thorough
        total_reasoning_length = sum(len(s) for s in reasoning_chain)
        if total_reasoning_length > 500:
            score += 0.1
        
        return min(score, 1.0)


# Module-level instance
groq_reasoning_agent = GroqReasoningAgent()
