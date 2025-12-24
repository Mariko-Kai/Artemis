"""
Enhanced Orchestrator - Multi-agent coordination with explicit planning.

This orchestrator:
1. Analyzes queries using QueryAnalyzer
2. Creates execution plans using ExecutionPlanner
3. Executes steps sequentially (WebAgent, ReasoningAgent, Synthesizer)
4. Logs all decisions using DecisionLogger

DESIGN PRINCIPLES:
1. Explicit planning (no hidden heuristics)
2. All decisions logged
3. Extensible (new agents without rewriting)
4. No GPU usage in orchestration logic
"""
import logging
import time
from typing import Optional, Dict, Any

from app.analyzer.query_analyzer import QueryAnalyzer
from app.analyzer.contracts import (
    AnalysisResult, ExecutionPlan, StepType,
    WebSearchResult, ReasoningResult, OrchestratorResponse
)
from app.orchestrator.execution_planner import ExecutionPlanner
from app.agents.web_agent import SearXNGAgent
from app.agents.reasoning_agent import GroqReasoningAgent
from app.agents.synthesizer import Synthesizer
from app.logs.decision_logger import DecisionLogger

logger = logging.getLogger(__name__)


class EnhancedOrchestrator:
    """
    Extended orchestrator with explicit planning and logging.
    
    Flow:
    1. Query Analyzer → AnalysisResult
    2. Execution Planner → ExecutionPlan
    3. Execute each step sequentially
    4. Log all decisions
    """
    
    def __init__(self):
        """Initialize all sub-components."""
        self.analyzer = QueryAnalyzer()
        self.planner = ExecutionPlanner()
        self.web_agent = SearXNGAgent()
        self.reasoning_agent = GroqReasoningAgent()
        self.synthesizer = Synthesizer()
        self.logger = DecisionLogger()
        
        logger.info("EnhancedOrchestrator initialized")
    
    async def run(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        force_web: bool = False,
        force_reasoning: bool = False
    ) -> OrchestratorResponse:
        """
        Execute the full orchestration pipeline.
        
        Args:
            query: User query to process
            session_id: Optional session ID for context
            force_web: Force web search regardless of analysis
            force_reasoning: Force reasoning regardless of analysis
            
        Returns:
            OrchestratorResponse with answer and trace
        """
        start_time = time.time()
        trace_id = self.logger.generate_trace_id()
        
        logger.info(f"[{trace_id}] Starting orchestration for: {query[:50]}...")
        
        try:
            # Step 1: Analyze query
            analysis = self._analyze_query(query, trace_id, force_web, force_reasoning)
            
            # Step 2: Create execution plan
            plan = self._create_plan(analysis, query, trace_id)
            
            # Step 3: Execute plan
            result = await self._execute_plan(plan, query, trace_id)
            
            # Log completion
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_completion(trace_id, success=True, total_duration_ms=duration_ms)
            
            return OrchestratorResponse(
                answer=result["response"],
                trace_id=trace_id,
                plan_executed=[s.value for s in plan.steps],
                analysis=analysis,
                sources=result.get("sources", [])
            )
            
        except Exception as e:
            logger.error(f"[{trace_id}] Orchestration failed: {e}")
            self.logger.log_error(e, trace_id)
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.log_completion(trace_id, success=False, total_duration_ms=duration_ms)
            
            # Return error response
            error_response = await self.synthesizer.synthesize_error(str(e), query)
            return OrchestratorResponse(
                answer=error_response,
                trace_id=trace_id,
                plan_executed=[],
                analysis=AnalysisResult(
                    needs_web=False,
                    needs_deep_reasoning=False,
                    complexity_score=0.0,
                    confidence=0.0,
                    reason=f"Error: {str(e)}"
                ),
                sources=[]
            )
    
    def _analyze_query(
        self, 
        query: str, 
        trace_id: str,
        force_web: bool,
        force_reasoning: bool
    ) -> AnalysisResult:
        """Analyze the query and log the decision."""
        analysis = self.analyzer.analyze(query)
        
        # Apply force flags
        if force_web:
            analysis = AnalysisResult(
                needs_web=True,
                needs_deep_reasoning=analysis.needs_deep_reasoning or force_reasoning,
                complexity_score=analysis.complexity_score,
                confidence=analysis.confidence,
                reason=f"[FORCED WEB] {analysis.reason}"
            )
        
        if force_reasoning:
            analysis = AnalysisResult(
                needs_web=analysis.needs_web,
                needs_deep_reasoning=True,
                complexity_score=max(analysis.complexity_score, 0.7),
                confidence=analysis.confidence,
                reason=f"[FORCED REASONING] {analysis.reason}"
            )
        
        self.logger.log_analysis(query, analysis, trace_id)
        
        return analysis
    
    def _create_plan(
        self, 
        analysis: AnalysisResult, 
        query: str, 
        trace_id: str
    ) -> ExecutionPlan:
        """Create execution plan and log it."""
        plan = self.planner.create_plan(analysis, query, trace_id)
        self.logger.log_plan(plan, trace_id)
        return plan
    
    async def _execute_plan(
        self, 
        plan: ExecutionPlan, 
        query: str, 
        trace_id: str
    ) -> Dict[str, Any]:
        """
        Execute the plan steps sequentially.
        
        Returns:
            Dict with 'response' and optional 'sources'
        """
        context: Dict[str, Any] = {
            "query": query,
            "web_result": None,
            "reasoning_result": None,
            "sources": []
        }
        
        for step in plan.steps:
            step_start = time.time()
            
            if step == StepType.SEARCH_WEB:
                context = await self._execute_web_search(context, trace_id, step_start)
                
            elif step == StepType.REASONING:
                context = await self._execute_reasoning(context, trace_id, step_start)
                
            elif step == StepType.SYNTHESIZE:
                response = await self._execute_synthesis(context, trace_id, step_start)
                return {
                    "response": response,
                    "sources": context.get("sources", [])
                }
        
        # Should not reach here, but fallback
        return {"response": "No synthesis step executed", "sources": []}
    
    async def _execute_web_search(
        self, 
        context: Dict[str, Any], 
        trace_id: str,
        step_start: float
    ) -> Dict[str, Any]:
        """Execute web search step."""
        query = context["query"]
        
        logger.info(f"[{trace_id}] Executing web search...")
        
        web_result = await self.web_agent.execute(query)
        
        duration_ms = (time.time() - step_start) * 1000
        self.logger.log_step_execution(
            StepType.SEARCH_WEB.value,
            {"query": query},
            web_result,
            trace_id,
            duration_ms
        )
        
        context["web_result"] = web_result
        context["sources"] = web_result.sources
        
        return context
    
    async def _execute_reasoning(
        self, 
        context: Dict[str, Any], 
        trace_id: str,
        step_start: float
    ) -> Dict[str, Any]:
        """Execute reasoning step."""
        query = context["query"]
        web_result: Optional[WebSearchResult] = context.get("web_result")
        
        logger.info(f"[{trace_id}] Executing reasoning...")
        
        # Build context for reasoning
        reasoning_context = ""
        web_facts = []
        
        if web_result:
            web_facts = web_result.facts
            reasoning_context = f"Web search found {len(web_facts)} relevant facts."
        
        reasoning_result = await self.reasoning_agent.reason(
            task=query,
            context=reasoning_context,
            web_facts=web_facts
        )
        
        duration_ms = (time.time() - step_start) * 1000
        self.logger.log_step_execution(
            StepType.REASONING.value,
            {"query": query, "facts_count": len(web_facts)},
            reasoning_result,
            trace_id,
            duration_ms
        )
        
        context["reasoning_result"] = reasoning_result
        
        return context
    
    async def _execute_synthesis(
        self, 
        context: Dict[str, Any], 
        trace_id: str,
        step_start: float
    ) -> str:
        """Execute synthesis step."""
        query = context["query"]
        web_result: Optional[WebSearchResult] = context.get("web_result")
        reasoning_result: Optional[ReasoningResult] = context.get("reasoning_result")
        
        logger.info(f"[{trace_id}] Executing synthesis...")
        
        response = await self.synthesizer.synthesize(
            reasoning_result=reasoning_result,
            original_query=query,
            web_result=web_result,
            include_sources=True,
            include_limitations=True
        )
        
        duration_ms = (time.time() - step_start) * 1000
        self.logger.log_step_execution(
            StepType.SYNTHESIZE.value,
            {"has_reasoning": reasoning_result is not None, "has_web": web_result is not None},
            {"response_length": len(response)},
            trace_id,
            duration_ms
        )
        
        return response
    
    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """
        Get the full decision trace for a request.
        
        Args:
            trace_id: The trace ID to retrieve
            
        Returns:
            Full trace data
        """
        return {
            "trace_id": trace_id,
            "entries": self.logger.get_decision_trace(trace_id)
        }


# Module-level instance
enhanced_orchestrator = EnhancedOrchestrator()
