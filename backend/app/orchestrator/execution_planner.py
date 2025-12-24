"""
Execution Planner - Converts analysis results into execution plans.

This module is purely deterministic:
- No LLM involved
- Direct mapping from AnalysisResult to ExecutionPlan
- Fully testable
"""
import logging
from typing import Optional

from app.analyzer.contracts import AnalysisResult, ExecutionPlan, StepType

logger = logging.getLogger(__name__)


class ExecutionPlanner:
    """
    Converts AnalysisResult into ExecutionPlan.
    
    Deterministic mapping, no LLM involved.
    Fully testable without any external dependencies.
    """
    
    def __init__(self):
        """Initialize the Execution Planner."""
        pass
    
    def create_plan(
        self, 
        analysis: AnalysisResult, 
        query: str,
        trace_id: Optional[str] = None
    ) -> ExecutionPlan:
        """
        Create an execution plan based on analysis results.
        
        Args:
            analysis: The query analysis result
            query: The original user query
            trace_id: Optional trace ID for logging
            
        Returns:
            ExecutionPlan with ordered steps
        """
        steps = []
        
        # Step 1: Web search if needed
        if analysis.needs_web:
            steps.append(StepType.SEARCH_WEB)
            logger.debug(f"Plan: Added SEARCH_WEB step (needs_web=True)")
        
        # Step 2: Deep reasoning if needed
        if analysis.needs_deep_reasoning:
            steps.append(StepType.REASONING)
            logger.debug(f"Plan: Added REASONING step (needs_deep_reasoning=True)")
        
        # Step 3: Always synthesize at the end
        # (combines results into user-friendly response)
        steps.append(StepType.SYNTHESIZE)
        logger.debug(f"Plan: Added SYNTHESIZE step (always)")
        
        # Build context
        context = {
            "query": query,
            "analysis": analysis.model_dump(),
        }
        
        plan = ExecutionPlan(
            steps=steps,
            context=context,
            trace_id=trace_id
        )
        
        logger.info(f"Created plan with {len(steps)} steps: {[s.value for s in steps]}")
        
        return plan
    
    def validate_plan(self, plan: ExecutionPlan) -> bool:
        """
        Validate that a plan is well-formed.
        
        Args:
            plan: The execution plan to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Must have at least one step
        if not plan.steps:
            logger.warning("Invalid plan: no steps")
            return False
        
        # Must end with synthesize
        if plan.steps[-1] != StepType.SYNTHESIZE:
            logger.warning("Invalid plan: must end with SYNTHESIZE")
            return False
        
        # Reasoning should come after web search if both present
        if StepType.SEARCH_WEB in plan.steps and StepType.REASONING in plan.steps:
            web_idx = plan.steps.index(StepType.SEARCH_WEB)
            reason_idx = plan.steps.index(StepType.REASONING)
            if reason_idx < web_idx:
                logger.warning("Invalid plan: REASONING should come after SEARCH_WEB")
                return False
        
        return True


# Module-level instance
execution_planner = ExecutionPlanner()
