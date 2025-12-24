
import pytest
import asyncio
import os
import json
import time
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from app.analyzer.query_analyzer import QueryAnalyzer
from app.analyzer.contracts import AnalysisResult, StepType, WebSearchResult, ReasoningResult
from app.orchestrator.enhanced_orchestrator import EnhancedOrchestrator
from app.logs.decision_logger import DecisionLogger

# -----------------------------------------------------------------------------
# Test Suite 2: Multi-Agent Architecture
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_2_1_query_analyzer_routing():
    """Verify deterministic routing rules in QueryAnalyzer."""
    analyzer = QueryAnalyzer()
    
    # Test Web Search requirement
    web_query = "What is the current price of Bitcoin?"
    result = analyzer.analyze(web_query)
    assert result.needs_web is True, f"Failed to detect web need for: {web_query}"
    assert "current" in result.reason.lower() or "price" in result.reason.lower()
    
    # Test Reasoning requirement
    reason_query = "Analyze the impact of interest rates on housing market."
    result = analyzer.analyze(reason_query)
    assert result.needs_deep_reasoning is True, f"Failed to detect reasoning need for: {reason_query}"
    assert "analyze" in result.reason.lower() or "impact" in result.reason.lower()
    
    # Test Simple query
    simple_query = "Hello, how are you?"
    result = analyzer.analyze(simple_query)
    assert result.needs_web is False
    assert result.needs_deep_reasoning is False
    assert result.complexity_score < 0.3
    
    print("[Test 2.1] Passed")

@pytest.mark.asyncio
async def test_2_2_decision_logging_structure():
    """Verify that decisions are logged in a structured format."""
    # Use a temporary log path
    log_path = Path("test_decisions.jsonl")
    if log_path.exists():
        log_path.unlink()
        
    logger = DecisionLogger(log_path=str(log_path))
    
    query = "Test query for logging"
    analysis = AnalysisResult(
        needs_web=True,
        needs_deep_reasoning=False,
        complexity_score=0.5,
        confidence=0.9,
        reason="Test reason"
    )
    
    trace_id = logger.log_analysis(query, analysis)
    assert trace_id.startswith("trace-")
    
    # Verify file content
    assert log_path.exists()
    with open(log_path, "r") as f:
        line = f.readline()
        entry = json.loads(line)
        assert entry["trace_id"] == trace_id
        assert entry["step_type"] == "analysis"
        assert entry["input_data"]["query"] == query
        assert entry["output_data"]["needs_web"] is True
        
    # Verify trace retrieval
    trace = logger.get_decision_trace(trace_id)
    assert len(trace) == 1
    assert trace[0]["step_type"] == "analysis"
    
    # Cleanup
    if log_path.exists():
        log_path.unlink()
        
    print("[Test 2.2] Passed")

@pytest.mark.asyncio
async def test_2_3_multi_agent_integration_mocked():
    """Verify full pipeline integration with mocked agents."""
    orchestrator = EnhancedOrchestrator()
    
    # Mock Web Agent
    mock_web_result = WebSearchResult(
        facts=["Fact A", "Fact B"],
        sources=["https://example.com/a", "https://example.com/b"],
        raw_snippets=["Snippet A", "Snippet B"],
        confidence=0.9,
        query_used="Test Query"
    )
    orchestrator.web_agent.execute = AsyncMock(return_value=mock_web_result)
    
    # Mock Reasoning Agent
    mock_reasoning_result = ReasoningResult(
        conclusion="Final logical conclusion.",
        reasoning_chain=["Step 1", "Step 2"],
        confidence=0.95,
        limitations=["Mock limitation"]
    )
    orchestrator.reasoning_agent.reason = AsyncMock(return_value=mock_reasoning_result)
    
    # Execute with forced flags to ensure all paths are hit
    query = "Why is the current technology evolving so fast?"
    response = await orchestrator.run(query, force_web=True, force_reasoning=True)
    
    assert response.answer is not None
    assert "Final logical conclusion" in response.answer
    assert "Reasoning" in response.answer
    assert "Sources" in response.answer
    assert response.trace_id is not None
    assert StepType.SEARCH_WEB.value in response.plan_executed
    assert StepType.REASONING.value in response.plan_executed
    
    # Verify logs were generated
    trace = orchestrator.get_trace(response.trace_id)
    assert len(trace["entries"]) >= 5 # Analysis, Plan, Web, Reasoning, Synthesis, Completion
    
    print("[Test 2.3] Passed")

@pytest.mark.asyncio
async def test_2_4_trace_retrieval():
    """Verify that traces can be retrieved by ID after execution."""
    orchestrator = EnhancedOrchestrator()
    
    # Mock minimal response
    query = "Simple test"
    response = await orchestrator.run(query)
    
    trace_id = response.trace_id
    trace = orchestrator.get_trace(trace_id)
    
    assert trace["trace_id"] == trace_id
    assert len(trace["entries"]) > 0
    
    # Check for specific step
    steps = [e["step_type"] for e in trace["entries"]]
    assert "analysis" in steps
    assert "completion" in steps
    
    print("[Test 2.4] Passed")

if __name__ == "__main__":
    # Setup for manual run if needed
    import asyncio
    async def run_all():
        await test_2_1_query_analyzer_routing()
        await test_2_2_decision_logging_structure()
        await test_2_3_multi_agent_integration_mocked()
        await test_2_4_trace_retrieval()
        print("\nAll Suite 2 tests passed!")

    asyncio.run(run_all())
