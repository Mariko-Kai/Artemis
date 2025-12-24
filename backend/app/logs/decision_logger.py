"""
Decision Logger - Structured logging for agent decisions.

All agent decisions are logged to:
1. File (JSONL format for easy parsing)
2. Database (for querying and analysis)

This enables:
- Full explainability of routing decisions
- Debugging and troubleshooting
- Performance analysis
- Audit trails
"""
import json
import logging
import uuid
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from app.analyzer.contracts import (
    AnalysisResult, ExecutionPlan, WebSearchResult, 
    ReasoningResult, DecisionLogEntry
)

logger = logging.getLogger(__name__)


class DecisionLogger:
    """
    Logs all agent decisions to file and optionally to database.
    
    All logs are structured JSON for easy parsing and analysis.
    """
    
    def __init__(self, log_path: Optional[str] = None):
        """
        Initialize the Decision Logger.
        
        Args:
            log_path: Path to the JSONL log file. Defaults to ./logs/agent_decisions.jsonl
        """
        self.log_path = Path(log_path or "./logs/agent_decisions.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory trace storage for current session
        self._traces: Dict[str, List[DecisionLogEntry]] = {}
    
    def generate_trace_id(self) -> str:
        """Generate a unique trace ID for a request."""
        return f"trace-{uuid.uuid4().hex[:12]}"
    
    def log_analysis(self, query: str, result: AnalysisResult, trace_id: Optional[str] = None) -> str:
        """
        Log query analysis decision.
        
        Args:
            query: The original user query
            result: The analysis result
            trace_id: Optional trace ID (will be generated if not provided)
            
        Returns:
            The trace ID used
        """
        trace_id = trace_id or self.generate_trace_id()
        
        entry = DecisionLogEntry(
            trace_id=trace_id,
            step_type="analysis",
            input_data={"query": query},
            output_data=result.model_dump()
        )
        
        self._write_log(entry)
        self._store_trace(trace_id, entry)
        
        logger.info(
            f"[{trace_id}] ANALYSIS: web={result.needs_web}, "
            f"reasoning={result.needs_deep_reasoning}, "
            f"confidence={result.confidence:.2f}"
        )
        
        return trace_id
    
    def log_plan(self, plan: ExecutionPlan, trace_id: str) -> None:
        """
        Log execution plan.
        
        Args:
            plan: The execution plan
            trace_id: The trace ID for this request
        """
        entry = DecisionLogEntry(
            trace_id=trace_id,
            step_type="planning",
            input_data={"context": plan.context},
            output_data={"steps": [s.value for s in plan.steps]}
        )
        
        self._write_log(entry)
        self._store_trace(trace_id, entry)
        
        logger.info(f"[{trace_id}] PLAN: {[s.value for s in plan.steps]}")
    
    def log_step_execution(
        self, 
        step: str, 
        input_data: Any, 
        output_data: Any, 
        trace_id: str,
        duration_ms: Optional[float] = None
    ) -> None:
        """
        Log a step execution.
        
        Args:
            step: The step type (search_web, reasoning, synthesize)
            input_data: Input to this step
            output_data: Output from this step
            trace_id: The trace ID for this request
            duration_ms: Execution time in milliseconds
        """
        # Convert Pydantic models to dict if needed
        if hasattr(input_data, "model_dump"):
            input_data = input_data.model_dump()
        elif not isinstance(input_data, dict):
            input_data = {"value": str(input_data)}
            
        if hasattr(output_data, "model_dump"):
            output_data = output_data.model_dump()
        elif not isinstance(output_data, dict):
            output_data = {"value": str(output_data)[:500]}  # Truncate long outputs
        
        entry = DecisionLogEntry(
            trace_id=trace_id,
            step_type=f"execution:{step}",
            input_data=input_data,
            output_data=output_data,
            duration_ms=duration_ms
        )
        
        self._write_log(entry)
        self._store_trace(trace_id, entry)
        
        duration_str = f" ({duration_ms:.1f}ms)" if duration_ms else ""
        logger.info(f"[{trace_id}] STEP {step.upper()}{duration_str}")
    
    def log_error(self, error: Exception, trace_id: str, step: Optional[str] = None) -> None:
        """
        Log an error during execution.
        
        Args:
            error: The exception that occurred
            trace_id: The trace ID for this request
            step: The step where the error occurred
        """
        entry = DecisionLogEntry(
            trace_id=trace_id,
            step_type=f"error:{step}" if step else "error",
            input_data={"step": step},
            output_data={
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
        )
        
        self._write_log(entry)
        self._store_trace(trace_id, entry)
        
        logger.error(f"[{trace_id}] ERROR in {step}: {error}")
    
    def log_completion(self, trace_id: str, success: bool, total_duration_ms: float) -> None:
        """
        Log request completion.
        
        Args:
            trace_id: The trace ID for this request
            success: Whether the request completed successfully
            total_duration_ms: Total execution time
        """
        entry = DecisionLogEntry(
            trace_id=trace_id,
            step_type="completion",
            input_data={},
            output_data={
                "success": success,
                "total_duration_ms": total_duration_ms
            },
            duration_ms=total_duration_ms
        )
        
        self._write_log(entry)
        self._store_trace(trace_id, entry)
        
        status = "SUCCESS" if success else "FAILURE"
        logger.info(f"[{trace_id}] {status} in {total_duration_ms:.1f}ms")
    
    def get_decision_trace(self, trace_id: str) -> List[Dict[str, Any]]:
        """
        Get the full decision trace for a request.
        
        Args:
            trace_id: The trace ID to retrieve
            
        Returns:
            List of log entries for this trace
        """
        if trace_id in self._traces:
            return [e.model_dump() for e in self._traces[trace_id]]
        
        # If not in memory, try to read from file
        return self._read_trace_from_file(trace_id)
    
    def _write_log(self, entry: DecisionLogEntry) -> None:
        """Write a log entry to file."""
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to write decision log: {e}")
    
    def _store_trace(self, trace_id: str, entry: DecisionLogEntry) -> None:
        """Store entry in memory for quick retrieval."""
        if trace_id not in self._traces:
            self._traces[trace_id] = []
        self._traces[trace_id].append(entry)
        
        # Limit memory usage - keep only last 100 traces
        if len(self._traces) > 100:
            oldest = next(iter(self._traces))
            del self._traces[oldest]
    
    def _read_trace_from_file(self, trace_id: str) -> List[Dict[str, Any]]:
        """Read trace from log file."""
        entries = []
        try:
            if self.log_path.exists():
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get("trace_id") == trace_id:
                                entries.append(entry)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Failed to read decision trace: {e}")
        return entries


# Module-level instance
decision_logger = DecisionLogger()
