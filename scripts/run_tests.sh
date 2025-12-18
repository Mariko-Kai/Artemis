#!/bin/bash
set -e

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default to running all tests if no args
ARGS=${@:-tests/memory}

echo "Running tests with coverage..."
echo "Target: $ARGS"

# Set PYTHONPATH to include root
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Run pytest with coverage
# --cov=memory_module: Measure coverage for memory_module
# --cov-report=term-missing: Show missing lines in terminal
# -v: Verbose
python -m pytest $ARGS --cov=memory_module --cov-report=term-missing --cov-report=html:coverage_html -v
