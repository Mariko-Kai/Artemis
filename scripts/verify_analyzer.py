import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from app.analyzer.query_analyzer import QueryAnalyzer

def test_analyzer():
    analyzer = QueryAnalyzer()
    
    test_cases = [
        {
            "query": "What is the current price of Bitcoin?",
            "expected_web": True,
            "expected_reasoning": False
        },
        {
            "query": "Analyze the economic impact of AI on European labor markets",
            "expected_web": True,
            "expected_reasoning": True
        },
        {
            "query": "How would a 2% interest rate hike change the housing market in 2025?",
            "expected_web": True,
            "expected_reasoning": True
        },
        {
            "query": "Define quantum entanglement",
            "expected_web": False,
            "expected_reasoning": False
        },
        {
            "query": "Hello, how are you?",
            "expected_web": False,
            "expected_reasoning": False
        },
        {
            "query": "What happened in the news today?",
            "expected_web": True,
            "expected_reasoning": False
        }
    ]
    
    print("Starting Query Analyzer Verification...\n")
    all_passed = True
    
    for tc in test_cases:
        result = analyzer.analyze(tc["query"])
        
        web_ok = result.needs_web == tc["expected_web"]
        reason_ok = result.needs_deep_reasoning == tc["expected_reasoning"]
        
        status = "PASS" if web_ok and reason_ok else "FAIL"
        if status == "FAIL":
            all_passed = False
            
        print(f"Query: {tc['query']}")
        print(f"Result: web={result.needs_web}, reasoning={result.needs_deep_reasoning}, complexity={result.complexity_score}")
        print(f"Reason: {result.reason}")
        print(f"Status: {status}\n")
        
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED.")

if __name__ == "__main__":
    test_analyzer()
