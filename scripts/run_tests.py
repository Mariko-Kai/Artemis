
import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Artemis Integration Tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Set environment variables for testing
    os.environ["TESTING"] = "True"
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///"
    
    # Ensure we are in the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Add project root and backend to PYTHONPATH
    backend_path = os.path.join(project_root, "backend")
    sys.path.insert(0, project_root)
    sys.path.insert(0, backend_path)
    
    path_sep = os.pathsep
    os.environ["PYTHONPATH"] = f"{project_root}{path_sep}{backend_path}{path_sep}" + os.environ.get("PYTHONPATH", "")

    print("Starting Artemis Integration Test Suite...")
    
    cmd = [sys.executable, "-m", "pytest"]
    if args.verbose:
        cmd.append("-v")
    
    cmd.extend([
        "tests/test_memory_workflow.py",
        "tests/test_preemption.py"
    ])

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("\nAll tests passed!")
            sys.exit(0)
        else:
            print(f"\nSome tests failed (Exit Code: {result.returncode})")
            sys.exit(result.returncode)
    except Exception as e:
        print(f"\nError running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
