#!/usr/bin/env python3
"""
Lint script to run all linting tools on the project.
"""

import subprocess
import sys
from typing import List


def run_command(command: List[str]) -> bool:
    """Run a command and return True if successful, False otherwise."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        print(result.stdout)
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True


def main() -> int:
    """Run all linting tools."""
    print("Running full linting checks...")
    
    success = True
    
    # Run flake8 with all checks
    if not run_command(["flake8", "solr_mcp", "tests"]):
        success = False
    
    # Run mypy type checking
    if not run_command(["mypy", "solr_mcp", "tests"]):
        success = False
    
    if success:
        print("All linting checks passed!")
        return 0
    else:
        print("Some linting checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())