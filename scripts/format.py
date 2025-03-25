#!/usr/bin/env python3
"""
Format script to run all code formatters on the project.
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
    """Run all code formatters."""
    print("Running code formatters...")
    
    success = True
    
    # Run black
    if not run_command(["black", "solr_mcp", "tests"]):
        success = False
    
    # Run isort
    if not run_command(["isort", "solr_mcp", "tests"]):
        success = False
    
    if success:
        print("All formatting completed successfully!")
        return 0
    else:
        print("Some formatting commands failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())