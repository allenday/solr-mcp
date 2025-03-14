#!/usr/bin/env python3
"""
Direct MCP server test script.
Tests the raw JSON-RPC interface that Claude uses to communicate with MCP servers.
"""

import sys
import os
import json
import subprocess
import time
from threading import Thread
import tempfile

# Add the project root to your path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# First clean up any existing MCP servers
os.system("pkill -f 'python -m solr_mcp.server'")
time.sleep(1)  # Let them shut down

def write_to_stdin(process, data):
    """Write data to the stdin of a process and flush."""
    process.stdin.write(data)
    process.stdin.flush()

def read_from_stdout(process):
    """Read a JSON-RPC message from stdout of a process."""
    line = process.stdout.readline().strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {line}")
        return None

# Start a new MCP server process
cmd = ["python", "-m", "solr_mcp.server"]
server_process = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,  # Line buffered
)

print("MCP server started.")
time.sleep(2)  # Give it time to initialize

# Test search methods
def test_search(query):
    print(f"\n\nTesting search for: '{query}'")
    
    # Try a standard search
    request = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "execute_tool",
        "params": {
            "name": "solr_search",
            "arguments": {
                "query": query
            }
        }
    }
    
    print("\nSending search request:", json.dumps(request, indent=2))
    write_to_stdin(server_process, json.dumps(request) + "\n")
    response = read_from_stdout(server_process)
    print("\nGot response:", json.dumps(response, indent=2) if response else "No response")
    
    # Try a hybrid search
    request = {
        "jsonrpc": "2.0",
        "id": "2",
        "method": "execute_tool",
        "params": {
            "name": "solr_hybrid_search",
            "arguments": {
                "query": query,
                "blend_factor": 0.5
            }
        }
    }
    
    print("\nSending hybrid search request:", json.dumps(request, indent=2))
    write_to_stdin(server_process, json.dumps(request) + "\n")
    response = read_from_stdout(server_process)
    print("\nGot hybrid response:", json.dumps(response, indent=2) if response else "No response")

# Test with a query we know exists
test_search("double spend")

# Test with another query
test_search("blockchain")

# Clean up
print("\nCleaning up...")
server_process.terminate()
server_process.wait()
print("Done!")