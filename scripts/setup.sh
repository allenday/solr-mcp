#!/bin/bash
# Setup script for Solr MCP Server

set -e  # Exit immediately if a command exits with a non-zero status

echo "=== Setting up Solr MCP Server ==="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and Docker Compose first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install poetry
poetry install

# Start Docker containers
echo "Starting SolrCloud, ZooKeeper, and Ollama containers..."
docker-compose up -d

# Wait for Solr to be ready
echo "Waiting for SolrCloud to be ready..."
sleep 10
attempts=0
max_attempts=30
while ! curl -s http://localhost:8983/solr/ > /dev/null; do
    attempts=$((attempts+1))
    if [ $attempts -ge $max_attempts ]; then
        echo "Error: SolrCloud did not start in time. Please check docker-compose logs."
        exit 1
    fi
    echo "Waiting for SolrCloud to start... (attempt $attempts/$max_attempts)"
    sleep 5
done

# Create unified collection
echo "Creating unified collection..."
python scripts/create_unified_collection.py

# Process demo data (Bitcoin whitepaper)
echo "Processing demo data..."
python scripts/process_markdown.py data/bitcoin-whitepaper.md --output data/processed/bitcoin_sections.json

# Index demo data to unified collection
echo "Indexing demo data to unified collection..."
python scripts/unified_index.py data/processed/bitcoin_sections.json --collection unified

# Test search to ensure content is indexed properly
echo "Testing search functionality..."
python -c "
import httpx
import asyncio

async def test_search():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            'http://localhost:8983/solr/unified/select',
            params={
                'q': 'content:\"double spend\"~5',
                'wt': 'json'
            }
        )
        results = response.json()
        if results.get('response', {}).get('numFound', 0) > 0:
            print('✅ Search test successful! Found documents matching \"double spend\"')
        else:
            print('❌ Warning: No documents found for \"double spend\". Search may not work properly.')
            print('   Try running: python scripts/diagnose_search.py --collection unified --term \"double spend\"')

asyncio.run(test_search())
"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "You can now use the Solr MCP server with the following commands:"
echo ""
echo "1. Start the MCP server:"
echo "   python -m solr_mcp.server"
echo ""
echo "2. Try hybrid search on the demo data:"
echo "   python scripts/demo_hybrid_search.py \"blockchain\" --mode compare"
echo ""
echo "3. Use the Claude Desktop integration by configuring the MCP server"
echo "   in Claude's configuration file (see README.md for details)."
echo ""
echo "For more information, please refer to the documentation in README.md."