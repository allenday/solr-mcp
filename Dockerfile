FROM python:3.10-slim

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry==2.1.1

# Copy essential files first
COPY pyproject.toml poetry.lock* README.md ./
COPY solr_mcp ./solr_mcp

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --without dev --no-interaction --no-ansi

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SOLR_MCP_ZK_HOSTS=zookeeper:2181 \
    SOLR_MCP_SOLR_URL=http://solr1:8983/solr \
    SOLR_MCP_DEFAULT_COLLECTION=unified \
    OLLAMA_BASE_URL=http://ollama:11434

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "solr_mcp.server"]