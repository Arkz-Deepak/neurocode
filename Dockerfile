FROM python:3.12-slim

WORKDIR /app

# Copy project configuration and sources
COPY pyproject.toml README.md LICENSE /app/
COPY src/ /app/src/

# Install neurocode
RUN pip install --no-cache-dir -e .

# Default entrypoint starts the stdio MCP server for introspection
ENTRYPOINT ["neurocode-mcp"]
