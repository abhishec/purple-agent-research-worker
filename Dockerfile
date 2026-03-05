FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/

# Expose A2A port
EXPOSE 9010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:9010/health').raise_for_status()"

# Run server
CMD ["python", "-m", "uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "9010"]
