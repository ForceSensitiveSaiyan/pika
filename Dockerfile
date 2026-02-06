FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (gosu for privilege de-escalation in entrypoint)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip (base image ships 24.0 which has SSL download issues)
RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch first (Ollama handles GPU inference separately)
# Pin numpy<2.0 here too so torch doesn't pull in numpy 2.x
# Shell retry loop handles Docker Desktop SSL mid-stream failures on Windows
RUN for i in 1 2 3 4 5; do \
        pip install --retries 3 --timeout 120 "numpy<2.0" torch --index-url https://download.pytorch.org/whl/cpu \
        && break || echo "Retry $i..." && sleep 10; \
    done && pip cache purge

RUN for i in 1 2 3 4 5; do \
        pip install --retries 3 --timeout 120 -r requirements.txt \
        && break || echo "Retry $i..." && sleep 10; \
    done && pip cache purge

# Copy application code
COPY src/ ./src/
COPY pyproject.toml .

# Create directories for data persistence
RUN mkdir -p /app/documents /app/data/chroma /app/data/hf_cache

# Create non-root user
RUN groupadd --system pika && useradd --system --gid pika pika \
    && chown -R pika:pika /app

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src
ENV DOCUMENTS_DIR=/app/documents
ENV CHROMA_PERSIST_DIR=/app/data/chroma
ENV HF_HOME=/app/data/hf_cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Entrypoint fixes volume permissions then drops to non-root user
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "src.pika.main:app", "--host", "0.0.0.0", "--port", "8000"]
