# PIKA

**P**rivate **I**ntelligent **K**nowledge **A**ssistant

PIKA is a fully self-hosted document intelligence system that lets you ask questions about your documents using local AI. Unlike cloud-based solutions, PIKA runs entirely on your own hardware—your documents never leave your network. It uses Ollama for local LLM inference, ChromaDB for vector storage, and sentence-transformers for embeddings, giving you complete control over your data and privacy.

## Quick Start

### Using the Setup Script (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/pika.git
cd pika

# Run the setup script
./setup.sh
```

The setup script will:
- Start the Docker containers
- Pull the default LLM model (mistral:7b)
- Open PIKA in your browser

### Using Docker Compose

```bash
# Start the services
docker compose up -d

# Pull the LLM model (first time only)
docker compose exec ollama ollama pull mistral:7b

# Open http://localhost:8000 in your browser
```

## Development

### Running Tests

PIKA includes a comprehensive test suite. To run tests:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest tests/ --cov=src/pika --cov-report=term-missing -v

# Or use the test script
./scripts/test.sh      # Linux/Mac
scripts\test.bat       # Windows
```

The test suite includes:
- **API tests** (`test_api.py`) - HTTP endpoint testing
- **Document tests** (`test_documents.py`) - Document processing and chunking
- **RAG tests** (`test_rag.py`) - Vector storage and retrieval
- **Auth tests** (`test_auth.py`) - Authentication and sessions
- **Security tests** (`test_security.py`) - Password hashing, CSRF, rate limiting
- **Queue tests** (`test_queue.py`) - Query queue and concurrency management
- **Production tests** (`test_production_readiness.py`) - Settings validation, cleanup, pooling
- **Backup tests** (`test_backup.py`) - Backup/restore functionality and security

Tests run without requiring Ollama (mocked) and complete in about 2-3 minutes.

### Test Coverage

View the coverage report after running tests:

```bash
# HTML report
open coverage_html/index.html    # Mac
xdg-open coverage_html/index.html  # Linux
start coverage_html\index.html   # Windows
```

## Features

### Multi-User Support
- User authentication with secure bcrypt password hashing
- Role-based access (admin/user roles)
- Per-user chat history and query tracking
- Session management with configurable expiry

### Query Queue System
- FIFO queue for fair query processing in multi-user environments
- Configurable concurrency (1 for CPU, 2+ for GPU)
- Per-user queue limits to prevent one user blocking others
- Queue position and estimated wait time display
- Automatic timeout for stale queries

### Backup & Restore
- Full backup of documents, index, configuration, and user database
- Background backup with progress tracking
- One-click restore from ZIP archive
- Metadata included (version, timestamp)
- API support for automated backups (see [Automated Backups](#automated-backups))
- Configurable retention (default: keep last 5 backups)

### Security
- CSRF protection for all forms
- Rate limiting on queries and uploads
- Secure session management (in-memory; server restart requires re-login)
- Audit logging for admin actions
- Path traversal prevention

### Feedback System
- Thumbs up/down feedback on query responses
- Feedback stored for analytics purposes
- Per-query rating with question/answer context

### Observability
- Prometheus metrics endpoint at `/metrics`
- Structured JSON logging in production (human-readable in debug mode)
- Request latency histograms and counters
- Query performance metrics (latency, success/error rates, confidence levels)

### Reliability
- Automatic retry with exponential backoff for Ollama requests
- Graceful handling of connection failures and timeouts
- Configurable retry attempts (default: 3 retries)

## Hardware Requirements

| Component | Minimum (Evaluation) | Recommended (Production) |
|-----------|----------------------|--------------------------|
| RAM | 8 GB | 16 GB+ |
| Storage | 10 GB | 20 GB+ |
| CPU | 4 cores | 8+ cores |
| GPU | Not required | **NVIDIA GPU (required)** |

> **Important:** CPU-only mode is suitable for evaluation and testing only. For production use with acceptable response times, an NVIDIA GPU is required. Without a GPU, queries may take 30+ seconds to complete.

### Model Recommendations

| Use Case | Model | Size | Notes |
|----------|-------|------|-------|
| CPU evaluation | `phi3:mini` | 2.2 GB | Fastest on CPU, good for testing |
| GPU production | `llama3.1:8b` | 4.7 GB | Recommended for production use |
| GPU high quality | `llama3.1:70b` | 40 GB | Highest quality, requires more VRAM |

## Deployment Notes

### Single-Instance Architecture

PIKA is designed for single-instance deployment (one container/process). This keeps the architecture simple and is appropriate for small teams or personal use.

**Implications:**

- **Sessions**: Stored in-memory; server restarts require users to re-login
- **Query queue**: Stored in-memory; pending queries are lost on restart (users simply resubmit)
- **History/Feedback**: Stored in JSON files with thread locking; safe for concurrent requests within a single process
- **No horizontal scaling**: Running multiple PIKA instances would cause session inconsistencies and potential file conflicts

**For most use cases, this is fine.** PIKA is designed for small teams querying internal documents, not high-traffic public deployments. If you need multi-instance scaling, consider adding Redis for session storage and a shared database for history.

### HTTPS / Reverse Proxy

PIKA does not include built-in HTTPS. For production deployments, use a reverse proxy to handle TLS termination.

**Recommended:** [Caddy](https://caddyserver.com/) provides automatic HTTPS with zero configuration:

```
# Caddyfile
pika.yourdomain.com {
    reverse_proxy localhost:8000
}
```

Caddy automatically provisions and renews Let's Encrypt certificates.

**Alternatives:** nginx, Traefik, or any reverse proxy that supports TLS.

**Local/development use:** HTTPS is not required when accessing PIKA on `localhost` or within a trusted private network.

### Continuous Deployment

PIKA includes a GitHub Actions workflow for SSH deployment to multiple environments.

**Step 1: Generate SSH key (on your local machine)**

```bash
# Generate key pair with NO passphrase (required for automation)
ssh-keygen -t ed25519 -C "github-deploy" -f pika_deploy_key -N ""

# This creates two files:
# - pika_deploy_key     (private key → goes to GitHub)
# - pika_deploy_key.pub (public key → goes to VPS)
```

**Step 2: Add public key to your VPS**

```bash
# View the public key
cat pika_deploy_key.pub

# SSH into your VPS and add it
ssh user@your-vps-ip
echo "ssh-ed25519 AAAA... github-deploy" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

**Step 3: Create GitHub environments**

1. Go to your repo → **Settings → Environments**
2. Click **New environment** → name it `production`
3. (Optional) Enable "Required reviewers" for approval before deploy
4. Repeat for `staging` if needed

**Step 4: Add secrets to each environment**

In each environment, click **Add secret** for:

   | Secret | Example | Description |
   |--------|---------|-------------|
   | `VPS_HOST` | `203.0.113.50` | Server IP or hostname |
   | `VPS_USER` | `deploy` | SSH username |
   | `VPS_SSH_KEY` | `-----BEGIN OPENSSH...` | Entire contents of `pika_deploy_key` (private key) |
   | `VPS_PORT` | `22` | SSH port |
   | `VPS_APP_PATH` | `~/pika` | Path to PIKA on server |

**Step 5: Clone repo on VPS (first time only)**

```bash
git clone https://github.com/yourusername/pika.git ~/pika
cd ~/pika
docker compose up -d
```

**Step 6: Deploy**

1. Go to **GitHub → Actions → Deploy**
2. Click **Run workflow**
3. Select environment (`production` or `staging`)
4. Click **Run workflow**

The workflow will SSH into your server, pull latest code, and restart containers.

## Configuration

PIKA is configured via environment variables. Set these in your `docker-compose.yml` or `.env` file:

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | LLM model to use for generation |
| `OLLAMA_TIMEOUT` | `120` | Timeout in seconds for LLM requests |
| `DOCUMENTS_DIR` | `./documents` | Directory for uploaded documents |

### RAG Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `500` | Number of characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Number of relevant chunks to retrieve |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model for embeddings |
| `INDEX_TIMEOUT` | `600` | Timeout in seconds for async indexing operations |

### Vector Store Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Directory for ChromaDB persistence |

### Confidence Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_HIGH` | `0.7` | Threshold for high confidence answers |
| `CONFIDENCE_MEDIUM` | `0.5` | Threshold for medium confidence answers |
| `CONFIDENCE_LOW` | `0.3` | Threshold for low confidence answers |

### Authentication Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_ENABLED` | `true` | Enable/disable authentication |
| `SESSION_SECRET` | *auto-generated* | Secret key for session encryption |
| `SESSION_EXPIRY` | `86400` | Session lifetime in seconds (24 hours) |

### Query Queue Settings (Multi-User)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_QUERIES` | `1` | Max queries running simultaneously |
| `MAX_QUEUED_PER_USER` | `3` | Max pending queries per user |
| `QUEUE_TIMEOUT` | `300` | Seconds before queued query times out |
| `MAX_QUEUE_SIZE` | `100` | Maximum total queue length |

### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_QUERY` | `10/minute` | Query rate limit per user |
| `RATE_LIMIT_UPLOAD` | `20/minute` | Upload rate limit per user |
| `MAX_UPLOAD_SIZE` | `52428800` | Max upload size in bytes (50 MB) |

### Backup Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKUP_RETENTION_COUNT` | `5` | Number of backups to keep (0 = unlimited) |

## Supported File Types

| Format | Extension | Description |
|--------|-----------|-------------|
| PDF | `.pdf` | Adobe PDF documents |
| Word | `.docx` | Microsoft Word documents |
| Text | `.txt` | Plain text files |
| Markdown | `.md` | Markdown files |

## Troubleshooting

### Container won't start: "service unhealthy"

The Ollama container may take time to initialize, especially on first run. Wait a minute and try again:

```bash
docker compose down
docker compose up -d
```

### "Query failed" or timeout errors

This usually means the LLM is taking too long to respond. Try:

1. **Increase the timeout** in `docker-compose.yml`:
   ```yaml
   environment:
     - OLLAMA_TIMEOUT=300
   ```

2. **Use a smaller model** for faster responses:
   ```bash
   docker compose exec ollama ollama pull phi3:mini
   ```
   Then update `OLLAMA_MODEL=phi3:mini` in your configuration.

3. **Check Ollama logs** for errors:
   ```bash
   docker compose logs ollama
   ```

### Model not found

If you see "model not found" errors, pull the model first:

```bash
docker compose exec ollama ollama pull mistral:7b
```

### Out of memory errors

LLMs require significant RAM. If you're running out of memory:

1. Close other applications
2. Use a smaller model (e.g., `phi3:mini` instead of `mistral:7b`)
3. Increase your system's swap space

### Documents not appearing after upload

After uploading documents, click **"Refresh Index"** to process and index them. Documents must be indexed before they can be queried.

### Indexing large document sets

For large document collections, indexing may take several minutes. PIKA supports async indexing with progress reporting via the API:

- `POST /api/v1/index/start` - Start background indexing (returns immediately)
- `GET /api/v1/index/status` - Poll for progress (shows current file, percent complete)
- `POST /api/v1/index/cancel` - Cancel a running indexing operation
- `GET /api/v1/index/info` - Get combined stats and document list (optimized, cached)

The default timeout for indexing is 10 minutes (600 seconds). Adjust with the `INDEX_TIMEOUT` environment variable if needed.

**Performance note:** Stats and document lists are cached for 60 seconds to improve admin page load times with large document sets.

### Poor answer quality

If answers are incorrect or irrelevant:

1. Check that documents are indexed (see sidebar for document/chunk count)
2. Try rephrasing your question
3. Ensure your documents contain the relevant information
4. Check the confidence level—low confidence answers may be unreliable

### Checking logs

View detailed logs for debugging:

```bash
# All services
docker compose logs

# PIKA app only
docker compose logs pika

# Ollama only
docker compose logs ollama

# Follow logs in real-time
docker compose logs -f
```

## Automated Backups

PIKA supports automated backups via API, enabling integration with cron or other schedulers.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/backup/start` | POST | Start a background backup |
| `/admin/backup/status` | GET | Check backup progress |
| `/admin/backup/download` | GET | Download completed backup |

All endpoints require authentication via `X-API-Key` header or admin session.

### Cron Example (Linux/Mac)

```bash
# Daily backup at 2 AM, download to /backups/
0 2 * * * curl -X POST -H "X-API-Key: your-api-key" http://localhost:8000/admin/backup/start && \
          sleep 60 && \
          curl -H "X-API-Key: your-api-key" http://localhost:8000/admin/backup/download -o /backups/pika_$(date +\%Y\%m\%d).zip
```

### Backup Retention

PIKA automatically deletes old backups based on the `BACKUP_RETENTION_COUNT` setting (default: 5). Set to `0` to keep all backups.

```yaml
environment:
  - BACKUP_RETENTION_COUNT=10  # Keep last 10 backups
```

### Generating an API Key

During initial setup, check "Generate API Key" to create a key for automated access. The key is displayed once—store it securely.

## Observability

### Prometheus Metrics

PIKA exposes Prometheus metrics at `/metrics`. Available metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `pika_http_requests_total` | Counter | HTTP requests by method, endpoint, status |
| `pika_http_request_duration_seconds` | Histogram | Request latency |
| `pika_queries_total` | Counter | RAG queries by status and confidence |
| `pika_query_duration_seconds` | Histogram | Query processing time |
| `pika_active_queries` | Gauge | Queries currently processing |
| `pika_queued_queries` | Gauge | Queries waiting in queue |
| `pika_index_documents_total` | Gauge | Documents in index |
| `pika_index_chunks_total` | Gauge | Chunks in index |
| `pika_ollama_healthy` | Gauge | Ollama health status |

Example Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: 'pika'
    static_configs:
      - targets: ['localhost:8000']
```

### Structured Logging

In production (DEBUG=false), logs are output as JSON for easy parsing:

```json
{"timestamp": "2024-01-15 10:30:45", "level": "INFO", "logger": "pika.main", "message": "Starting PIKA v0.1.0"}
```

In development (DEBUG=true), logs use human-readable format.

## License

MIT License - See [LICENSE](LICENSE) for details.
