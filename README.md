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

## Screenshots

<!-- Screenshots will be added here -->

*Coming soon*

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

## License

MIT License - See [LICENSE](LICENSE) for details.
