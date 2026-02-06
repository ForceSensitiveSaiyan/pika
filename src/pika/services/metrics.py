"""Prometheus metrics for PIKA observability."""

from prometheus_client import Counter, Gauge, Histogram, Info

# Application info
APP_INFO = Info("pika", "PIKA application information")

# Request metrics
REQUEST_COUNT = Counter(
    "pika_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "pika_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# Query metrics
QUERY_COUNT = Counter(
    "pika_queries_total",
    "Total RAG queries",
    ["status", "confidence"],
)

QUERY_LATENCY = Histogram(
    "pika_query_duration_seconds",
    "Query processing time in seconds",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

ACTIVE_QUERIES = Gauge(
    "pika_active_queries",
    "Number of queries currently being processed",
)

QUEUED_QUERIES = Gauge(
    "pika_queued_queries",
    "Number of queries waiting in queue",
)

# Index metrics
INDEX_DOCUMENTS = Gauge(
    "pika_index_documents_total",
    "Total documents in index",
)

INDEX_CHUNKS = Gauge(
    "pika_index_chunks_total",
    "Total chunks in index",
)

INDEXING_IN_PROGRESS = Gauge(
    "pika_indexing_in_progress",
    "Whether indexing is currently running (1) or not (0)",
)

# Ollama metrics
OLLAMA_HEALTHY = Gauge(
    "pika_ollama_healthy",
    "Whether Ollama is healthy (1) or not (0)",
)

OLLAMA_REQUEST_COUNT = Counter(
    "pika_ollama_requests_total",
    "Total requests to Ollama",
    ["status"],
)

OLLAMA_REQUEST_LATENCY = Histogram(
    "pika_ollama_request_duration_seconds",
    "Ollama request latency in seconds",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# Session metrics
ACTIVE_SESSIONS = Gauge(
    "pika_active_sessions",
    "Number of active user sessions",
)

# Circuit breaker metrics
CIRCUIT_BREAKER_STATE = Gauge(
    "pika_circuit_breaker_state",
    "Circuit breaker state: 0=closed, 1=half_open, 2=open",
)

CIRCUIT_BREAKER_TRIPS = Counter(
    "pika_circuit_breaker_trips_total",
    "Total number of times the circuit breaker has tripped open",
)

# Query cache metrics
QUERY_CACHE_HITS = Counter(
    "pika_query_cache_hits_total",
    "Total query cache hits",
)

QUERY_CACHE_MISSES = Counter(
    "pika_query_cache_misses_total",
    "Total query cache misses",
)


def set_app_info(version: str, model: str) -> None:
    """Set application info metrics."""
    APP_INFO.info({
        "version": version,
        "ollama_model": model,
    })


def update_index_metrics(documents: int, chunks: int) -> None:
    """Update index-related metrics."""
    INDEX_DOCUMENTS.set(documents)
    INDEX_CHUNKS.set(chunks)


def update_queue_metrics(active: int, queued: int) -> None:
    """Update queue-related metrics."""
    ACTIVE_QUERIES.set(active)
    QUEUED_QUERIES.set(queued)
