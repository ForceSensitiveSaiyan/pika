"""RAG engine using ChromaDB and sentence-transformers."""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import AsyncIterator, Callable

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from pika.config import Settings, get_settings
from pika.services.documents import (
    DocumentProcessor,
    FileTooLargeError,
    get_document_processor,
)
from pika.services.ollama import (
    OllamaClient,
    OllamaCircuitOpenError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    get_circuit_breaker,
    get_ollama_client,
)

logger = logging.getLogger(__name__)

# Cache TTL in seconds
STATS_CACHE_TTL = 60  # 1 minute cache for stats
DOCS_CACHE_TTL = 60   # 1 minute cache for indexed documents list

# Query status cleanup settings
QUERY_STATUS_TTL = 300  # 5 minutes - completed queries are cleaned up after this
QUERY_CLEANUP_INTERVAL = 60  # Run cleanup every 60 seconds


class Confidence(str, Enum):
    """Confidence level for query results."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"
    DEGRADED = "degraded"  # Search-only mode when Ollama unavailable


def _format_degraded_response(sources: list["Source"]) -> str:
    """Format a degraded response when Ollama is unavailable.

    Returns a readable answer showing the top source snippets.
    """
    if not sources:
        return (
            "The AI assistant is temporarily unavailable, and no relevant documents "
            "were found for your query."
        )

    lines = [
        "The AI assistant is temporarily unavailable. "
        "Here are the most relevant sections from your documents:\n"
    ]

    for source in sources[:3]:  # Show top 3 sources
        # Truncate content to first 200 chars
        content = source.content[:200].strip()
        if len(source.content) > 200:
            content += "..."
        lines.append(f"**{source.filename}**: {content}\n")

    return "\n".join(lines)


class QueryCache:
    """LRU cache for query results with TTL expiration.

    Reduces Ollama load by caching identical queries.
    """

    def __init__(self, max_size: int = 100, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self._cache: dict[str, tuple[float, "QueryResult"]] = {}
        self._lock = Lock()

    def _make_key(self, question: str, doc_count: int, chunk_count: int) -> str:
        """Generate a cache key from query parameters."""
        import hashlib
        normalized = question.lower().strip()
        key_string = f"{normalized}:{doc_count}:{chunk_count}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def get(self, question: str, doc_count: int, chunk_count: int) -> "QueryResult | None":
        """Get a cached result if it exists and is not expired."""
        from pika.services.metrics import QUERY_CACHE_HITS, QUERY_CACHE_MISSES

        key = self._make_key(question, doc_count, chunk_count)

        with self._lock:
            if key not in self._cache:
                QUERY_CACHE_MISSES.inc()
                return None

            timestamp, result = self._cache[key]
            if time.time() - timestamp > self.ttl:
                # Expired
                del self._cache[key]
                QUERY_CACHE_MISSES.inc()
                return None

            # Move to end (most recently used)
            del self._cache[key]
            self._cache[key] = (timestamp, result)
            QUERY_CACHE_HITS.inc()
            return result

    def set(self, question: str, doc_count: int, chunk_count: int, result: "QueryResult") -> None:
        """Cache a query result."""
        key = self._make_key(question, doc_count, chunk_count)

        with self._lock:
            # If at max size, remove oldest entry (LRU)
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[key] = (time.time(), result)

    def invalidate(self) -> None:
        """Clear the entire cache. Call on index rebuild."""
        with self._lock:
            self._cache.clear()
            logger.info("[QueryCache] Cache invalidated")

    def size(self) -> int:
        """Return current cache size."""
        with self._lock:
            return len(self._cache)


# Global query cache instance
_query_cache: QueryCache | None = None


def get_query_cache() -> QueryCache:
    """Get or create the query cache singleton."""
    global _query_cache
    if _query_cache is None:
        from pika.config import get_settings
        settings = get_settings()
        _query_cache = QueryCache(
            max_size=settings.query_cache_max_size,
            ttl=settings.query_cache_ttl,
        )
    return _query_cache


def invalidate_query_cache() -> None:
    """Invalidate the query cache. Call on index rebuild."""
    global _query_cache
    if _query_cache is not None:
        _query_cache.invalidate()


@dataclass
class Source:
    """A source document referenced in the answer."""

    filename: str
    chunk_index: int
    content: str
    similarity: float


@dataclass
class QueryResult:
    """Result from a RAG query."""

    answer: str
    sources: list[Source]
    confidence: Confidence


@dataclass
class IndexStats:
    """Statistics about the current index."""

    total_documents: int
    total_chunks: int
    collection_name: str


@dataclass
class IndexedDocument:
    """Information about an indexed document."""

    filename: str
    chunk_count: int


@dataclass
class QueuedQuery:
    """A query waiting in the queue."""

    query_id: str
    question: str
    username: str | None
    top_k: int | None
    queued_at: datetime = field(default_factory=datetime.now)


@dataclass
class QueueStats:
    """Statistics for estimating wait times."""

    recent_durations: list[float] = field(default_factory=list)
    max_samples: int = 10

    def record_duration(self, duration: float) -> None:
        """Record a query duration for averaging."""
        self.recent_durations.append(duration)
        if len(self.recent_durations) > self.max_samples:
            self.recent_durations.pop(0)

    def get_average_duration(self) -> float:
        """Get average query duration, default 30s if no data."""
        if not self.recent_durations:
            return 30.0  # Default estimate
        return sum(self.recent_durations) / len(self.recent_durations)


@dataclass
class QueryStatus:
    """Status of an active or completed query."""

    query_id: str
    question: str
    status: str = "pending"  # pending, queued, running, completed, error, cancelled
    result: "QueryResult | None" = None
    error: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None  # Set when query finishes (for TTL cleanup)
    # Queue-related fields
    queue_position: int | None = None
    queue_length: int | None = None
    estimated_wait_seconds: int | None = None

    def mark_completed(self) -> None:
        """Mark the query as completed and set completion time."""
        self.completed_at = datetime.now()

    def to_dict(self) -> dict:
        result_dict = None
        if self.result:
            result_dict = {
                "answer": self.result.answer,
                "sources": [
                    {
                        "filename": s.filename,
                        "chunk_index": s.chunk_index,
                        "content": s.content,
                        "similarity": s.similarity,
                    }
                    for s in self.result.sources
                ],
                "confidence": self.result.confidence.value,
            }
        return {
            "query_id": self.query_id,
            "question": self.question,
            "status": self.status,
            "result": result_dict,
            "error": self.error,
            "queue_position": self.queue_position,
            "queue_length": self.queue_length,
            "estimated_wait_seconds": self.estimated_wait_seconds,
        }


# Per-user query status tracking
_active_queries: dict[str, QueryStatus] = {}
_query_tasks: dict[str, asyncio.Task] = {}

# Key for anonymous users
ANONYMOUS_USER = "__anonymous__"

# Global query queue state
_query_queue: deque[QueuedQuery] = deque()
_queue_lock = asyncio.Lock()
_running_queries: set[str] = set()  # Set of query_ids currently running
_queue_stats = QueueStats()
_queue_processor_task: asyncio.Task | None = None
_queue_shutdown_event: asyncio.Event | None = None


def _get_user_key(username: str | None) -> str:
    """Get the key for storing user-specific query data."""
    return username if username else ANONYMOUS_USER


def get_active_query(username: str | None = None) -> QueryStatus | None:
    """Get the currently active or most recent query status for a user."""
    key = _get_user_key(username)
    return _active_queries.get(key)


def clear_query_status(username: str | None = None) -> None:
    """Clear the query status for a user."""
    key = _get_user_key(username)
    if key in _active_queries:
        del _active_queries[key]


def _set_query_status(status: QueryStatus | None, username: str | None = None) -> None:
    """Set the query status for a user."""
    key = _get_user_key(username)
    if status is None:
        if key in _active_queries:
            del _active_queries[key]
    else:
        _active_queries[key] = status


def cleanup_expired_queries() -> int:
    """Remove completed/errored queries older than TTL.

    Returns the number of queries cleaned up.
    """
    now = datetime.now()
    expired_keys = []

    for key, status in _active_queries.items():
        # Only clean up terminal states
        if status.status in ("completed", "error", "cancelled"):
            if status.completed_at:
                age = (now - status.completed_at).total_seconds()
                if age > QUERY_STATUS_TTL:
                    expired_keys.append(key)
            else:
                # No completed_at set but in terminal state - use started_at as fallback
                age = (now - status.started_at).total_seconds()
                if age > QUERY_STATUS_TTL * 2:  # More lenient for legacy entries
                    expired_keys.append(key)

    for key in expired_keys:
        del _active_queries[key]
        if key in _query_tasks:
            del _query_tasks[key]

    if expired_keys:
        logger.debug(f"[Queue] Cleaned up {len(expired_keys)} expired query statuses")

    return len(expired_keys)


# ==================== Queue Management ====================


def get_queue_length() -> int:
    """Get current queue length."""
    return len(_query_queue)


def get_running_count() -> int:
    """Get number of currently running queries."""
    return len(_running_queries)


def get_user_queued_count(username: str | None) -> int:
    """Count queries in queue for a specific user."""
    key = _get_user_key(username)
    return sum(1 for q in _query_queue if _get_user_key(q.username) == key)


def _update_queue_positions() -> None:
    """Recalculate queue positions for all queued queries."""
    settings = get_settings()
    avg_duration = _queue_stats.get_average_duration()
    running_count = len(_running_queries)
    max_concurrent = settings.max_concurrent_queries

    for i, queued in enumerate(_query_queue):
        key = _get_user_key(queued.username)
        status = _active_queries.get(key)
        if status and status.query_id == queued.query_id and status.status == "queued":
            position = i + 1
            queue_length = len(_query_queue)

            # Estimate wait: queries ahead / concurrent slots * avg duration
            # Plus consider currently running queries
            queries_ahead = i
            if running_count >= max_concurrent:
                # All slots full, need to wait for current queries + queue position
                estimated_wait = int((queries_ahead + 1) * avg_duration / max_concurrent)
            else:
                # Some slots available, might start soon
                estimated_wait = int(queries_ahead * avg_duration / max_concurrent)

            status.queue_position = position
            status.queue_length = queue_length
            status.estimated_wait_seconds = estimated_wait


def remove_from_queue(query_id: str) -> bool:
    """Remove a query from the queue (for cancellation).

    Returns True if query was found and removed.
    """
    # Find and remove the query
    found_idx = None
    for i, queued in enumerate(_query_queue):
        if queued.query_id == query_id:
            found_idx = i
            break

    if found_idx is not None:
        # Rebuild queue without the removed item
        items = list(_query_queue)
        _query_queue.clear()
        for i, item in enumerate(items):
            if i != found_idx:
                _query_queue.append(item)
        logger.info(f"Removed query {query_id} from queue position {found_idx + 1}")
        _update_queue_positions()
        return True
    return False


async def _execute_query(queued: QueuedQuery) -> None:
    """Execute a single query from the queue."""
    import time as time_module

    # Import here to avoid circular import
    from pika.services.audit import get_audit_logger
    from pika.services.app_config import get_app_config
    from pika.services.history import get_history_service

    key = _get_user_key(queued.username)
    status = _active_queries.get(key)

    if not status or status.query_id != queued.query_id:
        # Query was cancelled or replaced
        logger.info(f"Query {queued.query_id} no longer active, skipping execution")
        return

    # Update status to running
    status.status = "running"
    status.queue_position = None
    status.queue_length = None
    status.estimated_wait_seconds = None

    query_start = time_module.time()
    logger.info(f"[Queue] Executing query {queued.query_id}: '{queued.question[:50]}...'")

    try:
        rag = get_rag_engine()
        settings = get_settings()
        timeout = settings.queue_timeout

        result = await asyncio.wait_for(
            rag.query(question=queued.question, top_k=queued.top_k),
            timeout=timeout,
        )
        status.result = result
        status.status = "completed"
        status.mark_completed()
        elapsed = time_module.time() - query_start

        # Record duration for wait estimation
        _queue_stats.record_duration(elapsed)

        logger.info(f"[Queue] Query completed: {queued.query_id} in {elapsed:.1f}s")

        # Audit log
        audit = get_audit_logger()
        audit.log_query(
            question=queued.question,
            model=get_app_config().get_current_model(),
            confidence=result.confidence.value,
            sources=[s.filename for s in result.sources],
        )

        # Save to history
        history = get_history_service()
        history.add_query(
            question=queued.question,
            answer=result.answer,
            confidence=result.confidence.value,
            sources=[s.filename for s in result.sources],
            username=queued.username,
        )

    except asyncio.CancelledError:
        elapsed = time_module.time() - query_start
        status.status = "cancelled"
        status.error = "Query was cancelled"
        status.mark_completed()
        logger.info(f"[Queue] Query cancelled: {queued.query_id} after {elapsed:.1f}s")

    except asyncio.TimeoutError:
        elapsed = time_module.time() - query_start
        status.status = "error"
        status.error = f"Query timed out after {get_settings().queue_timeout} seconds"
        status.mark_completed()
        logger.error(f"[Queue] Query timed out: {queued.query_id} after {elapsed:.1f}s")

        audit = get_audit_logger()
        audit.log_query(
            question=queued.question,
            model=get_app_config().get_current_model(),
            confidence="none",
            sources=[],
            error="Query timed out",
        )

    except Exception as e:
        elapsed = time_module.time() - query_start
        status.status = "error"
        status.error = f"{type(e).__name__}: {e}"  # Include error type for debugging
        status.mark_completed()
        logger.error(f"[Queue] Query failed: {queued.query_id} after {elapsed:.1f}s - {type(e).__name__}: {e}")

        audit = get_audit_logger()
        audit.log_query(
            question=queued.question,
            model=get_app_config().get_current_model(),
            confidence="none",
            sources=[],
            error=str(e),
        )


async def _process_queue() -> None:
    """Background coroutine that processes queued queries."""
    global _queue_shutdown_event

    logger.info("[Queue] Queue processor started")
    settings = get_settings()
    last_cleanup_time = time.time()

    while True:
        # Check for shutdown
        if _queue_shutdown_event and _queue_shutdown_event.is_set():
            logger.info("[Queue] Shutdown signal received, stopping processor")
            break

        try:
            # Periodic cleanup of expired query statuses
            if time.time() - last_cleanup_time > QUERY_CLEANUP_INTERVAL:
                cleanup_expired_queries()
                last_cleanup_time = time.time()

            async with _queue_lock:
                # Check if we can start more queries
                if len(_running_queries) >= settings.max_concurrent_queries:
                    # All slots full, wait
                    pass
                elif _query_queue:
                    # Get next query from queue
                    queued = _query_queue.popleft()

                    # Check for timeout
                    wait_time = (datetime.now() - queued.queued_at).total_seconds()
                    if wait_time > settings.queue_timeout:
                        # Query timed out while waiting
                        key = _get_user_key(queued.username)
                        status = _active_queries.get(key)
                        if status and status.query_id == queued.query_id:
                            status.status = "error"
                            status.error = f"Query timed out after waiting {int(wait_time)} seconds in queue"
                            status.mark_completed()
                        logger.warning(f"[Queue] Query {queued.query_id} timed out in queue after {wait_time:.1f}s")
                        _update_queue_positions()
                        continue

                    # Track as running
                    _running_queries.add(queued.query_id)
                    _update_queue_positions()

                    # Execute in background task
                    async def run_and_cleanup(q: QueuedQuery):
                        try:
                            await _execute_query(q)
                        finally:
                            async with _queue_lock:
                                _running_queries.discard(q.query_id)

                    asyncio.create_task(run_and_cleanup(queued))

            # Small sleep to prevent busy loop
            await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("[Queue] Queue processor cancelled")
            break
        except Exception as e:
            logger.error(f"[Queue] Queue processor error: {e}")
            await asyncio.sleep(1)  # Back off on error

    logger.info("[Queue] Queue processor stopped")


async def init_queue_processor() -> None:
    """Initialize and start the queue processor."""
    global _queue_processor_task, _queue_shutdown_event

    _queue_shutdown_event = asyncio.Event()
    _queue_processor_task = asyncio.create_task(_process_queue())
    logger.info("[Queue] Queue processor initialized")


async def shutdown_queue_processor() -> None:
    """Shutdown the queue processor gracefully."""
    global _queue_processor_task, _queue_shutdown_event

    if _queue_shutdown_event:
        _queue_shutdown_event.set()

    if _queue_processor_task:
        _queue_processor_task.cancel()
        try:
            await _queue_processor_task
        except asyncio.CancelledError:
            pass
        _queue_processor_task = None

    logger.info("[Queue] Queue processor shutdown complete")


@dataclass
class IndexStatus:
    """Status of an active or completed indexing operation."""

    index_id: str
    status: str = "pending"  # pending, running, completed, error, cancelled
    total_documents: int = 0
    processed_documents: int = 0
    total_chunks: int = 0
    current_file: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    error: str | None = None

    @property
    def percent(self) -> int:
        """Calculate progress percentage."""
        if self.total_documents == 0:
            return 0
        return int((self.processed_documents / self.total_documents) * 100)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "index_id": self.index_id,
            "status": self.status,
            "total_documents": self.total_documents,
            "processed_documents": self.processed_documents,
            "total_chunks": self.total_chunks,
            "current_file": self.current_file,
            "percent": self.percent,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


# Global indexing status tracker (only one index operation at a time)
_active_index: IndexStatus | None = None
_index_task: asyncio.Task | None = None


def get_active_index() -> IndexStatus | None:
    """Get the currently active index status, if any."""
    return _active_index


def _set_active_index(status: IndexStatus | None) -> None:
    """Set the active index status."""
    global _active_index
    _active_index = status


def is_indexing_running() -> bool:
    """Check if an indexing task is currently running."""
    return _index_task is not None and not _index_task.done()


def cancel_index_task() -> bool:
    """Cancel the currently running index task.

    Returns True if a task was cancelled, False if no task was running.
    """
    global _index_task, _active_index

    if _index_task is None or _index_task.done():
        return False

    _index_task.cancel()

    if _active_index:
        _active_index.status = "cancelled"
        _active_index.error = "Indexing cancelled by user"
        _active_index.completed_at = datetime.now()

    logger.info("Index task cancelled by user")
    return True


async def start_index_task(timeout: int | None = None) -> IndexStatus:
    """Start a background indexing task.

    Args:
        timeout: Optional timeout in seconds (default from settings.index_timeout)

    Returns:
        IndexStatus object tracking the indexing progress
    """
    global _index_task

    # Import here to avoid circular import
    from pika.services.audit import get_audit_logger

    if is_indexing_running():
        # Return existing status if already running
        return _active_index

    # Generate unique index ID
    import uuid
    index_id = str(uuid.uuid4())[:8]

    # Get timeout from settings if not provided
    if timeout is None:
        timeout = get_settings().index_timeout

    # Create index status
    index_status = IndexStatus(index_id=index_id, status="running")
    _set_active_index(index_status)

    async def run_indexing():
        import time as time_module
        index_start = time_module.time()
        logger.info(f"[RAG] Starting async indexing {index_id} with timeout={timeout}s")

        rag = get_rag_engine()

        try:
            # Wrap the indexing in a timeout
            result = await asyncio.wait_for(
                rag.index_documents_async(
                    progress_callback=lambda processed, total, current_file, chunks: _update_index_progress(
                        index_status, processed, total, current_file, chunks
                    )
                ),
                timeout=timeout,
            )

            index_status.status = "completed"
            index_status.total_documents = result.total_documents
            index_status.total_chunks = result.total_chunks
            index_status.processed_documents = result.total_documents
            index_status.completed_at = datetime.now()

            elapsed = time_module.time() - index_start
            logger.info(f"[RAG] Indexing completed: {index_id} in {elapsed:.1f}s - {result.total_documents} docs, {result.total_chunks} chunks")

            # Audit log
            audit = get_audit_logger()
            audit.log_admin_action(
                "index_documents_async",
                {
                    "index_id": index_id,
                    "total_documents": result.total_documents,
                    "total_chunks": result.total_chunks,
                    "elapsed_seconds": round(elapsed, 1),
                },
            )

        except asyncio.CancelledError:
            elapsed = time_module.time() - index_start
            index_status.status = "cancelled"
            index_status.error = "Indexing was cancelled"
            index_status.completed_at = datetime.now()
            logger.info(f"[RAG] Indexing cancelled: {index_id} after {elapsed:.1f}s")

            # Clear partial index on cancellation
            try:
                rag.clear_index()
                logger.info(f"[RAG] Cleared partial index after cancellation")
            except Exception as e:
                logger.error(f"[RAG] Failed to clear partial index: {e}")

        except asyncio.TimeoutError:
            elapsed = time_module.time() - index_start
            index_status.status = "error"
            index_status.error = f"Indexing timed out after {timeout} seconds"
            index_status.completed_at = datetime.now()
            logger.error(f"[RAG] Indexing timed out: {index_id} after {elapsed:.1f}s (limit was {timeout}s)")

            # Clear partial index on timeout
            try:
                rag.clear_index()
                logger.info(f"[RAG] Cleared partial index after timeout")
            except Exception as e:
                logger.error(f"[RAG] Failed to clear partial index: {e}")

            # Audit log timeout
            audit = get_audit_logger()
            audit.log_admin_action(
                "index_documents_async",
                {
                    "index_id": index_id,
                    "error": "timeout",
                    "elapsed_seconds": round(elapsed, 1),
                },
            )

        except Exception as e:
            elapsed = time_module.time() - index_start
            index_status.status = "error"
            index_status.error = str(e)
            index_status.completed_at = datetime.now()
            logger.error(f"[RAG] Indexing failed: {index_id} after {elapsed:.1f}s - {type(e).__name__}: {e}")

            # Clear partial index on error
            try:
                rag.clear_index()
                logger.info(f"[RAG] Cleared partial index after error")
            except Exception as clear_e:
                logger.error(f"[RAG] Failed to clear partial index: {clear_e}")

            # Audit log error
            audit = get_audit_logger()
            audit.log_admin_action(
                "index_documents_async",
                {
                    "index_id": index_id,
                    "error": str(e),
                    "elapsed_seconds": round(elapsed, 1),
                },
            )

    _index_task = asyncio.create_task(run_indexing())
    return index_status


def _update_index_progress(
    status: IndexStatus,
    processed: int,
    total: int,
    current_file: str | None,
    chunks: int,
) -> None:
    """Update the indexing progress status."""
    status.processed_documents = processed
    status.total_documents = total
    status.current_file = current_file
    status.total_chunks = chunks


class RAGEngine:
    """RAG engine with ChromaDB vector store and sentence-transformers embeddings."""

    COLLECTION_NAME = "pika_documents"

    def __init__(
        self,
        settings: Settings | None = None,
        document_processor: DocumentProcessor | None = None,
        ollama_client: OllamaClient | None = None,
    ):
        self.settings = settings or get_settings()
        self.document_processor = document_processor or get_document_processor()
        self.ollama_client = ollama_client or get_ollama_client()

        # Initialize embedding model
        self._embedding_model: SentenceTransformer | None = None

        # Initialize ChromaDB with error handling
        persist_dir = Path(self.settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(
                f"Failed to initialize vector database at {persist_dir}. "
                f"Check permissions and disk space. Error: {e}"
            ) from e

        self._collection = None

        # Cache for stats and indexed documents (thread-safe)
        self._cache_lock = Lock()
        self._stats_cache: IndexStats | None = None
        self._stats_cache_time: float = 0
        self._docs_cache: list[IndexedDocument] | None = None
        self._docs_cache_time: float = 0

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._embedding_model is None:
            self._load_embedding_model()
        return self._embedding_model

    def _load_embedding_model(self) -> None:
        """Load the embedding model (can be called during startup or lazily)."""
        if self._embedding_model is not None:
            return  # Already loaded

        import time as time_module
        load_start = time_module.time()
        logger.info(f"[RAG] Loading embedding model: {self.settings.embedding_model}...")
        try:
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
            load_elapsed = time_module.time() - load_start
            logger.info(f"[RAG] Embedding model loaded in {load_elapsed:.1f}s")
        except Exception as e:
            logger.error(f"[RAG] Failed to load embedding model: {e}")
            raise RuntimeError(
                f"Failed to load embedding model '{self.settings.embedding_model}'. "
                f"Error: {e}"
            ) from e

    def preload(self) -> None:
        """Pre-load the embedding model. Call during startup to avoid first-query latency."""
        self._load_embedding_model()

    @property
    def collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            try:
                self._collection = self.chroma_client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                error_str = str(e).lower()
                if "readonly" in error_str or "read-only" in error_str or "code: 1032" in error_str:
                    import gc
                    logger.warning("[RAG] ChromaDB readonly/moved error - attempting recovery")

                    # Release current client before recovery to free file handles
                    self.chroma_client = None
                    gc.collect()

                    # First try: fix permissions (includes journal cleanup) and reinitialize
                    self._fix_chroma_permissions()
                    self._reinitialize_chroma()

                    try:
                        self._collection = self.chroma_client.get_or_create_collection(
                            name=self.COLLECTION_NAME,
                            metadata={"hnsw:space": "cosine"},
                        )
                    except Exception as retry_error:
                        # ChromaDB's Rust bindings cache state globally - clearing files isn't enough
                        # Clear everything and start fresh (user will need to re-index)
                        logger.warning(f"[RAG] Permission fix failed ({retry_error}), clearing ChromaDB")
                        self._clear_and_reinitialize_chroma()
                        self._collection = self.chroma_client.get_or_create_collection(
                            name=self.COLLECTION_NAME,
                            metadata={"hnsw:space": "cosine"},
                        )
                        logger.info("[RAG] ChromaDB recovered - re-indexing required")
                else:
                    raise
        return self._collection

    def _fix_chroma_permissions(self) -> None:
        """Fix ChromaDB file permissions (e.g., after restore from backup)."""
        import os

        persist_dir = Path(self.settings.chroma_persist_dir)
        if not persist_dir.exists():
            return

        logger.info(f"[RAG] Fixing permissions in {persist_dir}")
        try:
            # Fix the directory itself
            persist_dir.chmod(0o777)

            # Delete SQLite journal files first - these can cause "readonly database" errors
            # if they contain stale state from a backup or different environment
            journal_extensions = ["-wal", "-shm", "-journal"]
            journals_deleted = 0
            for root, dirs, files in os.walk(persist_dir):
                for f in files:
                    if any(f.endswith(ext) for ext in journal_extensions):
                        journal_path = Path(root) / f
                        try:
                            journal_path.unlink()
                            journals_deleted += 1
                            logger.debug(f"[RAG] Deleted journal file: {journal_path}")
                        except Exception as e:
                            logger.debug(f"[RAG] Could not delete journal file {f}: {e}")
            if journals_deleted > 0:
                logger.info(f"[RAG] Deleted {journals_deleted} SQLite journal file(s)")

            # Fix all subdirectories and files
            for root, dirs, files in os.walk(persist_dir):
                for d in dirs:
                    try:
                        (Path(root) / d).chmod(0o777)
                    except Exception as e:
                        logger.debug(f"[RAG] Could not fix permissions on dir {d}: {e}")
                for f in files:
                    try:
                        (Path(root) / f).chmod(0o666)
                    except Exception as e:
                        logger.debug(f"[RAG] Could not fix permissions on file {f}: {e}")

            logger.info("[RAG] ChromaDB permissions fixed")
        except Exception as e:
            logger.error(f"[RAG] Failed to fix permissions: {e}")

    def _clear_and_reinitialize_chroma(self) -> None:
        """Clear all ChromaDB data and reinitialize (nuclear option for recovery)."""
        import gc
        import shutil

        persist_dir = Path(self.settings.chroma_persist_dir)
        logger.warning(f"[RAG] Clearing ChromaDB data at {persist_dir}")

        # Release current client
        self.chroma_client = None
        self._collection = None
        gc.collect()

        # Delete all contents of the chroma directory
        if persist_dir.exists():
            for item in persist_dir.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                except Exception as e:
                    logger.error(f"[RAG] Failed to delete {item}: {e}")

        # Ensure directory exists with correct permissions
        persist_dir.mkdir(parents=True, exist_ok=True)
        persist_dir.chmod(0o777)

        # Create fresh ChromaDB client
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("[RAG] ChromaDB cleared and reinitialized")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def _get_confidence(self, similarities: list[float]) -> Confidence:
        """Determine confidence level based on similarity scores."""
        if not similarities:
            return Confidence.NONE

        max_similarity = max(similarities)

        if max_similarity >= self.settings.confidence_high:
            return Confidence.HIGH
        elif max_similarity >= self.settings.confidence_medium:
            return Confidence.MEDIUM
        elif max_similarity >= self.settings.confidence_low:
            return Confidence.LOW
        else:
            return Confidence.NONE

    def index_documents(self) -> IndexStats:
        """Index all documents from the documents directory."""
        # Clear existing index and invalidate query cache
        self.clear_index()
        invalidate_query_cache()

        # Process all documents
        chunks = self.document_processor.process_all_documents()

        if not chunks:
            stats = IndexStats(
                total_documents=0,
                total_chunks=0,
                collection_name=self.COLLECTION_NAME,
            )
            self._update_cache(stats, [])
            return stats

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        # Track chunks per source for cache
        source_chunks: dict[str, int] = {}

        for i, chunk in enumerate(chunks):
            ids.append(f"chunk_{i}")
            documents.append(chunk.content)
            metadatas.append({
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
            })
            source_chunks[chunk.source] = source_chunks.get(chunk.source, 0) + 1

        # Generate embeddings
        embeddings = self._embed(documents)

        # Add to collection in batches to avoid memory/size issues
        batch_size = 1000
        total_items = len(ids)
        for i in range(0, total_items, batch_size):
            end = min(i + batch_size, total_items)
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
            logger.debug(f"[RAG] Added batch {i//batch_size + 1}/{(total_items + batch_size - 1)//batch_size} to ChromaDB")

        stats = IndexStats(
            total_documents=len(source_chunks),
            total_chunks=len(chunks),
            collection_name=self.COLLECTION_NAME,
        )

        # Build and cache the indexed documents list
        indexed_docs = [
            IndexedDocument(filename=filename, chunk_count=count)
            for filename, count in sorted(source_chunks.items())
        ]
        self._update_cache(stats, indexed_docs)

        return stats

    async def index_documents_async(
        self,
        progress_callback: Callable[[int, int, str | None, int], None] | None = None,
    ) -> IndexStats:
        """Index all documents from the documents directory with progress reporting.

        Args:
            progress_callback: Optional callback(processed, total, current_file, chunks)
                              called after each document is processed

        Returns:
            IndexStats with indexing results
        """
        # Clear existing index and invalidate query cache
        self.clear_index()
        invalidate_query_cache()

        # Get list of documents using the document processor (consistent with sync version)
        doc_list = self.document_processor.list_documents()
        total_docs = len(doc_list)

        if total_docs == 0:
            if progress_callback:
                progress_callback(0, 0, None, 0)
            stats = IndexStats(
                total_documents=0,
                total_chunks=0,
                collection_name=self.COLLECTION_NAME,
            )
            self._update_cache(stats, [])
            return stats

        # Process documents one by one with progress reporting
        all_chunks = []
        source_chunks: dict[str, int] = {}  # Track chunks per source for cache
        loop = asyncio.get_running_loop()

        for i, doc_info in enumerate(doc_list):
            # Report progress before processing
            if progress_callback:
                progress_callback(i, total_docs, doc_info.filename, len(all_chunks))

            # Allow other async tasks to run (makes cancellation responsive)
            await asyncio.sleep(0)

            try:
                # Process document in thread pool (CPU-intensive, don't block event loop)
                chunks = await loop.run_in_executor(
                    None, self.document_processor.process_document, doc_info.path
                )
                all_chunks.extend(chunks)
                source_chunks[doc_info.filename] = len(chunks)
            except FileTooLargeError as e:
                # File exceeds size limit - skip with clear message
                logger.warning(f"[RAG] Skipping {doc_info.filename}: {e.size_mb:.1f}MB exceeds {e.max_mb}MB limit")
            except Exception as e:
                # Log warning but continue with other documents
                logger.warning(f"[RAG] Failed to process {doc_info.filename}: {e}")

            # Clean up memory after each document (helps with large files)
            import gc
            gc.collect()

        # Final progress update
        if progress_callback:
            progress_callback(total_docs, total_docs, None, len(all_chunks))

        if not all_chunks:
            stats = IndexStats(
                total_documents=0,
                total_chunks=0,
                collection_name=self.COLLECTION_NAME,
            )
            self._update_cache(stats, [])
            return stats

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(all_chunks):
            ids.append(f"chunk_{i}")
            documents.append(chunk.content)
            metadatas.append({
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
            })

        # Generate embeddings in thread pool (CPU-intensive, don't block event loop)
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, self._embed, documents)

        # Add to collection in batches to avoid memory/size issues
        batch_size = 1000
        total_items = len(ids)
        for i in range(0, total_items, batch_size):
            end = min(i + batch_size, total_items)
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
            logger.debug(f"[RAG] Added batch {i//batch_size + 1}/{(total_items + batch_size - 1)//batch_size} to ChromaDB")
            # Allow other async tasks to run between batches
            await asyncio.sleep(0)

        stats = IndexStats(
            total_documents=len(source_chunks),
            total_chunks=len(all_chunks),
            collection_name=self.COLLECTION_NAME,
        )

        # Build and cache the indexed documents list
        indexed_docs = [
            IndexedDocument(filename=filename, chunk_count=count)
            for filename, count in sorted(source_chunks.items())
        ]
        self._update_cache(stats, indexed_docs)

        return stats

    def invalidate_cache(self) -> None:
        """Invalidate all cached data. Call after index changes."""
        with self._cache_lock:
            self._stats_cache = None
            self._stats_cache_time = 0
            self._docs_cache = None
            self._docs_cache_time = 0

    def _update_cache(self, stats: IndexStats, docs: list[IndexedDocument] | None = None) -> None:
        """Update the cache with fresh data after indexing."""
        with self._cache_lock:
            self._stats_cache = stats
            self._stats_cache_time = time.time()
            if docs is not None:
                self._docs_cache = docs
                self._docs_cache_time = time.time()

    def get_stats(self) -> IndexStats:
        """Get current index statistics (cached for performance)."""
        # Check cache first
        with self._cache_lock:
            if self._stats_cache and (time.time() - self._stats_cache_time) < STATS_CACHE_TTL:
                return self._stats_cache

        # Cache miss - compute stats
        count = self.collection.count()

        # Get unique sources - only if there are documents
        if count > 0:
            results = self.collection.get(include=["metadatas"])
            sources = set(m["source"] for m in results["metadatas"])
            doc_count = len(sources)
        else:
            doc_count = 0

        stats = IndexStats(
            total_documents=doc_count,
            total_chunks=count,
            collection_name=self.COLLECTION_NAME,
        )

        # Update cache
        with self._cache_lock:
            self._stats_cache = stats
            self._stats_cache_time = time.time()

        return stats

    def get_indexed_documents(self) -> list[IndexedDocument]:
        """Get list of indexed documents with their chunk counts (cached for performance)."""
        # Check cache first
        with self._cache_lock:
            if self._docs_cache is not None and (time.time() - self._docs_cache_time) < DOCS_CACHE_TTL:
                return self._docs_cache

        # Cache miss - compute document list
        if self.collection.count() == 0:
            docs = []
        else:
            results = self.collection.get(include=["metadatas"])

            # Count chunks per document
            chunk_counts: dict[str, int] = {}
            for metadata in results["metadatas"]:
                source = metadata["source"]
                chunk_counts[source] = chunk_counts.get(source, 0) + 1

            # Sort by filename
            docs = [
                IndexedDocument(filename=filename, chunk_count=count)
                for filename, count in sorted(chunk_counts.items())
            ]

        # Update cache
        with self._cache_lock:
            self._docs_cache = docs
            self._docs_cache_time = time.time()

        return docs

    def clear_index(self) -> None:
        """Clear all documents from the index."""
        try:
            self.chroma_client.delete_collection(self.COLLECTION_NAME)
            self._collection = None
        except ValueError:
            # Collection doesn't exist
            pass
        except Exception as e:
            # ChromaDB might be in a bad state (e.g., after restore)
            # Try to reinitialize the client
            logger.warning(f"[RAG] Error clearing index, reinitializing ChromaDB: {e}")
            try:
                self._reinitialize_chroma()
                # Try delete again after reinit
                try:
                    self.chroma_client.delete_collection(self.COLLECTION_NAME)
                except ValueError:
                    pass  # Collection doesn't exist
            except Exception as reinit_error:
                logger.error(f"[RAG] Failed to reinitialize ChromaDB: {reinit_error}")
                # Continue anyway - collection will be recreated
            self._collection = None
        # Invalidate cache
        self.invalidate_cache()

    def _reinitialize_chroma(self) -> None:
        """Reinitialize the ChromaDB client (e.g., after restore)."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_dir = Path(self.settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = None
        logger.info("[RAG] ChromaDB client reinitialized")

    def shutdown(self) -> None:
        """Shutdown the RAG engine and release ChromaDB resources.

        This must be called before restoring ChromaDB files to ensure
        SQLite connections are properly closed.
        """
        import gc

        logger.info("[RAG] Shutting down RAG engine...")

        # Clear collection reference
        self._collection = None

        # Clear ChromaDB client reference - this should release SQLite connections
        if hasattr(self, 'chroma_client') and self.chroma_client is not None:
            # ChromaDB doesn't have an explicit close method, but clearing
            # the reference and forcing GC should release the SQLite connection
            self.chroma_client = None

        # Clear embedding model to free memory
        self._embedding_model = None

        # Clear caches
        self._stats_cache = None
        self._docs_cache = None

        # Force garbage collection to ensure SQLite connections are released
        gc.collect()

        logger.info("[RAG] RAG engine shutdown complete")

    async def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> QueryResult:
        """Query the RAG system with a question."""
        import time as time_module
        from pika.services.metrics import QUERY_COUNT, QUERY_LATENCY

        query_start = time_module.time()
        logger.info(f"[RAG.query] Starting query: '{question[:50]}...'")

        top_k = top_k or self.settings.top_k

        # Check if index has documents
        try:
            doc_count = self.collection.count()
        except Exception:
            doc_count = 0

        if doc_count == 0:
            QUERY_COUNT.labels(status="no_documents", confidence="none").inc()
            return QueryResult(
                answer="No documents indexed yet. Upload documents and click Refresh Index.",
                sources=[],
                confidence=Confidence.NONE,
            )

        # Check query cache if enabled
        if self.settings.query_cache_enabled:
            stats = self.get_stats()
            cached_result = get_query_cache().get(question, stats.total_documents, stats.total_chunks)
            if cached_result is not None:
                logger.info("[RAG.query] Cache hit, returning cached result")
                QUERY_COUNT.labels(status="cache_hit", confidence=cached_result.confidence.value).inc()
                return cached_result

        # Generate embedding for question
        embed_start = time_module.time()
        question_embedding = self._embed([question])[0]
        embed_elapsed = time_module.time() - embed_start
        logger.info(f"[RAG.query] Embedding generated in {embed_elapsed:.2f}s")

        # Query ChromaDB
        chroma_start = time_module.time()
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chroma_elapsed = time_module.time() - chroma_start
        logger.info(f"[RAG.query] ChromaDB query completed in {chroma_elapsed:.2f}s")

        # Convert distances to similarities (cosine distance to similarity)
        distances = results["distances"][0]
        similarities = [1 - d for d in distances]

        # Build sources
        sources = []
        for i, (doc, metadata, similarity) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                similarities,
            )
        ):
            sources.append(
                Source(
                    filename=metadata["source"],
                    chunk_index=metadata["chunk_index"],
                    content=doc,
                    similarity=similarity,
                )
            )

        # Determine confidence
        confidence = self._get_confidence(similarities)

        # Generate answer based on confidence
        if confidence == Confidence.NONE:
            answer = (
                "I couldn't find relevant information in the indexed documents "
                "to answer your question. Please try rephrasing or ensure the "
                "relevant documents are indexed."
            )
        else:
            # Build context from retrieved chunks
            context_parts = []
            for source in sources:
                context_parts.append(
                    f"[From {source.filename}]:\n{source.content}"
                )
            context = "\n\n---\n\n".join(context_parts)

            # Generate answer using Ollama
            system_prompt = (
                "You are a helpful assistant answering questions based on the provided context. "
                "Use only the information from the context to answer. "
                "If the context doesn't contain enough information, say so. "
                "Be concise and cite which document(s) your answer comes from."
            )

            prompt = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""

            try:
                ollama_start = time_module.time()
                logger.info(f"[RAG.query] Calling Ollama generate (prompt_len={len(prompt)}, system_len={len(system_prompt)})")
                answer = await self.ollama_client.generate(
                    prompt=prompt,
                    system=system_prompt,
                )
                ollama_elapsed = time_module.time() - ollama_start
                total_elapsed = time_module.time() - query_start
                logger.info(f"[RAG.query] Ollama completed in {ollama_elapsed:.1f}s, total query time: {total_elapsed:.1f}s")
            except OllamaCircuitOpenError:
                # Degraded mode: return search results without LLM answer
                logger.info("[RAG.query] Circuit breaker open, returning degraded response")
                QUERY_COUNT.labels(status="degraded", confidence="degraded").inc()
                duration = time_module.time() - query_start
                QUERY_LATENCY.observe(duration)
                return QueryResult(
                    answer=_format_degraded_response(sources),
                    sources=sources,
                    confidence=Confidence.DEGRADED,
                )
            except OllamaConnectionError:
                # Degraded mode: return search results without LLM answer
                logger.info("[RAG.query] Ollama connection error, returning degraded response")
                QUERY_COUNT.labels(status="degraded", confidence="degraded").inc()
                duration = time_module.time() - query_start
                QUERY_LATENCY.observe(duration)
                return QueryResult(
                    answer=_format_degraded_response(sources),
                    sources=sources,
                    confidence=Confidence.DEGRADED,
                )
            except OllamaModelNotFoundError as e:
                QUERY_COUNT.labels(status="ollama_model_not_found", confidence="none").inc()
                return QueryResult(
                    answer=(
                        f"The model '{e.model}' is not available. "
                        "Please go to Admin to pull the model or select a different one."
                    ),
                    sources=sources,
                    confidence=Confidence.NONE,
                )
            except OllamaTimeoutError:
                # Degraded mode: return search results without LLM answer
                logger.info("[RAG.query] Ollama timeout, returning degraded response")
                QUERY_COUNT.labels(status="degraded", confidence="degraded").inc()
                duration = time_module.time() - query_start
                QUERY_LATENCY.observe(duration)
                return QueryResult(
                    answer=_format_degraded_response(sources),
                    sources=sources,
                    confidence=Confidence.DEGRADED,
                )

        # Record metrics
        duration = time_module.time() - query_start
        QUERY_LATENCY.observe(duration)
        QUERY_COUNT.labels(status="success", confidence=confidence.value).inc()

        result = QueryResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
        )

        # Cache the result if caching is enabled and it's a successful response
        if self.settings.query_cache_enabled and confidence != Confidence.NONE:
            stats = self.get_stats()
            get_query_cache().set(question, stats.total_documents, stats.total_chunks, result)

        return result

    async def query_stream(
        self,
        question: str,
        top_k: int | None = None,
    ) -> AsyncIterator[dict]:
        """Query the RAG system with streaming response.

        Yields JSON-serializable dicts:
        - {"type": "metadata", "sources": [...], "confidence": "..."}
        - {"type": "token", "content": "..."}
        - {"type": "done", "answer": "..."}
        - {"type": "error", "message": "..."}
        """
        import time as time_module

        query_start = time_module.time()

        if top_k is None:
            top_k = self.settings.top_k

        # Check if index has documents
        try:
            doc_count = self.collection.count()
        except Exception:
            doc_count = 0

        if doc_count == 0:
            yield {"type": "metadata", "sources": [], "confidence": "none"}
            yield {"type": "token", "content": "No documents indexed yet. Upload documents and click Refresh Index."}
            yield {"type": "done", "answer": "No documents indexed yet. Upload documents and click Refresh Index."}
            return

        # Get embedding for the question
        try:
            logger.info(f"[RAG.query_stream] Starting query: '{question[:50]}...' top_k={top_k}")
            embed_start = time_module.time()
            question_embedding = self._embed([question])[0]
            embed_elapsed = time_module.time() - embed_start
            logger.info(f"[RAG.query_stream] Embedding completed in {embed_elapsed:.2f}s")
        except Exception as e:
            logger.error(f"[RAG.query_stream] Failed to get embedding: {e}")
            yield {"type": "error", "message": "Failed to process question"}
            return

        # Query ChromaDB
        chroma_start = time_module.time()
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chroma_elapsed = time_module.time() - chroma_start
        logger.info(f"[RAG.query_stream] ChromaDB query completed in {chroma_elapsed:.2f}s")

        # Convert distances to similarities
        distances = results["distances"][0]
        similarities = [1 - d for d in distances]

        # Build sources
        sources = []
        for i, (doc, metadata, similarity) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                similarities,
            )
        ):
            sources.append(
                Source(
                    filename=metadata["source"],
                    chunk_index=metadata["chunk_index"],
                    content=doc,
                    similarity=similarity,
                )
            )

        # Determine confidence
        confidence = self._get_confidence(similarities)

        # Yield metadata first so UI can show sources while streaming
        from dataclasses import asdict
        yield {
            "type": "metadata",
            "sources": [asdict(s) for s in sources],
            "confidence": confidence.value,
        }

        # Generate answer based on confidence
        if confidence == Confidence.NONE:
            answer = (
                "I couldn't find relevant information in the indexed documents "
                "to answer your question. Please try rephrasing or ensure the "
                "relevant documents are indexed."
            )
            yield {"type": "token", "content": answer}
            yield {"type": "done", "answer": answer}
            return

        # Build context from retrieved chunks
        context_parts = []
        for source in sources:
            context_parts.append(f"[From {source.filename}]:\n{source.content}")
        context = "\n\n---\n\n".join(context_parts)

        # Generate streaming answer using Ollama
        system_prompt = (
            "You are a helpful assistant answering questions based on the provided context. "
            "Use only the information from the context to answer. "
            "If the context doesn't contain enough information, say so. "
            "Be concise and cite which document(s) your answer comes from."
        )

        prompt = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""

        try:
            ollama_start = time_module.time()
            logger.info(f"[RAG.query_stream] Starting Ollama stream (prompt_len={len(prompt)})")

            full_answer = ""
            async for token in self.ollama_client.generate_stream(
                prompt=prompt,
                system=system_prompt,
            ):
                full_answer += token
                yield {"type": "token", "content": token}

            ollama_elapsed = time_module.time() - ollama_start
            total_elapsed = time_module.time() - query_start
            logger.info(f"[RAG.query_stream] Completed in {total_elapsed:.1f}s (Ollama: {ollama_elapsed:.1f}s)")

            # Log to history and audit (same as _execute_query)
            try:
                from pika.services.audit import get_audit_logger
                from pika.services.app_config import get_app_config
                from pika.services.history import get_history_service

                # Audit log
                audit = get_audit_logger()
                audit.log_query(
                    question=question,
                    model=get_app_config().get_current_model(),
                    confidence=confidence.value,
                    sources=[s.filename for s in sources],
                )

                # Save to history (username will be None for streaming - that's OK)
                history = get_history_service()
                history.add_query(
                    question=question,
                    answer=full_answer,
                    confidence=confidence.value,
                    sources=[s.filename for s in sources],
                    username=None,  # Streaming doesn't track user
                )
            except Exception as log_error:
                logger.warning(f"[RAG.query_stream] Failed to log query: {log_error}")

            yield {"type": "done", "answer": full_answer}

        except OllamaCircuitOpenError:
            # Degraded mode: return search results without streaming
            logger.info("[RAG.query_stream] Circuit breaker open, returning degraded response")
            degraded_answer = _format_degraded_response(sources)
            yield {"type": "metadata", "sources": [asdict(s) for s in sources], "confidence": "degraded"}
            yield {"type": "token", "content": degraded_answer}
            yield {"type": "done", "answer": degraded_answer}
        except OllamaConnectionError:
            # Degraded mode: return search results without streaming
            logger.info("[RAG.query_stream] Ollama connection error, returning degraded response")
            degraded_answer = _format_degraded_response(sources)
            yield {"type": "metadata", "sources": [asdict(s) for s in sources], "confidence": "degraded"}
            yield {"type": "token", "content": degraded_answer}
            yield {"type": "done", "answer": degraded_answer}
        except OllamaModelNotFoundError as e:
            yield {
                "type": "error",
                "message": f"The model '{e.model}' is not available. Please go to Admin to pull or select a model.",
            }
        except OllamaTimeoutError:
            # Degraded mode: return search results without streaming
            logger.info("[RAG.query_stream] Ollama timeout, returning degraded response")
            degraded_answer = _format_degraded_response(sources)
            yield {"type": "metadata", "sources": [asdict(s) for s in sources], "confidence": "degraded"}
            yield {"type": "token", "content": degraded_answer}
            yield {"type": "done", "answer": degraded_answer}
        except Exception as e:
            logger.error(f"[RAG.query_stream] Unexpected error: {e}")
            yield {"type": "error", "message": "An unexpected error occurred"}


# Singleton instance
_rag_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    """Get or create the RAG engine singleton."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


def prepare_for_restore() -> None:
    """Prepare for restore by shutting down the RAG engine.

    This must be called before restoring ChromaDB files to ensure
    SQLite connections are properly closed. After restore, the engine
    will be automatically recreated on next access.
    """
    global _rag_engine
    if _rag_engine is not None:
        _rag_engine.shutdown()
        _rag_engine = None
        logger.info("[RAG] Engine prepared for restore - singleton cleared")


async def preload_embedding_model() -> None:
    """Pre-load the embedding model during app startup.

    This avoids latency on the first query by loading the model in advance.
    Runs in a thread pool to avoid blocking the event loop.
    """
    import asyncio

    def _load():
        try:
            engine = get_rag_engine()
            engine.preload()
        except Exception as e:
            logger.error(f"Failed to preload embedding model: {e}")
            # Don't raise - app can still start, model will load on first query

    await asyncio.get_event_loop().run_in_executor(None, _load)


def is_query_running(username: str | None = None) -> bool:
    """Check if a query task is currently running for a user."""
    key = _get_user_key(username)
    task = _query_tasks.get(key)
    return task is not None and not task.done()


async def cancel_query(username: str | None = None) -> bool:
    """Cancel the currently running or queued query for a user.

    Returns True if a query was cancelled, False if no query was found.
    """
    key = _get_user_key(username)
    query = _active_queries.get(key)

    if query is None:
        return False

    # Check if it's queued
    if query.status == "queued":
        # Remove from queue
        removed = remove_from_queue(query.query_id)
        if removed:
            query.status = "cancelled"
            query.error = "Query was cancelled by user"
            logger.info(f"Queued query cancelled by user: {username or 'anonymous'}")
            return True

    # Check for running task
    task = _query_tasks.get(key)
    if task is not None and not task.done():
        task.cancel()
        query.status = "cancelled"
        query.error = "Query was cancelled by user"
        logger.info(f"Running query cancelled by user: {username or 'anonymous'}")
        return True

    # Also check if query is tracked as running in queue system
    async with _queue_lock:
        if query.query_id in _running_queries:
            query.status = "cancelled"
            query.error = "Query was cancelled by user"
            _running_queries.discard(query.query_id)
            logger.info(f"Query cancelled by user: {username or 'anonymous'}")
            return True

    return False


class QueueFullError(Exception):
    """Raised when the query queue is full."""

    pass


class UserQueueLimitError(Exception):
    """Raised when user has too many queries in queue."""

    pass


async def start_query_task(
    question: str,
    query_id: str,
    top_k: int | None = None,
    username: str | None = None,
) -> QueryStatus:
    """Start or queue a query for a specific user.

    If slots are available, the query runs immediately.
    Otherwise it's queued with position tracking.

    Raises:
        QueueFullError: If the global queue is full
        UserQueueLimitError: If user has too many pending queries
    """
    settings = get_settings()

    async with _queue_lock:
        # Check global queue limit
        if len(_query_queue) >= settings.max_queue_size:
            raise QueueFullError(f"Queue is full ({settings.max_queue_size} queries)")

        # Check per-user queue limit
        user_queued = get_user_queued_count(username)
        if user_queued >= settings.max_queued_per_user:
            raise UserQueueLimitError(
                f"Too many pending queries ({settings.max_queued_per_user} max per user)"
            )

        # Check if we can run immediately
        if len(_running_queries) < settings.max_concurrent_queries:
            # Run immediately
            query_status = QueryStatus(query_id=query_id, question=question, status="running")
            _set_query_status(query_status, username)

            # Track as running
            _running_queries.add(query_id)

            # Create queued query object for execution
            queued = QueuedQuery(
                query_id=query_id,
                question=question,
                username=username,
                top_k=top_k,
            )

            # Execute in background
            async def run_and_cleanup():
                try:
                    await _execute_query(queued)
                finally:
                    async with _queue_lock:
                        _running_queries.discard(query_id)

            asyncio.create_task(run_and_cleanup())
            logger.info(f"[Queue] Query {query_id} started immediately (slots: {len(_running_queries)}/{settings.max_concurrent_queries})")

            return query_status
        else:
            # Queue the query
            queued = QueuedQuery(
                query_id=query_id,
                question=question,
                username=username,
                top_k=top_k,
            )
            _query_queue.append(queued)

            # Calculate initial position
            position = len(_query_queue)
            avg_duration = _queue_stats.get_average_duration()
            estimated_wait = int(position * avg_duration / settings.max_concurrent_queries)

            query_status = QueryStatus(
                query_id=query_id,
                question=question,
                status="queued",
                queue_position=position,
                queue_length=position,
                estimated_wait_seconds=estimated_wait,
            )
            _set_query_status(query_status, username)

            logger.info(f"[Queue] Query {query_id} queued at position {position} (estimated wait: {estimated_wait}s)")

            return query_status
