"""RAG engine using ChromaDB and sentence-transformers."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Callable

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from pika.config import Settings, get_settings
from pika.services.documents import DocumentProcessor, get_document_processor
from pika.services.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    get_ollama_client,
)

logger = logging.getLogger(__name__)

# Cache TTL in seconds
STATS_CACHE_TTL = 60  # 1 minute cache for stats
DOCS_CACHE_TTL = 60   # 1 minute cache for indexed documents list


class Confidence(str, Enum):
    """Confidence level for query results."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


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
class QueryStatus:
    """Status of an active or completed query."""

    query_id: str
    question: str
    status: str = "pending"  # pending, running, completed, error
    result: "QueryResult | None" = None
    error: str | None = None
    started_at: datetime = field(default_factory=datetime.now)

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
        }


# Per-user query status tracking
_active_queries: dict[str, QueryStatus] = {}
_query_tasks: dict[str, asyncio.Task] = {}

# Key for anonymous users
ANONYMOUS_USER = "__anonymous__"


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

        # Initialize ChromaDB
        persist_dir = Path(self.settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

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
            import time as time_module
            load_start = time_module.time()
            logger.info(f"[RAG] Loading embedding model: {self.settings.embedding_model}...")
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
            load_elapsed = time_module.time() - load_start
            logger.info(f"[RAG] Embedding model loaded in {load_elapsed:.1f}s")
        return self._embedding_model

    @property
    def collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            self._collection = self.chroma_client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

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
        # Clear existing index
        self.clear_index()

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
        # Clear existing index
        self.clear_index()

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

        for i, doc_info in enumerate(doc_list):
            # Report progress before processing
            if progress_callback:
                progress_callback(i, total_docs, doc_info.filename, len(all_chunks))

            # Allow other async tasks to run (makes cancellation responsive)
            await asyncio.sleep(0)

            try:
                # Process this document
                chunks = self.document_processor.process_document(doc_info.path)
                all_chunks.extend(chunks)
                source_chunks[doc_info.filename] = len(chunks)
            except Exception as e:
                # Log warning but continue with other documents
                logger.warning(f"[RAG] Failed to process {doc_info.filename}: {e}")
                continue

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

        # Generate embeddings (this can be slow)
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
        # Invalidate cache
        self.invalidate_cache()

    async def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> QueryResult:
        """Query the RAG system with a question."""
        import time as time_module
        query_start = time_module.time()
        logger.info(f"[RAG.query] Starting query: '{question[:50]}...'")

        top_k = top_k or self.settings.top_k

        # Check if index has documents
        try:
            doc_count = self.collection.count()
        except Exception:
            doc_count = 0

        if doc_count == 0:
            return QueryResult(
                answer="No documents indexed yet. Upload documents and click Refresh Index.",
                sources=[],
                confidence=Confidence.NONE,
            )

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
            except OllamaConnectionError:
                return QueryResult(
                    answer=(
                        "Unable to connect to the AI model service (Ollama). "
                        "Please check that Ollama is running and try again."
                    ),
                    sources=sources,
                    confidence=Confidence.NONE,
                )
            except OllamaModelNotFoundError as e:
                return QueryResult(
                    answer=(
                        f"The model '{e.model}' is not available. "
                        "Please go to Admin to pull the model or select a different one."
                    ),
                    sources=sources,
                    confidence=Confidence.NONE,
                )
            except OllamaTimeoutError:
                return QueryResult(
                    answer=(
                        "The request took too long and timed out. "
                        "Please try a shorter or simpler question, or try again later."
                    ),
                    sources=sources,
                    confidence=Confidence.NONE,
                )

        return QueryResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
        )


# Singleton instance
_rag_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    """Get or create the RAG engine singleton."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


def is_query_running(username: str | None = None) -> bool:
    """Check if a query task is currently running for a user."""
    key = _get_user_key(username)
    task = _query_tasks.get(key)
    return task is not None and not task.done()


def cancel_query(username: str | None = None) -> bool:
    """Cancel the currently running query task for a user.

    Returns True if a query was cancelled, False if no query was running.
    """
    key = _get_user_key(username)
    task = _query_tasks.get(key)
    query = _active_queries.get(key)

    if task is not None and not task.done():
        task.cancel()
        if query:
            query.status = "cancelled"
            query.error = "Query was cancelled by user"
        logger.info(f"Query cancelled by user: {username or 'anonymous'}")
        return True
    return False


# Query timeout in seconds (default 5 minutes)
QUERY_TIMEOUT = 300


async def start_query_task(
    question: str,
    query_id: str,
    top_k: int | None = None,
    username: str | None = None,
) -> QueryStatus:
    """Start a background query task for a specific user."""
    # Import here to avoid circular import
    from pika.services.audit import get_audit_logger
    from pika.services.app_config import get_app_config
    from pika.services.history import get_history_service

    # Create query status
    query_status = QueryStatus(query_id=query_id, question=question, status="running")
    _set_query_status(query_status, username)

    async def run_query():
        import time as time_module
        query_start = time_module.time()
        logger.info(f"[RAG] Starting query {query_id}: '{question[:50]}...' with timeout={QUERY_TIMEOUT}s")

        try:
            rag = get_rag_engine()
            # Add timeout to prevent queries from running forever
            result = await asyncio.wait_for(
                rag.query(question=question, top_k=top_k),
                timeout=QUERY_TIMEOUT,
            )
            query_status.result = result
            query_status.status = "completed"
            elapsed = time_module.time() - query_start
            logger.info(f"[RAG] Query completed: {query_id} in {elapsed:.1f}s")

            # Audit log
            audit = get_audit_logger()
            audit.log_query(
                question=question,
                model=get_app_config().get_current_model(),
                confidence=result.confidence.value,
                sources=[s.filename for s in result.sources],
            )

            # Save to history
            history = get_history_service()
            history.add_query(
                question=question,
                answer=result.answer,
                confidence=result.confidence.value,
                sources=[s.filename for s in result.sources],
                username=username,
            )
        except asyncio.CancelledError:
            # Query was cancelled by user
            elapsed = time_module.time() - query_start
            query_status.status = "cancelled"
            query_status.error = "Query was cancelled"
            logger.info(f"[RAG] Query cancelled: {query_id} after {elapsed:.1f}s")
        except asyncio.TimeoutError:
            elapsed = time_module.time() - query_start
            query_status.status = "error"
            query_status.error = f"Query timed out after {QUERY_TIMEOUT} seconds"
            logger.error(f"[RAG] Query timed out: {query_id} after {elapsed:.1f}s (limit was {QUERY_TIMEOUT}s)")

            # Audit log timeout
            audit = get_audit_logger()
            audit.log_query(
                question=question,
                model=get_app_config().get_current_model(),
                confidence="none",
                sources=[],
                error="Query timed out",
            )
        except Exception as e:
            elapsed = time_module.time() - query_start
            query_status.error = str(e)
            query_status.status = "error"
            logger.error(f"[RAG] Query failed: {query_id} after {elapsed:.1f}s - {type(e).__name__}: {e}")

            # Audit log error
            audit = get_audit_logger()
            audit.log_query(
                question=question,
                model=get_app_config().get_current_model(),
                confidence="none",
                sources=[],
                error=str(e),
            )

    key = _get_user_key(username)
    _query_tasks[key] = asyncio.create_task(run_query())
    return query_status
