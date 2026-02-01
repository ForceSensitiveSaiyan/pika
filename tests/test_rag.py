"""Tests for RAG engine functionality."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRAGEngine:
    """Tests for RAGEngine class."""

    @pytest.fixture
    def fresh_temp_dirs(self):
        """Create fresh temporary directories for each test."""
        docs_dir = tempfile.mkdtemp()
        data_dir = tempfile.mkdtemp()

        # Ensure directories are truly empty
        for f in Path(docs_dir).iterdir():
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)

        yield {"docs": docs_dir, "data": data_dir}
        # Cleanup
        shutil.rmtree(docs_dir, ignore_errors=True)
        shutil.rmtree(data_dir, ignore_errors=True)

    @pytest.fixture
    def rag_engine(self, fresh_temp_dirs):
        """Create a RAG engine with fresh temp directories."""
        from pika.config import Settings
        from pika.services.documents import DocumentProcessor
        from pika.services.rag import RAGEngine

        # Ensure docs dir is empty
        docs_dir = Path(fresh_temp_dirs["docs"])
        for f in docs_dir.iterdir():
            if f.is_file():
                f.unlink()

        settings = Settings(
            documents_dir=fresh_temp_dirs["docs"],
            chroma_persist_dir=fresh_temp_dirs["data"],
            chunk_size=100,
            chunk_overlap=20,
            embedding_model="all-MiniLM-L6-v2",
        )

        # Create document processor with the same settings
        doc_processor = DocumentProcessor(settings=settings)

        # Create mock Ollama client
        mock_ollama = MagicMock()
        mock_ollama.generate = AsyncMock(return_value="Test answer from LLM")
        mock_ollama.model = "test-model"

        engine = RAGEngine(
            settings=settings,
            document_processor=doc_processor,
            ollama_client=mock_ollama,
        )
        # Ensure index is clear
        engine.clear_index()
        return engine

    @pytest.fixture
    def rag_with_documents(self, fresh_temp_dirs):
        """Create a RAG engine with indexed documents."""
        from pika.config import Settings
        from pika.services.documents import DocumentProcessor
        from pika.services.rag import RAGEngine

        docs_dir = Path(fresh_temp_dirs["docs"])
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Clear any existing files
        for f in docs_dir.iterdir():
            if f.is_file():
                f.unlink()

        # Create test documents
        (docs_dir / "doc1.txt").write_text(
            "PIKA is a self-hosted RAG system. It helps small businesses "
            "manage their knowledge base. PIKA uses ChromaDB for vector storage."
        )
        (docs_dir / "doc2.txt").write_text(
            "Installation guide for PIKA. First install Docker, then run "
            "the setup script. Configure your Ollama endpoint."
        )

        settings = Settings(
            documents_dir=fresh_temp_dirs["docs"],
            chroma_persist_dir=fresh_temp_dirs["data"],
            chunk_size=100,
            chunk_overlap=20,
            embedding_model="all-MiniLM-L6-v2",
        )

        # Create document processor with the same settings
        doc_processor = DocumentProcessor(settings=settings)

        # Create mock Ollama client
        mock_ollama = MagicMock()
        mock_ollama.generate = AsyncMock(return_value="Test answer from LLM")
        mock_ollama.model = "test-model"

        engine = RAGEngine(
            settings=settings,
            document_processor=doc_processor,
            ollama_client=mock_ollama,
        )

        # Ensure index is clear first, then index the documents
        engine.clear_index()
        engine.index_documents()
        return engine

    def test_index_creates_embeddings(self, rag_with_documents):
        """Verify indexing stores vectors in ChromaDB."""
        stats = rag_with_documents.get_stats()

        # Should have at least the 2 documents we created
        assert stats.total_documents >= 2
        assert stats.total_chunks > 0
        assert stats.collection_name == "pika_documents"

    def test_index_empty_directory(self, rag_engine):
        """Verify indexing empty directory returns zero counts."""
        # rag_engine fixture already clears the index
        stats = rag_engine.index_documents()

        # Should have 0 documents since directory is empty
        assert stats.total_documents == 0
        assert stats.total_chunks == 0

    def test_clear_index(self, rag_with_documents):
        """Verify clearing index removes all documents."""
        # First verify we have documents
        stats = rag_with_documents.get_stats()
        assert stats.total_chunks > 0

        # Clear and verify
        rag_with_documents.clear_index()
        stats = rag_with_documents.get_stats()
        assert stats.total_chunks == 0

    def test_get_indexed_documents(self, rag_with_documents):
        """Verify we can list indexed documents."""
        documents = rag_with_documents.get_indexed_documents()

        # Should have at least the 2 documents we created
        assert len(documents) >= 2
        filenames = [d.filename for d in documents]
        assert "doc1.txt" in filenames
        assert "doc2.txt" in filenames

    @pytest.mark.asyncio
    async def test_query_returns_sources(self, rag_with_documents):
        """Verify query includes source citations."""
        result = await rag_with_documents.query("What is PIKA?")

        assert result.answer is not None
        assert len(result.sources) > 0
        # Should find doc1.txt which mentions PIKA
        source_files = [s.filename for s in result.sources]
        assert "doc1.txt" in source_files

    @pytest.mark.asyncio
    async def test_empty_index_handled(self, rag_engine):
        """Verify query on empty index returns helpful message."""
        result = await rag_engine.query("What is PIKA?")

        assert "No documents indexed" in result.answer
        assert result.confidence.value == "none"
        assert len(result.sources) == 0

    @pytest.mark.asyncio
    async def test_query_includes_confidence(self, rag_with_documents):
        """Verify query result includes confidence level."""
        result = await rag_with_documents.query("What is PIKA?")

        assert result.confidence is not None
        assert result.confidence.value in ["high", "medium", "low", "none"]

    @pytest.mark.asyncio
    async def test_query_respects_top_k(self, rag_with_documents):
        """Verify query respects top_k parameter."""
        result = await rag_with_documents.query("What is PIKA?", top_k=1)

        assert len(result.sources) <= 1

    @pytest.mark.asyncio
    async def test_query_similarity_scores(self, rag_with_documents):
        """Verify sources include similarity scores."""
        result = await rag_with_documents.query("What is PIKA?")

        for source in result.sources:
            assert 0 <= source.similarity <= 1
            assert source.chunk_index >= 0
            assert source.content is not None

    def test_reindex_replaces_documents(self, fresh_temp_dirs):
        """Verify reindexing replaces all documents."""
        from pika.config import Settings
        from pika.services.documents import DocumentProcessor
        from pika.services.rag import RAGEngine

        docs_dir = Path(fresh_temp_dirs["docs"])
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Clear any existing files
        for f in docs_dir.iterdir():
            if f.is_file():
                f.unlink()

        # Create initial documents
        (docs_dir / "doc1.txt").write_text("First document.")
        (docs_dir / "doc2.txt").write_text("Second document.")

        settings = Settings(
            documents_dir=fresh_temp_dirs["docs"],
            chroma_persist_dir=fresh_temp_dirs["data"],
            chunk_size=100,
            chunk_overlap=20,
            embedding_model="all-MiniLM-L6-v2",
        )

        doc_processor = DocumentProcessor(settings=settings)

        mock_ollama = MagicMock()
        mock_ollama.generate = AsyncMock(return_value="Test answer")
        mock_ollama.model = "test-model"

        engine = RAGEngine(
            settings=settings,
            document_processor=doc_processor,
            ollama_client=mock_ollama,
        )

        # Clear and index initial documents
        engine.clear_index()
        initial_stats = engine.index_documents()
        assert initial_stats.total_documents == 2

        # Add another document
        (docs_dir / "doc3.txt").write_text("Third document content.")

        # Reindex
        new_stats = engine.index_documents()

        assert new_stats.total_documents == 3


class TestConfidenceLevels:
    """Tests for confidence level calculation."""

    @pytest.fixture
    def fresh_temp_dirs(self):
        """Create fresh temporary directories for each test."""
        docs_dir = tempfile.mkdtemp()
        data_dir = tempfile.mkdtemp()
        yield {"docs": docs_dir, "data": data_dir}
        # Cleanup
        shutil.rmtree(docs_dir, ignore_errors=True)
        shutil.rmtree(data_dir, ignore_errors=True)

    @pytest.fixture
    def rag_engine(self, fresh_temp_dirs):
        """Create a RAG engine for confidence tests."""
        from pika.config import Settings
        from pika.services.rag import RAGEngine

        settings = Settings(
            documents_dir=fresh_temp_dirs["docs"],
            chroma_persist_dir=fresh_temp_dirs["data"],
            confidence_high=0.8,
            confidence_medium=0.6,
            confidence_low=0.4,
        )

        mock_ollama = MagicMock()
        mock_ollama.generate = AsyncMock(return_value="Test answer")
        mock_ollama.model = "test-model"

        return RAGEngine(settings=settings, ollama_client=mock_ollama)

    def test_high_similarity_high_confidence(self, rag_engine):
        """Verify high similarity scores produce high confidence."""
        from pika.services.rag import Confidence

        confidence = rag_engine._get_confidence([0.95, 0.85, 0.75])
        assert confidence == Confidence.HIGH

    def test_medium_similarity_medium_confidence(self, rag_engine):
        """Verify medium similarity scores produce medium confidence."""
        from pika.services.rag import Confidence

        confidence = rag_engine._get_confidence([0.7, 0.65, 0.55])
        assert confidence == Confidence.MEDIUM

    def test_low_similarity_low_confidence(self, rag_engine):
        """Verify low similarity scores produce low confidence."""
        from pika.services.rag import Confidence

        confidence = rag_engine._get_confidence([0.5, 0.45, 0.35])
        assert confidence == Confidence.LOW

    def test_very_low_similarity_none_confidence(self, rag_engine):
        """Verify very low similarity scores produce none confidence."""
        from pika.services.rag import Confidence

        confidence = rag_engine._get_confidence([0.3, 0.2, 0.1])
        assert confidence == Confidence.NONE

    def test_empty_similarities_none_confidence(self, rag_engine):
        """Verify empty similarity list produces none confidence."""
        from pika.services.rag import Confidence

        confidence = rag_engine._get_confidence([])
        assert confidence == Confidence.NONE


class TestQueryStatus:
    """Tests for query status tracking."""

    def test_query_status_creation(self):
        """Verify QueryStatus can be created."""
        from pika.services.rag import QueryStatus

        status = QueryStatus(
            query_id="test123",
            question="What is PIKA?",
            status="running",
        )

        assert status.query_id == "test123"
        assert status.question == "What is PIKA?"
        assert status.status == "running"
        assert status.result is None
        assert status.error is None

    def test_query_status_to_dict(self):
        """Verify QueryStatus converts to dict correctly."""
        from pika.services.rag import QueryStatus, QueryResult, Source, Confidence

        result = QueryResult(
            answer="PIKA is a RAG system.",
            sources=[
                Source(
                    filename="doc.txt",
                    chunk_index=0,
                    content="PIKA content",
                    similarity=0.9,
                )
            ],
            confidence=Confidence.HIGH,
        )

        status = QueryStatus(
            query_id="test123",
            question="What is PIKA?",
            status="completed",
            result=result,
        )

        data = status.to_dict()

        assert data["query_id"] == "test123"
        assert data["status"] == "completed"
        assert data["result"]["answer"] == "PIKA is a RAG system."
        assert len(data["result"]["sources"]) == 1
        assert data["result"]["confidence"] == "high"


class TestOllamaErrors:
    """Tests for Ollama error handling in RAG."""

    @pytest.fixture
    def fresh_temp_dirs(self):
        """Create fresh temporary directories for each test."""
        docs_dir = tempfile.mkdtemp()
        data_dir = tempfile.mkdtemp()
        yield {"docs": docs_dir, "data": data_dir}
        # Cleanup
        shutil.rmtree(docs_dir, ignore_errors=True)
        shutil.rmtree(data_dir, ignore_errors=True)

    @pytest.fixture
    def rag_with_docs(self, fresh_temp_dirs):
        """Create a RAG engine with documents for error testing."""
        from pika.config import Settings
        from pika.services.documents import DocumentProcessor
        from pika.services.rag import RAGEngine

        docs_dir = Path(fresh_temp_dirs["docs"])
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Clear any existing files
        for f in docs_dir.iterdir():
            if f.is_file():
                f.unlink()

        (docs_dir / "test.txt").write_text("Test content for PIKA.")

        settings = Settings(
            documents_dir=fresh_temp_dirs["docs"],
            chroma_persist_dir=fresh_temp_dirs["data"],
        )

        doc_processor = DocumentProcessor(settings=settings)

        return settings, doc_processor

    @pytest.mark.asyncio
    async def test_connection_error_handled(self, rag_with_docs):
        """Verify Ollama connection errors trigger degraded mode."""
        from pika.services.ollama import OllamaConnectionError
        from pika.services.rag import RAGEngine, Confidence

        settings, doc_processor = rag_with_docs

        mock_ollama = MagicMock()
        mock_ollama.generate = AsyncMock(side_effect=OllamaConnectionError("Connection refused"))
        mock_ollama.model = "test-model"

        engine = RAGEngine(
            settings=settings,
            document_processor=doc_processor,
            ollama_client=mock_ollama,
        )
        engine.clear_index()
        engine.index_documents()

        result = await engine.query("What is PIKA?")

        # With graceful degradation, connection errors return degraded mode with sources
        assert "unavailable" in result.answer.lower()
        assert result.confidence == Confidence.DEGRADED
        assert len(result.sources) > 0  # Should include search results

    @pytest.mark.asyncio
    async def test_model_not_found_error_handled(self, rag_with_docs):
        """Verify model not found errors are handled gracefully."""
        from pika.services.ollama import OllamaModelNotFoundError
        from pika.services.rag import RAGEngine, Confidence

        settings, doc_processor = rag_with_docs

        mock_ollama = MagicMock()
        mock_ollama.generate = AsyncMock(
            side_effect=OllamaModelNotFoundError(model="missing-model")
        )
        mock_ollama.model = "missing-model"

        engine = RAGEngine(
            settings=settings,
            document_processor=doc_processor,
            ollama_client=mock_ollama,
        )
        engine.clear_index()
        engine.index_documents()

        result = await engine.query("What is PIKA?")

        assert "not available" in result.answer
        assert result.confidence == Confidence.NONE

    @pytest.mark.asyncio
    async def test_timeout_error_handled(self, rag_with_docs):
        """Verify timeout errors trigger degraded mode."""
        from pika.services.ollama import OllamaTimeoutError
        from pika.services.rag import RAGEngine, Confidence

        settings, doc_processor = rag_with_docs

        mock_ollama = MagicMock()
        mock_ollama.generate = AsyncMock(side_effect=OllamaTimeoutError("Timeout"))
        mock_ollama.model = "test-model"

        engine = RAGEngine(
            settings=settings,
            document_processor=doc_processor,
            ollama_client=mock_ollama,
        )
        engine.clear_index()
        engine.index_documents()

        result = await engine.query("What is PIKA?")

        # With graceful degradation, timeout errors return degraded mode with sources
        assert "unavailable" in result.answer.lower()
        assert result.confidence == Confidence.DEGRADED
        assert len(result.sources) > 0  # Should include search results


class TestQueryCancellation:
    """Tests for query cancellation functionality."""

    def test_is_query_running_false_initially(self):
        """Verify no query is running initially."""
        from pika.services.rag import is_query_running

        # Reset state
        import pika.services.rag as rag_module
        rag_module._query_task = None

        assert is_query_running() is False

    def test_cancel_query_when_none_running(self):
        """Verify cancelling when no query running returns False."""
        from pika.services.rag import cancel_query

        # Reset state
        import pika.services.rag as rag_module
        rag_module._query_task = None

        result = cancel_query()
        assert result is False

    def test_clear_query_status(self):
        """Verify query status can be cleared."""
        from pika.services.rag import (
            get_active_query,
            clear_query_status,
            _set_query_status,
            QueryStatus,
        )

        # Set a query status
        status = QueryStatus(query_id="test", question="Test?", status="running")
        _set_query_status(status)

        assert get_active_query() is not None

        clear_query_status()
        assert get_active_query() is None


class TestAsyncIndexing:
    """Tests for async indexing functionality."""

    def test_index_status_creation(self):
        """Verify IndexStatus can be created with default values."""
        from pika.services.rag import IndexStatus

        status = IndexStatus(index_id="test123")

        assert status.index_id == "test123"
        assert status.status == "pending"
        assert status.total_documents == 0
        assert status.processed_documents == 0
        assert status.total_chunks == 0
        assert status.current_file is None
        assert status.error is None
        assert status.started_at is not None
        assert status.completed_at is None

    def test_index_status_percent_zero_documents(self):
        """Verify percent is 0 when total_documents is 0."""
        from pika.services.rag import IndexStatus

        status = IndexStatus(index_id="test123")
        assert status.percent == 0

    def test_index_status_percent_partial(self):
        """Verify percent calculation with partial progress."""
        from pika.services.rag import IndexStatus

        status = IndexStatus(
            index_id="test123",
            total_documents=10,
            processed_documents=4,
        )
        assert status.percent == 40

    def test_index_status_percent_complete(self):
        """Verify percent is 100 when all documents processed."""
        from pika.services.rag import IndexStatus

        status = IndexStatus(
            index_id="test123",
            total_documents=5,
            processed_documents=5,
        )
        assert status.percent == 100

    def test_index_status_to_dict(self):
        """Verify IndexStatus converts to dict correctly."""
        from pika.services.rag import IndexStatus

        status = IndexStatus(
            index_id="test123",
            status="running",
            total_documents=10,
            processed_documents=5,
            total_chunks=100,
            current_file="test.pdf",
        )

        data = status.to_dict()

        assert data["index_id"] == "test123"
        assert data["status"] == "running"
        assert data["total_documents"] == 10
        assert data["processed_documents"] == 5
        assert data["total_chunks"] == 100
        assert data["current_file"] == "test.pdf"
        assert data["percent"] == 50
        assert data["started_at"] is not None
        assert data["completed_at"] is None
        assert data["error"] is None

    def test_is_indexing_running_false_initially(self):
        """Verify no indexing is running initially."""
        from pika.services.rag import is_indexing_running

        # Reset state
        import pika.services.rag as rag_module
        rag_module._index_task = None

        assert is_indexing_running() is False

    def test_cancel_index_when_not_running(self):
        """Verify cancelling when no indexing running returns False."""
        from pika.services.rag import cancel_index_task

        # Reset state
        import pika.services.rag as rag_module
        rag_module._index_task = None

        result = cancel_index_task()
        assert result is False

    def test_get_active_index_none_initially(self):
        """Verify get_active_index returns None initially."""
        from pika.services.rag import get_active_index, _set_active_index

        # Reset state
        _set_active_index(None)

        assert get_active_index() is None

    def test_set_and_get_active_index(self):
        """Verify setting and getting active index works."""
        from pika.services.rag import (
            get_active_index,
            _set_active_index,
            IndexStatus,
        )

        # Reset state first
        _set_active_index(None)
        assert get_active_index() is None

        # Set an index status
        status = IndexStatus(index_id="test456", status="running")
        _set_active_index(status)

        result = get_active_index()
        assert result is not None
        assert result.index_id == "test456"
        assert result.status == "running"

        # Cleanup
        _set_active_index(None)


class TestCaching:
    """Tests for stats and document caching."""

    def test_cache_invalidation(self, temp_dirs):
        """Verify cache is invalidated after clear_index."""
        import os
        from pika.services.rag import RAGEngine, IndexStats

        settings = MagicMock()
        settings.chroma_persist_dir = os.path.join(temp_dirs["data"], "chroma")
        settings.documents_dir = temp_dirs["docs"]
        settings.embedding_model = "all-MiniLM-L6-v2"

        rag = RAGEngine(settings=settings)

        # Manually set cache
        rag._stats_cache = IndexStats(
            total_documents=10,
            total_chunks=100,
            collection_name="test",
        )
        rag._stats_cache_time = 9999999999  # Far future

        # Clear index should invalidate cache
        rag.clear_index()

        assert rag._stats_cache is None
        assert rag._stats_cache_time == 0

    def test_update_cache(self, temp_dirs):
        """Verify _update_cache correctly sets cache values."""
        import os
        from pika.services.rag import RAGEngine, IndexStats, IndexedDocument

        settings = MagicMock()
        settings.chroma_persist_dir = os.path.join(temp_dirs["data"], "chroma")
        settings.documents_dir = temp_dirs["docs"]
        settings.embedding_model = "all-MiniLM-L6-v2"

        rag = RAGEngine(settings=settings)

        stats = IndexStats(
            total_documents=5,
            total_chunks=50,
            collection_name="test",
        )
        docs = [
            IndexedDocument(filename="doc1.pdf", chunk_count=25),
            IndexedDocument(filename="doc2.pdf", chunk_count=25),
        ]

        rag._update_cache(stats, docs)

        assert rag._stats_cache is not None
        assert rag._stats_cache.total_documents == 5
        assert rag._docs_cache is not None
        assert len(rag._docs_cache) == 2
