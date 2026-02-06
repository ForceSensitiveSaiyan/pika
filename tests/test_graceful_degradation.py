"""Tests for graceful degradation end-to-end scenarios."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDegradedMode:
    """Tests for degraded mode when Ollama is unavailable."""

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
    def rag_with_documents(self, fresh_temp_dirs):
        """Create a RAG engine with indexed documents."""
        from pika.config import Settings
        from pika.services.documents import DocumentProcessor
        from pika.services.ollama import OllamaConnectionError
        from pika.services.rag import RAGEngine

        docs_dir = Path(fresh_temp_dirs["docs"])
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Clear any existing files
        for f in docs_dir.iterdir():
            if f.is_file():
                f.unlink()

        # Create test documents
        (docs_dir / "vacation-policy.txt").write_text(
            "Company Vacation Policy\n\n"
            "All employees are entitled to 20 days of paid vacation per year. "
            "Vacation requests must be submitted at least 2 weeks in advance. "
            "Unused vacation days can be carried over to the next year, up to a maximum of 5 days."
        )
        (docs_dir / "expense-policy.txt").write_text(
            "Expense Reimbursement Policy\n\n"
            "Employees can submit expense reports for business-related costs. "
            "Receipts are required for expenses over $25. "
            "Reimbursements are processed within 10 business days."
        )

        settings = Settings(
            documents_dir=fresh_temp_dirs["docs"],
            chroma_persist_dir=fresh_temp_dirs["data"],
            chunk_size=200,
            chunk_overlap=20,
            embedding_model="all-MiniLM-L6-v2",
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=1,
            circuit_breaker_recovery_timeout=60,
        )

        doc_processor = DocumentProcessor(settings=settings)

        # Create mock Ollama client that will fail with OllamaConnectionError
        # This triggers degraded mode
        mock_ollama = MagicMock()
        mock_ollama.generate = AsyncMock(side_effect=OllamaConnectionError("Test: Ollama unavailable"))
        mock_ollama.model = "test-model"

        engine = RAGEngine(
            settings=settings,
            document_processor=doc_processor,
            ollama_client=mock_ollama,
        )

        # Index documents
        engine.index_documents()
        return engine

    @pytest.mark.asyncio
    async def test_degraded_mode_returns_sources(self, rag_with_documents):
        """Verify degraded mode returns relevant sources when Ollama fails."""
        from pika.services.rag import Confidence

        # Query should fail but return degraded response with sources
        result = await rag_with_documents.query("How many vacation days do I get?")

        assert result.confidence == Confidence.DEGRADED
        assert len(result.sources) > 0

        # Should include content from vacation policy
        source_content = " ".join([s.content for s in result.sources])
        assert "vacation" in source_content.lower()

    @pytest.mark.asyncio
    async def test_degraded_mode_message_is_user_friendly(self, rag_with_documents):
        """Verify degraded mode has user-friendly message."""
        result = await rag_with_documents.query("What is the expense policy?")

        # Message should explain AI is unavailable
        assert "unavailable" in result.answer.lower() or "relevant sections" in result.answer.lower()

        # Should NOT contain technical error messages
        assert "exception" not in result.answer.lower()
        assert "traceback" not in result.answer.lower()

    @pytest.mark.asyncio
    async def test_degraded_mode_includes_source_snippets(self, rag_with_documents):
        """Verify degraded mode answer includes source snippets."""
        result = await rag_with_documents.query("How do I submit expenses?")

        # The degraded response should include actual document content
        assert "Expense" in result.answer or "expense" in result.answer

    @pytest.mark.asyncio
    async def test_degraded_mode_no_sources_found(self, fresh_temp_dirs):
        """Verify graceful handling when no relevant sources found."""
        from pika.config import Settings
        from pika.services.documents import DocumentProcessor
        from pika.services.rag import Confidence, RAGEngine

        # Create empty index
        docs_dir = Path(fresh_temp_dirs["docs"])
        data_dir = Path(fresh_temp_dirs["data"])

        settings = Settings(
            documents_dir=str(docs_dir),
            chroma_persist_dir=str(data_dir),
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=1,
        )

        doc_processor = DocumentProcessor(settings=settings)
        mock_ollama = MagicMock()
        mock_ollama.generate = AsyncMock(side_effect=Exception("Unavailable"))

        engine = RAGEngine(
            settings=settings,
            document_processor=doc_processor,
            ollama_client=mock_ollama,
        )

        # Query with no documents indexed
        result = await engine.query("Random question?")

        assert result.confidence == Confidence.NONE
        # Should have a message indicating no documents found
        assert "no" in result.answer.lower() or "found" in result.answer.lower()


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration with RAG queries."""

    @pytest.fixture
    def fresh_temp_dirs(self):
        """Create fresh temporary directories."""
        docs_dir = tempfile.mkdtemp()
        data_dir = tempfile.mkdtemp()

        yield {"docs": docs_dir, "data": data_dir}
        shutil.rmtree(docs_dir, ignore_errors=True)
        shutil.rmtree(data_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, fresh_temp_dirs):
        """Verify circuit breaker opens after consecutive failures by recording failures directly."""
        import pika.services.ollama as ollama_module
        from pika.services.ollama import CircuitBreaker, CircuitState, get_circuit_breaker

        # Reset circuit breaker singleton
        ollama_module._circuit_breaker = None

        # Create a fresh circuit breaker with low threshold
        circuit = CircuitBreaker(failure_threshold=2, recovery_timeout=60)
        ollama_module._circuit_breaker = circuit

        # Record failures directly on the circuit breaker
        await circuit.record_failure()
        await circuit.record_failure()

        # Circuit should be open now
        assert circuit.state == CircuitState.OPEN

        # Get from singleton should be the same
        retrieved = get_circuit_breaker()
        assert retrieved.state == CircuitState.OPEN


class TestStatusEndpoint:
    """Tests for the quick status endpoint."""

    @pytest.mark.asyncio
    async def test_status_endpoint_returns_healthy(self):
        """Verify status endpoint returns healthy when all systems work."""
        from unittest.mock import AsyncMock, MagicMock

        import pika.services.ollama as ollama_module
        from pika.api.routes import quick_status
        from pika.services.ollama import OllamaClient
        from pika.services.rag import IndexStats, RAGEngine

        # Reset circuit breaker
        ollama_module._circuit_breaker = None

        # Mock healthy ollama
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.health_check = AsyncMock(return_value=True)

        # Mock RAG with some indexed content
        mock_rag = MagicMock(spec=RAGEngine)
        mock_rag.get_stats.return_value = IndexStats(
            total_documents=5,
            total_chunks=100,
            collection_name="test",
        )

        with patch("pika.api.routes.is_indexing_running", return_value=False):
            result = await quick_status(ollama=mock_ollama, rag=mock_rag)

        assert result.status == "healthy"
        assert result.ollama_connected is True
        assert result.circuit_breaker_open is False
        assert result.index_chunks == 100
        assert result.indexing_in_progress is False

    @pytest.mark.asyncio
    async def test_status_endpoint_returns_unhealthy_when_ollama_down(self):
        """Verify status endpoint returns unhealthy when Ollama is down."""
        from unittest.mock import AsyncMock, MagicMock

        import pika.services.ollama as ollama_module
        from pika.api.routes import quick_status
        from pika.services.ollama import OllamaClient
        from pika.services.rag import IndexStats, RAGEngine

        # Reset circuit breaker
        ollama_module._circuit_breaker = None

        # Mock unhealthy ollama
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.health_check = AsyncMock(return_value=False)

        # Mock RAG
        mock_rag = MagicMock(spec=RAGEngine)
        mock_rag.get_stats.return_value = IndexStats(
            total_documents=5,
            total_chunks=100,
            collection_name="test",
        )

        with patch("pika.api.routes.is_indexing_running", return_value=False):
            result = await quick_status(ollama=mock_ollama, rag=mock_rag)

        assert result.status == "unhealthy"
        assert result.ollama_connected is False

    @pytest.mark.asyncio
    async def test_status_endpoint_returns_degraded_when_circuit_open(self):
        """Verify status endpoint returns degraded when circuit breaker is open."""
        from unittest.mock import AsyncMock, MagicMock

        import pika.services.ollama as ollama_module
        from pika.api.routes import quick_status
        from pika.services.ollama import CircuitState, OllamaClient, get_circuit_breaker
        from pika.services.rag import IndexStats, RAGEngine

        # Reset and trip circuit breaker
        ollama_module._circuit_breaker = None
        circuit = get_circuit_breaker()

        # Trip the circuit
        for _ in range(10):  # More than threshold
            await circuit.record_failure()

        assert circuit.state == CircuitState.OPEN

        # Mock healthy ollama but circuit is open
        mock_ollama = MagicMock(spec=OllamaClient)
        mock_ollama.health_check = AsyncMock(return_value=True)

        mock_rag = MagicMock(spec=RAGEngine)
        mock_rag.get_stats.return_value = IndexStats(
            total_documents=5,
            total_chunks=100,
            collection_name="test",
        )

        with patch("pika.api.routes.is_indexing_running", return_value=False):
            result = await quick_status(ollama=mock_ollama, rag=mock_rag)

        assert result.status == "degraded"
        assert result.circuit_breaker_open is True


class TestCacheInvalidationOnIndex:
    """Tests for cache invalidation when index is rebuilt."""

    @pytest.fixture
    def fresh_temp_dirs(self):
        """Create fresh temporary directories."""
        docs_dir = tempfile.mkdtemp()
        data_dir = tempfile.mkdtemp()

        yield {"docs": docs_dir, "data": data_dir}
        shutil.rmtree(docs_dir, ignore_errors=True)
        shutil.rmtree(data_dir, ignore_errors=True)

    def test_index_documents_invalidates_cache(self, fresh_temp_dirs):
        """Verify index_documents() invalidates the query cache."""
        import pika.services.rag as rag_module
        from pika.config import Settings
        from pika.services.documents import DocumentProcessor
        from pika.services.rag import Confidence, QueryResult, RAGEngine, get_query_cache

        # Reset cache singleton
        rag_module._query_cache = None

        docs_dir = Path(fresh_temp_dirs["docs"])
        (docs_dir / "test.txt").write_text("Test content for caching tests.")

        settings = Settings(
            documents_dir=str(docs_dir),
            chroma_persist_dir=fresh_temp_dirs["data"],
            query_cache_enabled=True,
            query_cache_max_size=100,
            query_cache_ttl=300,
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

        # Add something to the cache
        cache = get_query_cache()
        result = QueryResult(answer="Cached", sources=[], confidence=Confidence.HIGH)
        cache.set("Test question?", doc_count=1, chunk_count=10, result=result)

        assert cache.size() > 0

        # Index documents should invalidate cache
        engine.index_documents()

        assert cache.size() == 0


class TestConfidenceDegraded:
    """Tests for the DEGRADED confidence level."""

    def test_degraded_confidence_exists(self):
        """Verify DEGRADED confidence level exists."""
        from pika.services.rag import Confidence

        assert hasattr(Confidence, "DEGRADED")
        assert Confidence.DEGRADED.value == "degraded"

    def test_degraded_response_format(self):
        """Verify degraded response is properly formatted."""
        from pika.services.rag import Source, _format_degraded_response

        sources = [
            Source(
                filename="test.pdf",
                chunk_index=0,
                content="This is the first source content that should appear in the response.",
                similarity=0.9,
            ),
            Source(
                filename="other.txt",
                chunk_index=1,
                content="This is the second source with different information.",
                similarity=0.8,
            ),
        ]

        response = _format_degraded_response(sources)

        assert "unavailable" in response.lower()
        assert "test.pdf" in response
        assert "other.txt" in response

    def test_degraded_response_empty_sources(self):
        """Verify degraded response handles empty sources."""
        from pika.services.rag import _format_degraded_response

        response = _format_degraded_response([])

        assert "unavailable" in response.lower()
        assert "no relevant documents" in response.lower()

    def test_degraded_response_truncates_long_content(self):
        """Verify degraded response truncates long source content."""
        from pika.services.rag import Source, _format_degraded_response

        long_content = "x" * 500  # 500 characters

        sources = [
            Source(
                filename="long.txt",
                chunk_index=0,
                content=long_content,
                similarity=0.9,
            ),
        ]

        response = _format_degraded_response(sources)

        # Content should be truncated (200 chars max + "...")
        assert len(response) < 500
        assert "..." in response
