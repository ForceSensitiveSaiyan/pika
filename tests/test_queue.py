"""Tests for the query queue system."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pika.services.rag import (
    QueryStatus,
    QueuedQuery,
    QueueFullError,
    QueueStats,
    UserQueueLimitError,
    _active_queries,
    _query_queue,
    _running_queries,
    get_queue_length,
    get_running_count,
    get_user_queued_count,
    init_queue_processor,
    remove_from_queue,
    shutdown_queue_processor,
    start_query_task,
)


class TestQueuedQuery:
    """Tests for QueuedQuery dataclass."""

    def test_create_queued_query(self):
        """Test creating a queued query."""
        query = QueuedQuery(
            query_id="test123",
            question="What is the answer?",
            username="testuser",
            top_k=5,
        )
        assert query.query_id == "test123"
        assert query.question == "What is the answer?"
        assert query.username == "testuser"
        assert query.top_k == 5
        assert isinstance(query.queued_at, datetime)

    def test_queued_query_anonymous(self):
        """Test queued query with no username."""
        query = QueuedQuery(
            query_id="anon123",
            question="Anonymous question",
            username=None,
            top_k=None,
        )
        assert query.username is None
        assert query.top_k is None


class TestQueueStats:
    """Tests for QueueStats dataclass."""

    def test_initial_state(self):
        """Test initial state with no samples."""
        stats = QueueStats()
        assert stats.recent_durations == []
        assert stats.get_average_duration() == 30.0  # Default

    def test_record_duration(self):
        """Test recording query durations."""
        stats = QueueStats()
        stats.record_duration(10.0)
        stats.record_duration(20.0)
        assert len(stats.recent_durations) == 2
        assert stats.get_average_duration() == 15.0

    def test_max_samples_limit(self):
        """Test that samples are limited to max_samples."""
        stats = QueueStats(max_samples=3)
        for i in range(5):
            stats.record_duration(float(i * 10))
        assert len(stats.recent_durations) == 3
        # Should keep last 3: 20, 30, 40
        assert stats.recent_durations == [20.0, 30.0, 40.0]
        assert stats.get_average_duration() == 30.0


class TestQueryStatusWithQueue:
    """Tests for QueryStatus with queue fields."""

    def test_query_status_queue_fields(self):
        """Test QueryStatus includes queue fields."""
        status = QueryStatus(
            query_id="q123",
            question="Test?",
            status="queued",
            queue_position=3,
            queue_length=5,
            estimated_wait_seconds=45,
        )
        assert status.queue_position == 3
        assert status.queue_length == 5
        assert status.estimated_wait_seconds == 45

    def test_query_status_to_dict_includes_queue(self):
        """Test to_dict includes queue fields."""
        status = QueryStatus(
            query_id="q123",
            question="Test?",
            status="queued",
            queue_position=2,
            queue_length=4,
            estimated_wait_seconds=60,
        )
        result = status.to_dict()
        assert result["queue_position"] == 2
        assert result["queue_length"] == 4
        assert result["estimated_wait_seconds"] == 60

    def test_query_status_null_queue_fields(self):
        """Test queue fields are null when running."""
        status = QueryStatus(
            query_id="q123",
            question="Test?",
            status="running",
        )
        assert status.queue_position is None
        assert status.queue_length is None
        assert status.estimated_wait_seconds is None


class TestQueueHelpers:
    """Tests for queue helper functions."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up global state before and after tests."""
        _query_queue.clear()
        _running_queries.clear()
        _active_queries.clear()
        yield
        _query_queue.clear()
        _running_queries.clear()
        _active_queries.clear()

    def test_get_queue_length(self):
        """Test getting queue length."""
        assert get_queue_length() == 0

        # Add item
        _query_queue.append(QueuedQuery("q1", "test", None, None))
        assert get_queue_length() == 1

    def test_get_running_count(self):
        """Test getting running count."""
        assert get_running_count() == 0

        _running_queries.add("q1")
        _running_queries.add("q2")
        assert get_running_count() == 2

    def test_get_user_queued_count(self):
        """Test counting queries for a user."""
        _query_queue.append(QueuedQuery("q1", "test", "user1", None))
        _query_queue.append(QueuedQuery("q2", "test", "user2", None))
        _query_queue.append(QueuedQuery("q3", "test", "user1", None))

        assert get_user_queued_count("user1") == 2
        assert get_user_queued_count("user2") == 1
        assert get_user_queued_count("user3") == 0


class TestRemoveFromQueue:
    """Tests for remove_from_queue function."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up global state before and after tests."""
        _query_queue.clear()
        _running_queries.clear()
        _active_queries.clear()
        yield
        _query_queue.clear()
        _running_queries.clear()
        _active_queries.clear()

    def test_remove_existing_query(self):
        """Test removing an existing query from queue."""
        _query_queue.append(QueuedQuery("q1", "test1", None, None))
        _query_queue.append(QueuedQuery("q2", "test2", None, None))

        result = remove_from_queue("q1")
        assert result is True
        assert get_queue_length() == 1
        assert list(_query_queue)[0].query_id == "q2"

    def test_remove_nonexistent_query(self):
        """Test removing a query that doesn't exist."""
        result = remove_from_queue("nonexistent")
        assert result is False


class TestStartQueryTask:
    """Tests for start_query_task function."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up global state before and after tests."""
        _query_queue.clear()
        _running_queries.clear()
        _active_queries.clear()
        yield
        _query_queue.clear()
        _running_queries.clear()
        _active_queries.clear()

    @pytest.mark.asyncio
    async def test_immediate_execution_when_slots_available(self):
        """Test query runs immediately when slots are available."""
        with patch("pika.services.rag.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                max_concurrent_queries=2,
                max_queued_per_user=3,
                max_queue_size=100,
                queue_timeout=300,
            )
            with patch("pika.services.rag._execute_query", new_callable=AsyncMock):
                status = await start_query_task(
                    question="Test question",
                    query_id="test123",
                    username="testuser",
                )
                assert status.status == "running"
                assert status.queue_position is None

    @pytest.mark.asyncio
    async def test_queue_when_slots_full(self):
        """Test query is queued when all slots are full."""
        # Fill running slots
        _running_queries.add("running1")
        _running_queries.add("running2")

        with patch("pika.services.rag.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                max_concurrent_queries=2,
                max_queued_per_user=3,
                max_queue_size=100,
                queue_timeout=300,
            )

            status = await start_query_task(
                question="Test question",
                query_id="test123",
                username="testuser",
            )

            assert status.status == "queued"
            assert status.queue_position == 1
            assert status.queue_length == 1
            assert status.estimated_wait_seconds is not None

    @pytest.mark.asyncio
    async def test_queue_full_error(self):
        """Test QueueFullError when queue is at capacity."""
        # Fill queue
        for i in range(5):
            _query_queue.append(QueuedQuery(f"q{i}", "test", f"user{i}", None))
        _running_queries.add("running1")

        with patch("pika.services.rag.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                max_concurrent_queries=1,
                max_queued_per_user=3,
                max_queue_size=5,  # Queue is full
                queue_timeout=300,
            )

            with pytest.raises(QueueFullError):
                await start_query_task(
                    question="Test question",
                    query_id="new123",
                    username="newuser",
                )

    @pytest.mark.asyncio
    async def test_user_queue_limit_error(self):
        """Test UserQueueLimitError when user has too many queries."""
        # Fill queue with user's queries
        for i in range(3):
            _query_queue.append(QueuedQuery(f"q{i}", "test", "testuser", None))
        _running_queries.add("running1")

        with patch("pika.services.rag.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                max_concurrent_queries=1,
                max_queued_per_user=3,
                max_queue_size=100,
                queue_timeout=300,
            )

            with pytest.raises(UserQueueLimitError):
                await start_query_task(
                    question="Test question",
                    query_id="new123",
                    username="testuser",
                )


class TestQueueProcessor:
    """Tests for queue processor lifecycle."""

    @pytest.mark.asyncio
    async def test_init_and_shutdown(self):
        """Test queue processor can be initialized and shut down."""
        await init_queue_processor()
        # Give it a moment to start
        await asyncio.sleep(0.1)
        await shutdown_queue_processor()
        # Should complete without errors


class TestAPIContracts:
    """Tests for API response model contracts."""

    def test_query_start_response_with_queue_fields(self):
        """Test QueryStartResponse includes queue fields."""
        from pika.api.routes import QueryStartResponse

        response = QueryStartResponse(
            query_id="q123",
            status="queued",
            queue_position=3,
            queue_length=5,
            estimated_wait_seconds=45,
        )
        assert response.query_id == "q123"
        assert response.status == "queued"
        assert response.queue_position == 3
        assert response.queue_length == 5
        assert response.estimated_wait_seconds == 45

    def test_query_start_response_nullable_queue_fields(self):
        """Test QueryStartResponse queue fields are nullable."""
        from pika.api.routes import QueryStartResponse

        response = QueryStartResponse(
            query_id="q123",
            status="running",
        )
        assert response.queue_position is None
        assert response.queue_length is None
        assert response.estimated_wait_seconds is None

    def test_query_status_response_with_queue_fields(self):
        """Test QueryStatusResponse includes queue fields."""
        from pika.api.routes import QueryStatusResponse

        response = QueryStatusResponse(
            query_id="q123",
            question="Test?",
            status="queued",
            queue_position=2,
            queue_length=4,
            estimated_wait_seconds=30,
        )
        assert response.queue_position == 2
        assert response.queue_length == 4
        assert response.estimated_wait_seconds == 30
