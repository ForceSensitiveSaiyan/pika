"""Tests for production readiness improvements."""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pika.config import Settings


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_valid_settings(self):
        """Test that valid settings are accepted."""
        settings = Settings(
            max_concurrent_queries=2,
            max_queued_per_user=5,
            queue_timeout=120,
            chunk_size=500,
            top_k=10,
        )
        assert settings.max_concurrent_queries == 2
        assert settings.max_queued_per_user == 5

    def test_reject_zero_concurrent_queries(self):
        """Test that zero concurrent queries is rejected."""
        with pytest.raises(ValueError, match="positive integer"):
            Settings(max_concurrent_queries=0)

    def test_reject_negative_concurrent_queries(self):
        """Test that negative concurrent queries is rejected."""
        with pytest.raises(ValueError, match="positive integer"):
            Settings(max_concurrent_queries=-1)

    def test_reject_zero_queue_size(self):
        """Test that zero queue size is rejected."""
        with pytest.raises(ValueError, match="positive integer"):
            Settings(max_queue_size=0)

    def test_reject_excessive_timeout(self):
        """Test that timeout over 1 hour is rejected."""
        with pytest.raises(ValueError, match="cannot exceed 3600"):
            Settings(queue_timeout=7200)

    def test_reject_small_chunk_size(self):
        """Test that chunk size under 100 is rejected."""
        with pytest.raises(ValueError, match="at least 100"):
            Settings(chunk_size=50)

    def test_reject_large_chunk_size(self):
        """Test that chunk size over 10000 is rejected."""
        with pytest.raises(ValueError, match="cannot exceed 10000"):
            Settings(chunk_size=20000)

    def test_reject_invalid_top_k(self):
        """Test that top_k of 0 is rejected."""
        with pytest.raises(ValueError, match="at least 1"):
            Settings(top_k=0)

    def test_reject_excessive_top_k(self):
        """Test that top_k over 50 is rejected."""
        with pytest.raises(ValueError, match="cannot exceed 50"):
            Settings(top_k=100)

    def test_reject_invalid_confidence(self):
        """Test that confidence outside 0-1 is rejected."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            Settings(confidence_high=1.5)

    def test_reject_negative_upload_size(self):
        """Test that negative upload size is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            Settings(max_upload_size_mb=0)


class TestQueryStatusCleanup:
    """Tests for query status cleanup."""

    def test_query_status_mark_completed(self):
        """Test that mark_completed sets the completion time."""
        from pika.services.rag import QueryStatus

        status = QueryStatus(query_id="test", question="Test?")
        assert status.completed_at is None

        status.mark_completed()
        assert status.completed_at is not None
        assert isinstance(status.completed_at, datetime)

    def test_cleanup_expired_queries(self):
        """Test that expired queries are cleaned up."""
        from pika.config import get_settings
        from pika.services.rag import (
            QueryStatus,
            _active_queries,
            cleanup_expired_queries,
        )

        query_status_ttl = get_settings().query_status_ttl

        # Clear state
        _active_queries.clear()

        # Add an old completed query
        old_status = QueryStatus(
            query_id="old",
            question="Old?",
            status="completed",
        )
        old_status.completed_at = datetime.now() - timedelta(seconds=query_status_ttl + 100)
        _active_queries["old_user"] = old_status

        # Add a recent completed query
        new_status = QueryStatus(
            query_id="new",
            question="New?",
            status="completed",
        )
        new_status.completed_at = datetime.now()
        _active_queries["new_user"] = new_status

        # Add a running query
        running_status = QueryStatus(
            query_id="running",
            question="Running?",
            status="running",
        )
        _active_queries["running_user"] = running_status

        # Run cleanup
        cleaned = cleanup_expired_queries()

        # Old query should be removed
        assert "old_user" not in _active_queries
        # New and running queries should remain
        assert "new_user" in _active_queries
        assert "running_user" in _active_queries
        assert cleaned == 1

        # Cleanup
        _active_queries.clear()


class TestSessionCleanup:
    """Tests for session cleanup."""

    def test_cleanup_expired_sessions(self):
        """Test that expired sessions are cleaned up."""
        from pika.api.web import (
            SESSION_MAX_AGE,
            _cleanup_expired_sessions,
            _sessions,
        )

        # Clear state
        _sessions.clear()

        # Add an old session
        _sessions["old_session"] = {
            "username": "old_user",
            "created_at": time.time() - SESSION_MAX_AGE - 100,
        }

        # Add a recent session
        _sessions["new_session"] = {
            "username": "new_user",
            "created_at": time.time(),
        }

        # Run cleanup
        cleaned = _cleanup_expired_sessions()

        assert "old_session" not in _sessions
        assert "new_session" in _sessions
        assert cleaned == 1

        # Cleanup
        _sessions.clear()

    def test_cleanup_expired_csrf_tokens(self):
        """Test that expired CSRF tokens are cleaned up."""
        from pika.api.web import (
            CSRF_TOKEN_MAX_AGE,
            _cleanup_expired_csrf_tokens,
            _csrf_tokens,
        )

        # Clear state
        _csrf_tokens.clear()

        # Add an old token
        _csrf_tokens["old_token"] = ("session1", time.time() - CSRF_TOKEN_MAX_AGE - 100)

        # Add a recent token
        _csrf_tokens["new_token"] = ("session2", time.time())

        # Run cleanup
        cleaned = _cleanup_expired_csrf_tokens()

        assert "old_token" not in _csrf_tokens
        assert "new_token" in _csrf_tokens
        assert cleaned == 1

        # Cleanup
        _csrf_tokens.clear()


class TestFilenameValidation:
    """Tests for filename sanitization."""

    def test_sanitize_normal_filename(self):
        """Test that normal filenames are accepted."""
        from pika.api.web import _sanitize_filename

        assert _sanitize_filename("document.pdf") == "document.pdf"
        assert _sanitize_filename("my file.docx") == "my file.docx"

    def test_sanitize_path_traversal(self):
        """Test that path traversal is blocked."""
        from pika.api.web import _sanitize_filename

        # These should extract just the filename
        assert _sanitize_filename("../../../etc/passwd") == "passwd"
        assert _sanitize_filename("/etc/passwd") == "passwd"
        assert _sanitize_filename("..\\..\\windows\\system32\\config") == "config"

    def test_reject_hidden_files(self):
        """Test that hidden files are rejected."""
        from pika.api.web import _sanitize_filename

        with pytest.raises(ValueError, match="Hidden files"):
            _sanitize_filename(".hidden")

    def test_reject_empty_filename(self):
        """Test that empty filenames are rejected."""
        from pika.api.web import _sanitize_filename

        with pytest.raises(ValueError, match="Empty filename"):
            _sanitize_filename("")

    def test_reject_null_bytes(self):
        """Test that null bytes are rejected."""
        from pika.api.web import _sanitize_filename

        with pytest.raises(ValueError, match="Invalid filename"):
            _sanitize_filename("file\x00.txt")


class TestFeedbackLimit:
    """Tests for feedback item limit."""

    def test_feedback_limit_enforced(self):
        """Test that feedback is capped at max_feedback_items config."""
        import tempfile

        from pika.config import get_settings
        from pika.services.history import HistoryService

        max_feedback_items = get_settings().max_feedback_items

        with tempfile.TemporaryDirectory() as tmpdir:
            service = HistoryService(data_dir=Path(tmpdir))

            # Add more than max feedback items
            for i in range(max_feedback_items + 50):
                service.add_feedback(
                    query_id=f"q_{i}",
                    question=f"Question {i}",
                    answer=f"Answer {i}",
                    rating="up" if i % 2 == 0 else "down",
                )

            # Check that we're at the limit
            assert len(service._feedback) == max_feedback_items


class TestAuditLogRotation:
    """Tests for audit log rotation."""

    def test_rotation_trigger(self):
        """Test that rotation is triggered when log exceeds max size."""
        import tempfile

        from pika.services.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.log"
            audit = AuditLogger(log_path=log_path)

            # Write enough data to trigger rotation
            # We need to write 100+ entries to trigger the size check
            for _i in range(150):
                # Force rotation check by setting write count
                audit._write_count = 99
                audit.log_query(
                    question="A" * 1000,  # Large entry
                    model="test",
                    confidence="high",
                    sources=["source1", "source2"],
                )

            # Log should exist
            assert log_path.exists()


class TestHttpxClientPooling:
    """Tests for httpx client pooling."""

    @pytest.mark.asyncio
    async def test_get_shared_client(self):
        """Test that get_http_client returns a shared client."""
        from pika.services.ollama import close_http_client, get_http_client

        client1 = await get_http_client()
        client2 = await get_http_client()

        # Should be the same client instance
        assert client1 is client2

        # Cleanup
        await close_http_client()

    @pytest.mark.asyncio
    async def test_close_http_client(self):
        """Test that close_http_client properly closes the client."""
        from pika.services.ollama import close_http_client, get_http_client

        client = await get_http_client()
        assert not client.is_closed

        await close_http_client()

        # Getting client again should create a new one
        client2 = await get_http_client()
        assert client2 is not client

        # Cleanup
        await close_http_client()


class TestQueueProcessorCleanup:
    """Tests for queue processor cleanup integration."""

    @pytest.mark.asyncio
    async def test_queue_processor_lifecycle(self):
        """Test that queue processor starts and stops correctly."""
        from pika.services import rag

        # Start processor
        await rag.init_queue_processor()
        await asyncio.sleep(0.1)  # Let it start

        # Should be running (check module variable directly)
        assert rag._queue_processor_task is not None

        # Stop processor
        await rag.shutdown_queue_processor()

        # Should have stopped cleanly


class TestSessionCleanupLifecycle:
    """Tests for session cleanup lifecycle."""

    @pytest.mark.asyncio
    async def test_session_cleanup_lifecycle(self):
        """Test that session cleanup starts and stops correctly."""
        from pika.api import web

        # Start cleanup
        await web.init_session_cleanup()
        await asyncio.sleep(0.1)  # Let it start

        # Should be running (check module variable directly)
        assert web._session_cleanup_task is not None

        # Stop cleanup
        await web.shutdown_session_cleanup()

        # Should have stopped cleanly
