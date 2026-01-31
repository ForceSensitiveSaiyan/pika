"""Concurrency safety tests for PIKA.

These tests verify that locks and synchronization mechanisms work correctly
under concurrent access to prevent race conditions and data corruption.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from pika.api.web import (
    _sessions,
    _sessions_lock,
    _csrf_tokens,
    _csrf_lock,
    create_session,
    delete_session,
    get_session,
    generate_csrf_token,
    validate_csrf_token,
)


class TestSessionLockSafety:
    """Test session lock prevents race conditions."""

    def test_concurrent_session_creation_no_data_loss(self):
        """Concurrent session creation doesn't lose any sessions."""
        num_sessions = 100
        created_sessions = []
        errors = []

        def create_session_safe(i):
            try:
                sid = create_session({"username": f"user{i}", "role": "user"})
                created_sessions.append(sid)
                return sid
            except Exception as e:
                errors.append(e)
                return None

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(create_session_safe, i) for i in range(num_sessions)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0, f"Errors during creation: {errors}"
        assert len(created_sessions) == num_sessions
        assert len(set(created_sessions)) == num_sessions  # All unique

        # Verify all sessions are accessible
        for sid in created_sessions:
            assert get_session(sid) is not None

    def test_concurrent_session_deletion_no_double_delete_error(self):
        """Concurrent deletion of same session doesn't raise errors."""
        session_id = create_session({"username": "testuser", "role": "user"})
        errors = []
        delete_count = [0]  # Use list to allow modification in nested function

        def delete_session_safe():
            try:
                delete_session(session_id)
                delete_count[0] += 1
            except Exception as e:
                errors.append(e)

        # Try to delete the same session 10 times concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(delete_session_safe) for _ in range(10)]
            for future in as_completed(futures):
                future.result()

        # Should not have any errors (delete_session handles missing sessions gracefully)
        assert len(errors) == 0

    def test_concurrent_read_write_no_corruption(self):
        """Concurrent reads and writes don't corrupt session data."""
        session_id = create_session({"username": "testuser", "role": "user", "counter": 0})
        errors = []
        read_results = []

        def read_session():
            try:
                data = get_session(session_id)
                if data:
                    read_results.append(data.get("username"))
            except Exception as e:
                errors.append(e)

        def write_session(i):
            try:
                # Note: In real code, session modification would need locking
                # This test verifies reads during writes don't crash
                with _sessions_lock:
                    if session_id in _sessions:
                        _sessions[session_id]["counter"] = i
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(50):
                futures.append(executor.submit(read_session))
                futures.append(executor.submit(write_session, i))

            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0
        # All reads should have gotten valid username
        assert all(r == "testuser" for r in read_results if r is not None)


class TestCSRFLockSafety:
    """Test CSRF token lock prevents race conditions."""

    def test_concurrent_csrf_generation_unique(self):
        """Concurrent CSRF token generation produces unique tokens."""
        num_tokens = 100
        tokens = []
        errors = []

        def generate_token():
            try:
                token = generate_csrf_token()
                tokens.append(token)
                return token
            except Exception as e:
                errors.append(e)
                return None

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(generate_token) for _ in range(num_tokens)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0
        assert len(tokens) == num_tokens
        assert len(set(tokens)) == num_tokens  # All unique

    def test_csrf_token_single_use_under_concurrency(self):
        """CSRF token can only be used once even under concurrent validation."""
        token = generate_csrf_token()
        validation_results = []
        errors = []

        def validate_token():
            try:
                result = validate_csrf_token(token)
                validation_results.append(result)
                return result
            except Exception as e:
                errors.append(e)
                return False

        # Try to validate the same token 10 times concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(validate_token) for _ in range(10)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0
        # Exactly one validation should succeed
        assert validation_results.count(True) == 1
        assert validation_results.count(False) == 9


class TestQueryLockSafety:
    """Test query-related lock safety."""

    def test_concurrent_query_status_updates(self):
        """Concurrent query status updates don't corrupt data."""
        from pika.services.rag import (
            _set_query_status,
            get_active_query,
            clear_query_status,
            QueryStatus,
        )

        errors = []
        usernames = [f"user{i}" for i in range(20)]

        def update_status(username):
            try:
                status = QueryStatus(
                    query_id=f"q_{username}",
                    question=f"Question from {username}",
                    status="running",
                )
                _set_query_status(status, username)
            except Exception as e:
                errors.append(e)

        def read_status(username):
            try:
                status = get_active_query(username)
                if status:
                    _ = status.query_id  # Access data
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for username in usernames:
                futures.append(executor.submit(update_status, username))
                futures.append(executor.submit(read_status, username))

            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0

        # Clean up
        for username in usernames:
            clear_query_status(username)


class TestIndexLockSafety:
    """Test index operation lock safety."""

    def test_concurrent_index_check_consistent(self):
        """Concurrent checks of indexing status are consistent."""
        from pika.services.rag import (
            is_indexing_running,
            get_active_index,
        )

        errors = []
        results = []

        def check_index_status():
            try:
                running = is_indexing_running()
                active = get_active_index()
                results.append((running, active))
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(check_index_status) for _ in range(50)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0


class TestAuditLogLockSafety:
    """Test audit log write lock safety."""

    def test_concurrent_audit_writes_no_corruption(self):
        """Concurrent audit log writes don't corrupt the log file."""
        from pika.services.audit import AuditLogger
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = Path(f.name)

        try:
            audit = AuditLogger(log_path)
            errors = []
            num_writes = 100

            def write_log(i):
                try:
                    audit.log_admin_action(f"action_{i}", {"index": i})
                except Exception as e:
                    errors.append(e)

            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(write_log, i) for i in range(num_writes)]
                for future in as_completed(futures):
                    future.result()

            assert len(errors) == 0

            # Verify all entries were written
            logs = audit.get_recent_logs(limit=num_writes + 10)
            assert len(logs) == num_writes

            # Verify no corruption (all entries parse correctly)
            for log in logs:
                assert "action" in log
                assert "timestamp" in log

        finally:
            if log_path.exists():
                log_path.unlink()


class TestHistoryLockSafety:
    """Test history file lock safety."""

    def test_concurrent_history_writes(self):
        """Concurrent history writes don't corrupt data."""
        from pika.services.history import HistoryService
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            history = HistoryService(Path(tmpdir))
            errors = []
            num_entries = 50

            def add_entry(i):
                try:
                    history.add_query(
                        question=f"Question {i}",
                        answer=f"Answer {i}",
                        confidence="high",
                        sources=[f"source{i}.txt"],
                        username=f"user{i % 5}",  # 5 different users
                    )
                except Exception as e:
                    errors.append(e)

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(add_entry, i) for i in range(num_entries)]
                for future in as_completed(futures):
                    future.result()

            assert len(errors) == 0

            # Verify all entries exist
            all_entries = history.get_history(limit=num_entries + 10)
            assert len(all_entries) == num_entries


class TestAsyncSafety:
    """Test async operation safety."""

    @pytest.mark.asyncio
    async def test_async_counter_with_lock(self):
        """Asyncio locks prevent race conditions in async code."""
        lock = asyncio.Lock()
        counter = [0]
        errors = []

        async def increment():
            try:
                async with lock:
                    current = counter[0]
                    await asyncio.sleep(0.001)  # Simulate async work
                    counter[0] = current + 1
            except Exception as e:
                errors.append(e)

        # Run many increments concurrently
        await asyncio.gather(*[increment() for _ in range(100)])

        assert len(errors) == 0
        assert counter[0] == 100  # Should be exactly 100 with proper locking


class TestCacheThreadSafety:
    """Test cache operations are thread-safe."""

    def test_concurrent_cache_access_no_errors(self):
        """Concurrent cache access doesn't crash."""
        from pika.services.rag import get_active_query, clear_query_status

        errors = []
        results = []

        def access_cache(i):
            try:
                # Try to access query cache for various users
                result = get_active_query(f"user{i}")
                results.append(result)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_cache, i) for i in range(50)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0
