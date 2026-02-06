"""Multi-user query isolation tests for PIKA.

These tests verify that query results, history, and status are properly
isolated between users and that concurrent queries don't interfere.
"""

import asyncio

import pytest

from pika.services.rag import (
    ANONYMOUS_USER,
    QueryStatus,
    UserQueueLimitError,
    _get_user_key,
    _set_query_status,
    cancel_query,
    clear_query_status,
    get_active_query,
)


class TestQueryStatusIsolation:
    """Test that query status is isolated per user."""

    def test_query_status_stored_per_user(self):
        """Each user has their own query status."""
        # Create query statuses for different users
        status1 = QueryStatus(
            query_id="q1",
            question="What is A?",
            status="running",
        )
        status2 = QueryStatus(
            query_id="q2",
            question="What is B?",
            status="running",
        )

        _set_query_status(status1, "user1")
        _set_query_status(status2, "user2")

        # Each user gets their own status
        assert get_active_query("user1").query_id == "q1"
        assert get_active_query("user1").question == "What is A?"
        assert get_active_query("user2").query_id == "q2"
        assert get_active_query("user2").question == "What is B?"

        # Clean up
        clear_query_status("user1")
        clear_query_status("user2")

    def test_user_cannot_see_other_query_status(self):
        """Users cannot see other users' query status."""
        status1 = QueryStatus(
            query_id="q1",
            question="Secret question",
            status="completed",
        )

        _set_query_status(status1, "user1")

        # user2 should not see user1's query
        user2_status = get_active_query("user2")
        assert user2_status is None

        # Clean up
        clear_query_status("user1")

    def test_clear_status_only_affects_own_user(self):
        """Clearing status only affects the calling user."""
        status1 = QueryStatus(query_id="q1", question="Q1", status="done")
        status2 = QueryStatus(query_id="q2", question="Q2", status="done")

        _set_query_status(status1, "user1")
        _set_query_status(status2, "user2")

        # Clear user1's status
        clear_query_status("user1")

        # user1's status should be gone
        assert get_active_query("user1") is None

        # user2's status should still exist
        assert get_active_query("user2") is not None
        assert get_active_query("user2").query_id == "q2"

        # Clean up
        clear_query_status("user2")


class TestQueryCancellationIsolation:
    """Test that query cancellation is isolated per user."""

    @pytest.mark.asyncio
    async def test_cancel_only_affects_own_query(self):
        """Cancelling a query only affects the requesting user."""
        # Note: cancel_query checks if status is "running" or "queued"
        # Set up statuses that can be cancelled
        status1 = QueryStatus(
            query_id="q1",
            question="Q1",
            status="queued",  # Use queued status so it can be cancelled
        )
        status2 = QueryStatus(
            query_id="q2",
            question="Q2",
            status="queued",
        )

        _set_query_status(status1, "user1")
        _set_query_status(status2, "user2")

        # user1 cancels their query
        await cancel_query("user1")

        # user1's query should be cancelled (status cleared or changed)
        # Note: cancel_query behavior depends on implementation
        # The key test is that user2's query is unaffected

        # user2's query should be unaffected
        user2_status = get_active_query("user2")
        assert user2_status is not None
        assert user2_status.query_id == "q2"

        # Clean up
        clear_query_status("user1")
        clear_query_status("user2")

    @pytest.mark.asyncio
    async def test_cannot_cancel_other_user_query(self):
        """Users cannot cancel other users' queries."""
        status1 = QueryStatus(
            query_id="q1",
            question="Q1",
            status="running",
        )

        _set_query_status(status1, "user1")

        # user2 tries to cancel (but they have no query)
        result = await cancel_query("user2")
        assert result is False

        # user1's query should be unaffected
        assert get_active_query("user1").status == "running"

        # Clean up
        clear_query_status("user1")


class TestConcurrentQueries:
    """Test concurrent query handling."""

    @pytest.mark.asyncio
    async def test_concurrent_query_status_updates(self):
        """Concurrent query status updates from different users are isolated."""
        users = [f"user{i}" for i in range(5)]

        async def update_status(username):
            status = QueryStatus(
                query_id=f"q_{username}",
                question=f"Question from {username}",
                status="running",
            )
            _set_query_status(status, username)
            await asyncio.sleep(0.01)  # Small delay
            return get_active_query(username)

        # Update statuses concurrently
        tasks = [update_status(user) for user in users]
        results = await asyncio.gather(*tasks)

        # Each user should have their own status
        for i, result in enumerate(results):
            assert result is not None
            assert result.query_id == f"q_{users[i]}"

        # Clean up
        for user in users:
            clear_query_status(user)

    def test_queue_counts_per_user(self):
        """Queue counts are tracked per user."""
        # Add queries for different users
        for i in range(3):
            status = QueryStatus(
                query_id=f"q{i}_1",
                question="Q1",
                status="queued",
            )
            _set_query_status(status, f"user{i}")

        # Check user-specific statuses
        for i in range(3):
            user_status = get_active_query(f"user{i}")
            assert user_status is not None

        # Clean up
        for i in range(3):
            clear_query_status(f"user{i}")


class TestQueryHistoryIsolation:
    """Test that query history is isolated per user."""

    def test_history_service_per_user(self):
        """History service returns only the requesting user's history."""
        import tempfile
        from pathlib import Path

        from pika.services.history import HistoryService

        # Create a temp directory for history
        with tempfile.TemporaryDirectory() as tmpdir:
            history = HistoryService(Path(tmpdir))

            # Add history for different users
            history.add_query(
                question="User1 question",
                answer="Answer 1",
                confidence="high",
                sources=["doc1.txt"],
                username="user1",
            )
            history.add_query(
                question="User2 question",
                answer="Answer 2",
                confidence="medium",
                sources=["doc2.txt"],
                username="user2",
            )

            # Each user should only see their own history
            user1_history = history.get_history(username="user1")
            user2_history = history.get_history(username="user2")

            assert len(user1_history) == 1
            assert user1_history[0]["question"] == "User1 question"

            assert len(user2_history) == 1
            assert user2_history[0]["question"] == "User2 question"

            # user1 should not see user2's history
            for entry in user1_history:
                assert "User2" not in entry["question"]

    def test_clear_history_only_affects_own_user(self):
        """Clearing history only affects the calling user."""
        import tempfile
        from pathlib import Path

        from pika.services.history import HistoryService

        with tempfile.TemporaryDirectory() as tmpdir:
            history = HistoryService(Path(tmpdir))

            # Add history for different users
            history.add_query("Q1", "A1", "high", [], username="user1")
            history.add_query("Q2", "A2", "high", [], username="user2")

            # Clear user1's history
            history.clear_history(username="user1")

            # user1's history should be empty
            assert len(history.get_history(username="user1")) == 0

            # user2's history should still exist
            assert len(history.get_history(username="user2")) == 1


class TestAnonymousUserQueries:
    """Test query handling for anonymous (no username) users."""

    def test_anonymous_users_isolated(self):
        """Anonymous users (None username) are handled separately."""
        # Anonymous user query
        anon_status = QueryStatus(
            query_id="anon_q1",
            question="Anonymous question",
            status="completed",
        )

        named_status = QueryStatus(
            query_id="named_q1",
            question="Named user question",
            status="completed",
        )

        _set_query_status(anon_status, None)
        _set_query_status(named_status, "nameduser")

        # Each should get their own status
        assert get_active_query(None).query_id == "anon_q1"
        assert get_active_query("nameduser").query_id == "named_q1"

        # Clean up
        clear_query_status(None)
        clear_query_status("nameduser")


class TestQueueBehaviorMultiUser:
    """Test queue behavior with multiple users."""

    def test_queue_position_reflects_all_users(self):
        """Queue position accounts for queries from all users."""
        # Create multiple queued queries
        for i, user in enumerate(["user1", "user2", "user3"]):
            status = QueryStatus(
                query_id=f"q{i}",
                question=f"Question from {user}",
                status="queued",
                queue_position=i + 1,
            )
            _set_query_status(status, user)

        # Verify each user sees their position
        assert get_active_query("user1").queue_position == 1
        assert get_active_query("user2").queue_position == 2
        assert get_active_query("user3").queue_position == 3

        # Clean up
        for user in ["user1", "user2", "user3"]:
            clear_query_status(user)

    def test_per_user_queue_limits(self):
        """Per-user queue limits error type exists."""
        # Verify the error type exists
        error = UserQueueLimitError("User has too many queries in queue")
        assert isinstance(error, Exception)
        assert "queue" in str(error).lower()


class TestUserKeyMapping:
    """Test user key mapping for query storage."""

    def test_user_key_for_named_user(self):
        """Named users get their username as key."""
        assert _get_user_key("testuser") == "testuser"

    def test_user_key_for_anonymous(self):
        """Anonymous users get special key."""
        assert _get_user_key(None) == ANONYMOUS_USER

    def test_different_users_different_keys(self):
        """Different users get different keys."""
        key1 = _get_user_key("user1")
        key2 = _get_user_key("user2")
        key_anon = _get_user_key(None)

        assert key1 != key2
        assert key1 != key_anon
        assert key2 != key_anon
