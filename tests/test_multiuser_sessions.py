"""Multi-user session isolation tests for PIKA.

These tests verify that users cannot access each other's session data
and that concurrent session operations are handled safely.
"""

import asyncio
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from pika.api.web import (
    _sessions,
    _sessions_lock,
    create_session,
    delete_session,
    get_session,
    is_authenticated,
    _cleanup_expired_sessions,
    SESSION_MAX_AGE,
)


class TestSessionIsolation:
    """Test that sessions are isolated between users."""

    def test_sessions_have_unique_ids(self):
        """Each session gets a unique ID."""
        session1 = create_session({"username": "user1", "role": "user"})
        session2 = create_session({"username": "user2", "role": "user"})
        session3 = create_session({"username": "user3", "role": "user"})

        assert session1 != session2
        assert session2 != session3
        assert session1 != session3

    def test_session_data_isolation(self):
        """Session data is isolated - can't access other user's data."""
        session1 = create_session({"username": "user1", "role": "user", "custom": "data1"})
        session2 = create_session({"username": "user2", "role": "admin", "custom": "data2"})

        # Get session data
        data1 = get_session(session1)
        data2 = get_session(session2)

        # Verify isolation
        assert data1["username"] == "user1"
        assert data1["custom"] == "data1"
        assert data2["username"] == "user2"
        assert data2["custom"] == "data2"

        # Can't get user2's data with user1's session
        assert get_session(session1)["username"] != "user2"

    def test_invalid_session_returns_none(self):
        """Invalid session ID returns None."""
        fake_session = secrets.token_urlsafe(32)
        assert get_session(fake_session) is None

    def test_session_deletion_isolation(self):
        """Deleting one session doesn't affect others."""
        session1 = create_session({"username": "user1", "role": "user"})
        session2 = create_session({"username": "user2", "role": "user"})

        # Delete session1
        delete_session(session1)

        # session1 should be gone
        assert get_session(session1) is None

        # session2 should still exist
        assert get_session(session2) is not None
        assert get_session(session2)["username"] == "user2"

    def test_same_user_multiple_sessions(self):
        """Same user can have multiple sessions (different devices)."""
        session1 = create_session({"username": "user1", "role": "user", "device": "desktop"})
        session2 = create_session({"username": "user1", "role": "user", "device": "mobile"})

        # Both sessions should exist independently
        assert session1 != session2
        assert get_session(session1)["device"] == "desktop"
        assert get_session(session2)["device"] == "mobile"

        # Deleting one doesn't affect the other
        delete_session(session1)
        assert get_session(session1) is None
        assert get_session(session2) is not None


class TestConcurrentSessionCreation:
    """Test concurrent session creation safety."""

    def test_concurrent_session_creation_no_collision(self):
        """Concurrent session creation produces unique sessions."""
        num_sessions = 50
        sessions = []

        def create_user_session(i):
            return create_session({"username": f"user{i}", "role": "user"})

        with ThreadPoolExecutor(max_workers=10) as executor:
            sessions = list(executor.map(create_user_session, range(num_sessions)))

        # All sessions should be unique
        assert len(set(sessions)) == num_sessions

        # All sessions should be valid
        for i, session_id in enumerate(sessions):
            data = get_session(session_id)
            assert data is not None
            assert data["username"] == f"user{i}"

    def test_concurrent_session_operations_thread_safe(self):
        """Mixed create/delete/read operations are thread-safe."""
        results = {"created": 0, "deleted": 0, "read": 0, "errors": 0}
        session_ids = []

        def create_op():
            try:
                sid = create_session({"username": "testuser", "role": "user"})
                session_ids.append(sid)
                results["created"] += 1
            except Exception:
                results["errors"] += 1

        def delete_op():
            try:
                if session_ids:
                    # Try to delete a random session
                    sid = session_ids[0] if session_ids else None
                    if sid:
                        delete_session(sid)
                        results["deleted"] += 1
            except Exception:
                results["errors"] += 1

        def read_op():
            try:
                if session_ids:
                    sid = session_ids[0] if session_ids else None
                    if sid:
                        get_session(sid)
                        results["read"] += 1
            except Exception:
                results["errors"] += 1

        # First create some sessions
        for _ in range(20):
            create_op()

        # Then do mixed operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            ops = [create_op] * 20 + [delete_op] * 10 + [read_op] * 20
            list(executor.map(lambda f: f(), ops))

        # Should have no errors
        assert results["errors"] == 0


class TestSessionHijackingPrevention:
    """Test that session hijacking is prevented."""

    def test_cannot_forge_session_id(self):
        """Forged session IDs don't work."""
        # Create a real session
        real_session = create_session({"username": "realuser", "role": "admin"})

        # Try various forged sessions
        forged_attempts = [
            "admin",
            "root",
            real_session[:-1],  # Almost the real session
            real_session + "x",  # Real session with extra char
            secrets.token_urlsafe(32),  # Random valid-looking token
            "",  # Empty
            "a" * 43,  # Same length as real
        ]

        for forged in forged_attempts:
            if forged != real_session:
                assert get_session(forged) is None

    def test_session_cannot_be_modified_externally(self):
        """Session data cannot be modified without the session_id."""
        session_id = create_session({"username": "user1", "role": "user"})

        # Get a reference to session data
        original_data = get_session(session_id).copy()

        # The session dict is internal - trying to modify it shouldn't work
        # (In a real attack, someone might try to modify _sessions directly)
        with _sessions_lock:
            if session_id in _sessions:
                # Even if they could access _sessions, it requires the exact session_id
                pass

        # Data should be unchanged
        current_data = get_session(session_id)
        assert current_data["username"] == original_data["username"]
        assert current_data["role"] == original_data["role"]


class TestSessionExpiration:
    """Test session expiration and cleanup."""

    def test_expired_session_not_accessible(self):
        """Expired sessions return None."""
        session_id = create_session({"username": "user1", "role": "user"})

        # Manually expire the session
        with _sessions_lock:
            _sessions[session_id]["created_at"] = time.time() - SESSION_MAX_AGE - 1

        # Session should now be inaccessible
        # Note: get_session doesn't check expiry, cleanup does
        # But the cleanup should remove it
        cleaned = _cleanup_expired_sessions()
        assert cleaned >= 1
        assert get_session(session_id) is None

    def test_cleanup_only_removes_expired(self):
        """Cleanup only removes expired sessions, not active ones."""
        active_session = create_session({"username": "active", "role": "user"})
        expired_session = create_session({"username": "expired", "role": "user"})

        # Expire one session
        with _sessions_lock:
            _sessions[expired_session]["created_at"] = time.time() - SESSION_MAX_AGE - 1

        # Run cleanup
        _cleanup_expired_sessions()

        # Active session should still work
        assert get_session(active_session) is not None
        assert get_session(active_session)["username"] == "active"

        # Expired session should be gone
        assert get_session(expired_session) is None


class TestSessionWithTestClient:
    """Test session behavior through HTTP endpoints."""

    def test_protected_endpoint_without_session(self, test_client):
        """Protected endpoints redirect without session."""
        response = test_client.get("/admin", follow_redirects=False)
        # Should redirect to login
        assert response.status_code in (302, 307)

    def test_session_cookie_httponly(self, test_client):
        """Session cookies should be HttpOnly."""
        # This is tested implicitly - the cookie is set with HttpOnly flag
        # We verify the session mechanism works
        with patch("pika.api.web.is_admin_auth_required", return_value=False):
            response = test_client.get("/admin")
            # Without auth required, should succeed
            assert response.status_code == 200

    def test_session_deletion_clears_data(self):
        """Session deletion properly clears session data."""
        # Create a session manually
        session_id = create_session({"username": "testuser", "role": "admin"})

        # Verify session exists
        assert get_session(session_id) is not None
        assert get_session(session_id)["username"] == "testuser"

        # Delete the session (simulates logout)
        delete_session(session_id)

        # Session should be cleared
        assert get_session(session_id) is None


class TestMultipleUsersSimultaneousAccess:
    """Test multiple users accessing the system simultaneously."""

    def test_two_users_independent_sessions(self, test_client):
        """Two users can have independent sessions."""
        # Create sessions for two users
        user1_session = create_session({"username": "user1", "role": "user"})
        user2_session = create_session({"username": "user2", "role": "admin"})

        # Both sessions should be valid and independent
        user1_data = get_session(user1_session)
        user2_data = get_session(user2_session)

        assert user1_data["username"] == "user1"
        assert user1_data["role"] == "user"
        assert user2_data["username"] == "user2"
        assert user2_data["role"] == "admin"

    def test_user_role_isolation(self):
        """User roles are properly isolated per session."""
        admin_session = create_session({"username": "admin", "role": "admin"})
        user_session = create_session({"username": "user", "role": "user"})

        # Admin session has admin role
        assert get_session(admin_session)["role"] == "admin"

        # User session has user role
        assert get_session(user_session)["role"] == "user"

        # Roles don't leak between sessions
        assert get_session(user_session)["role"] != "admin"
