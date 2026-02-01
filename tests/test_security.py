"""Security-related tests for PIKA."""

import secrets
import time

import pytest


class TestPasswordHashing:
    """Tests for password hashing functionality."""

    def test_hash_password_uses_bcrypt(self):
        """Verify password hashing uses bcrypt format."""
        from pika.api.web import hash_password

        hashed = hash_password("test_password")

        # bcrypt hashes start with $2b$, $2a$, or $2y$
        assert hashed.startswith(("$2b$", "$2a$", "$2y$"))

    def test_hash_password_unique_per_call(self):
        """Verify each hash is unique due to salt."""
        from pika.api.web import hash_password

        hash1 = hash_password("same_password")
        hash2 = hash_password("same_password")

        assert hash1 != hash2

    def test_verify_bcrypt_password(self):
        """Verify bcrypt password verification works."""
        from pika.api.web import hash_password
        from pika.services.auth import AuthService

        password = "test_password_123"
        hashed = hash_password(password)

        assert AuthService.verify_password(password, hashed) is True
        assert AuthService.verify_password("wrong_password", hashed) is False

    def test_verify_legacy_sha256_password(self):
        """Verify legacy SHA-256 passwords still work."""
        import hashlib
        from pika.services.auth import AuthService

        password = "legacy_password"
        # Simulate old SHA-256 hash
        legacy_hash = hashlib.sha256(password.encode()).hexdigest()

        assert AuthService.verify_password(password, legacy_hash) is True
        assert AuthService.verify_password("wrong_password", legacy_hash) is False


class TestCSRFProtection:
    """Tests for CSRF token functionality."""

    def test_generate_csrf_token_unique(self):
        """Verify CSRF tokens are unique."""
        from pika.api.web import generate_csrf_token

        token1 = generate_csrf_token()
        token2 = generate_csrf_token()

        assert token1 != token2

    def test_validate_csrf_token_valid(self):
        """Verify valid CSRF tokens are accepted."""
        from pika.api.web import generate_csrf_token, validate_csrf_token

        token = generate_csrf_token()

        assert validate_csrf_token(token) is True

    def test_validate_csrf_token_invalid(self):
        """Verify invalid CSRF tokens are rejected."""
        from pika.api.web import validate_csrf_token

        assert validate_csrf_token("invalid_token") is False
        assert validate_csrf_token(None) is False
        assert validate_csrf_token("") is False

    def test_csrf_token_single_use(self):
        """Verify CSRF tokens can only be used once."""
        from pika.api.web import generate_csrf_token, validate_csrf_token

        token = generate_csrf_token()

        # First validation should succeed
        assert validate_csrf_token(token) is True
        # Second validation should fail (token consumed)
        assert validate_csrf_token(token) is False


class TestSessionManagement:
    """Tests for session management."""

    def test_create_session(self):
        """Verify session creation works."""
        from pika.api.web import create_session, get_session

        session_id = create_session({"username": "test_user", "role": "admin"})

        assert session_id is not None
        session = get_session(session_id)
        assert session is not None
        assert session["username"] == "test_user"

    def test_delete_session(self):
        """Verify session deletion works."""
        from pika.api.web import create_session, get_session, delete_session

        session_id = create_session({"username": "test_user"})
        assert get_session(session_id) is not None

        delete_session(session_id)
        assert get_session(session_id) is None

    def test_session_expiration(self):
        """Verify sessions expire after max age."""
        from pika.api.web import (
            _sessions,
            _sessions_lock,
            get_session,
            SESSION_MAX_AGE,
        )

        # Create an expired session manually
        session_id = secrets.token_urlsafe(32)
        with _sessions_lock:
            _sessions[session_id] = {
                "username": "test",
                "created_at": time.time() - SESSION_MAX_AGE - 1,
            }

        # Should return None for expired session
        assert get_session(session_id) is None


class TestZipSlipPrevention:
    """Tests for Zip Slip vulnerability prevention."""

    def test_safe_extract_path_normal(self):
        """Verify normal paths are allowed."""
        from pathlib import Path
        from pika.api.web import _safe_extract_path

        base_dir = Path("/tmp/test")
        result = _safe_extract_path(base_dir, "subdir/file.txt")

        assert result is not None
        assert str(result).startswith(str(base_dir.resolve()))

    def test_safe_extract_path_traversal_blocked(self):
        """Verify path traversal attempts are blocked."""
        from pathlib import Path
        from pika.api.web import _safe_extract_path

        base_dir = Path("/tmp/test")

        # These should all return None (blocked)
        assert _safe_extract_path(base_dir, "../etc/passwd") is None
        assert _safe_extract_path(base_dir, "subdir/../../etc/passwd") is None
        assert _safe_extract_path(base_dir, "..\\..\\windows\\system32") is None


class TestRateLimiting:
    """Tests for rate limiting configuration."""

    def test_rate_limit_settings_exist(self):
        """Verify rate limit settings are configured."""
        from pika.config import get_settings

        settings = get_settings()

        assert hasattr(settings, "rate_limit_auth")
        assert hasattr(settings, "rate_limit_query")
        assert "minute" in settings.rate_limit_auth.lower() or "/" in settings.rate_limit_auth


class TestSecretGeneration:
    """Tests for secret generation."""

    def test_session_secret_auto_generated(self):
        """Verify session secret is auto-generated if not provided."""
        import os

        # Clear any existing env var
        original = os.environ.get("PIKA_SESSION_SECRET")
        os.environ["PIKA_SESSION_SECRET"] = ""

        try:
            # Force reload of settings
            from pika.config import Settings
            settings = Settings()

            assert settings.pika_session_secret != ""
            assert len(settings.pika_session_secret) >= 32
        finally:
            if original:
                os.environ["PIKA_SESSION_SECRET"] = original
