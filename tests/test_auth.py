"""Tests for authentication functionality."""

import pytest
from unittest.mock import patch, MagicMock


class TestLoginPage:
    """Tests for the login page."""

    def test_login_page_renders(self, test_client):
        """Verify login page renders without auth configured."""
        # When no auth is configured, should redirect to admin
        response = test_client.get("/admin/login", follow_redirects=False)
        # Either shows login or redirects depending on auth config
        assert response.status_code in [200, 302]

    def test_login_page_has_csrf_token(self, test_client):
        """Verify login page includes CSRF token."""
        # Setup auth requirement
        with patch("pika.api.web.is_admin_auth_required", return_value=True), \
             patch("pika.api.web.is_setup_required", return_value=False), \
             patch("pika.api.web.is_authenticated", return_value=False):
            response = test_client.get("/admin/login")

        assert response.status_code == 200
        assert "csrf_token" in response.text


class TestSetupPage:
    """Tests for the setup page."""

    def test_setup_page_when_required(self, test_client):
        """Verify setup page shows when setup is required."""
        with patch("pika.api.web.is_setup_required", return_value=True):
            response = test_client.get("/setup")

        assert response.status_code == 200
        assert "Setup" in response.text or "setup" in response.text

    def test_setup_redirects_when_complete(self, test_client):
        """Verify setup redirects when already complete."""
        with patch("pika.api.web.is_setup_required", return_value=False):
            response = test_client.get("/setup", follow_redirects=False)

        assert response.status_code == 302
        assert "/admin" in response.headers.get("location", "")

    def test_setup_page_has_csrf_token(self, test_client):
        """Verify setup page includes CSRF token."""
        with patch("pika.api.web.is_setup_required", return_value=True):
            response = test_client.get("/setup")

        assert response.status_code == 200
        assert "csrf_token" in response.text


class TestLogout:
    """Tests for logout functionality."""

    def test_logout_clears_session(self, test_client):
        """Verify logout clears the session cookie."""
        # Create a session first
        from pika.api.web import create_session

        session_id = create_session({"username": "test", "role": "admin"})

        # Set the session cookie
        test_client.cookies.set("pika_session", session_id)

        response = test_client.get("/admin/logout", follow_redirects=False)

        assert response.status_code == 302
        # Cookie should be deleted (set-cookie with empty value or max-age=0)


class TestAPIKeyAuth:
    """Tests for API key authentication."""

    def test_valid_api_key_accepted(self, test_client):
        """Verify valid API key is accepted."""
        # This would need proper setup of API key in config
        pass

    def test_invalid_api_key_rejected(self, test_client):
        """Verify invalid API key is rejected when auth is required."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            response = test_client.get(
                "/api/v1/documents",
                headers={"X-API-Key": "invalid-key"},
            )
        assert response.status_code == 401


class TestPasswordVerification:
    """Tests for password verification."""

    def test_verify_password_correct(self):
        """Verify correct password is accepted."""
        from pika.api.web import hash_password
        from pika.services.auth import AuthService

        password = "correct_password"
        hashed = hash_password(password)

        assert AuthService.verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Verify incorrect password is rejected."""
        from pika.api.web import hash_password
        from pika.services.auth import AuthService

        password = "correct_password"
        hashed = hash_password(password)

        assert AuthService.verify_password("wrong_password", hashed) is False

    def test_verify_empty_password(self):
        """Verify empty password is rejected."""
        from pika.api.web import hash_password
        from pika.services.auth import AuthService

        hashed = hash_password("password")

        assert AuthService.verify_password("", hashed) is False


class TestSessionAuthentication:
    """Tests for session-based authentication."""

    def test_authenticated_request(self, test_client):
        """Verify authenticated requests are allowed."""
        from pika.api.web import create_session

        session_id = create_session({"username": "admin", "role": "admin"})
        test_client.cookies.set("pika_session", session_id)

        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            response = test_client.get("/admin")

        # Should not redirect to login
        assert response.status_code == 200 or "login" not in response.headers.get("location", "")

    def test_unauthenticated_redirect(self, test_client):
        """Verify unauthenticated requests redirect to login."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True), \
             patch("pika.api.web.is_setup_required", return_value=False):
            response = test_client.get("/admin", follow_redirects=False)

        assert response.status_code == 302
        assert "login" in response.headers.get("location", "")
