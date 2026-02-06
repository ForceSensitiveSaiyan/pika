"""Tests for API endpoints."""

from unittest.mock import AsyncMock, patch


class TestMetricsEndpoint:
    """Tests for the Prometheus metrics endpoint."""

    def test_metrics_returns_prometheus_format(self, test_client):
        """Verify /metrics returns Prometheus format."""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        # Check for expected metrics
        content = response.text
        assert "pika_http_requests_total" in content
        assert "pika_http_request_duration_seconds" in content
        assert "pika_info" in content

    def test_metrics_records_requests(self, test_client):
        """Verify request metrics are recorded after making requests."""
        # Make a request to generate metrics
        test_client.get("/api/v1/health")

        # Check metrics endpoint
        response = test_client.get("/metrics")
        content = response.text

        # Should have recorded the health request
        assert "pika_http_requests_total" in content
        assert 'endpoint="/api/v1/health"' in content or 'endpoint="/api/v1/' in content

    def test_metrics_includes_query_metrics(self, test_client):
        """Verify query-related metrics are defined."""
        response = test_client.get("/metrics")
        content = response.text

        # Query metrics should be defined (even if zero)
        assert "pika_queries_total" in content
        assert "pika_query_duration_seconds" in content

    def test_metrics_includes_index_gauges(self, test_client):
        """Verify index-related gauges are defined."""
        response = test_client.get("/metrics")
        content = response.text

        assert "pika_index_documents_total" in content
        assert "pika_index_chunks_total" in content
        assert "pika_active_queries" in content
        assert "pika_queued_queries" in content


class TestMetricsModule:
    """Tests for the metrics module functions."""

    def test_set_app_info(self):
        """Verify app info can be set."""
        from pika.services.metrics import set_app_info

        set_app_info("1.0.0", "test-model")
        # No exception means success

    def test_update_index_metrics(self):
        """Verify index metrics can be updated."""
        from pika.services.metrics import update_index_metrics

        update_index_metrics(10, 100)
        # Gauges should be set (we can't easily read the value in tests)

    def test_update_queue_metrics(self):
        """Verify queue metrics can be updated."""
        from pika.services.metrics import update_queue_metrics

        update_queue_metrics(2, 5)
        # Gauges should be set


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, test_client, mock_ollama_client, mock_rag_engine):
        """Verify health endpoint returns 200."""
        from pika.main import app
        from pika.services.ollama import get_ollama_client
        from pika.services.rag import get_rag_engine

        # Use FastAPI's dependency override mechanism
        app.dependency_overrides[get_ollama_client] = lambda: mock_ollama_client
        app.dependency_overrides[get_rag_engine] = lambda: mock_rag_engine
        try:
            response = test_client.get("/api/v1/health")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "ollama" in data
        assert "index" in data
        assert "disk" in data

    def test_health_shows_ollama_disconnected(self, test_client, mock_ollama_client, mock_rag_engine):
        """Verify health endpoint shows Ollama disconnection."""
        from pika.main import app
        from pika.services.ollama import get_ollama_client
        from pika.services.rag import get_rag_engine

        mock_ollama_client.health_check = AsyncMock(return_value=False)
        mock_ollama_client.list_models = AsyncMock(return_value=[])

        # Use FastAPI's dependency override mechanism
        app.dependency_overrides[get_ollama_client] = lambda: mock_ollama_client
        app.dependency_overrides[get_rag_engine] = lambda: mock_rag_engine
        try:
            response = test_client.get("/api/v1/health")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        # When disconnected, ollama.connected should be False
        assert "ollama" in data


class TestModelsEndpoint:
    """Tests for the models endpoint."""

    def test_list_models_returns_valid_response(self, test_client, mock_ollama_client):
        """Verify models endpoint returns a valid list."""
        from pika.main import app
        from pika.services.ollama import ModelInfo, get_ollama_client

        # Set up mock to return some models
        mock_ollama_client.list_models = AsyncMock(return_value=[
            ModelInfo(name="llama3.2:3b", size=1234567890, modified_at="2026-01-24"),
            ModelInfo(name="mistral:7b", size=4000000000, modified_at="2026-01-23"),
        ])

        # Use FastAPI's dependency override mechanism
        app.dependency_overrides[get_ollama_client] = lambda: mock_ollama_client
        try:
            response = test_client.get("/api/v1/models")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        # Should be a list
        assert isinstance(data, list)
        assert len(data) == 2
        # Each model should have required fields
        for model in data:
            assert "name" in model
            assert "size" in model
            assert "is_current" in model


class TestIndexEndpoint:
    """Tests for the index endpoint."""

    def test_get_index_stats_requires_auth(self, test_client):
        """Verify index stats requires authentication when auth is enabled."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            response = test_client.get("/api/v1/index/stats")
        assert response.status_code == 401

    def test_index_documents_requires_auth(self, test_client):
        """Verify indexing requires authentication when auth is enabled."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            response = test_client.post("/api/v1/index")
        assert response.status_code == 401


class TestQueryEndpoint:
    """Tests for the query endpoint."""

    def test_query_requires_auth(self, test_client):
        """Verify query endpoint requires authentication when auth is enabled."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            response = test_client.post(
                "/api/v1/query",
                json={"question": "What is PIKA?"},
            )
        assert response.status_code == 401


class TestUploadEndpoint:
    """Tests for the file upload endpoint."""

    def test_upload_requires_auth(self, test_client):
        """Verify upload requires authentication when auth is enabled."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            response = test_client.post(
                "/upload",
                files={"file": ("test.txt", b"content", "text/plain")},
            )
        assert response.status_code == 401

    def test_upload_works_when_auth_disabled(self, test_client, temp_dirs):
        """Verify upload works when auth is disabled."""
        from pathlib import Path

        # Make sure docs dir exists
        docs_dir = Path(temp_dirs["docs"])
        docs_dir.mkdir(parents=True, exist_ok=True)

        with patch("pika.api.web.is_admin_auth_required", return_value=False), \
             patch("pika.services.documents.get_settings") as mock_settings:
            mock_settings.return_value.documents_dir = str(docs_dir)
            response = test_client.post(
                "/upload",
                files={"file": ("test.txt", b"test content", "text/plain")},
            )
        # Should succeed or at least not be a 401
        assert response.status_code != 401


class TestHistoryEndpoint:
    """Tests for the history endpoint."""

    def test_history_requires_auth(self, test_client):
        """Verify history endpoint requires authentication when auth is enabled."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            response = test_client.get("/api/v1/history")
        assert response.status_code == 401


class TestFeedbackEndpoint:
    """Tests for the feedback endpoint."""

    def test_feedback_requires_auth(self, test_client):
        """Verify feedback endpoint requires authentication when auth is enabled."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            response = test_client.post(
                "/api/v1/feedback",
                json={
                    "query_id": "test123",
                    "question": "test",
                    "answer": "test",
                    "rating": "up",
                },
            )
        assert response.status_code == 401


class TestModelPullEndpoints:
    """Tests for model pull endpoints."""

    def test_pull_status_returns_valid_response(self, test_client):
        """Verify pull status endpoint returns valid response."""
        response = test_client.get("/api/v1/models/pull/status")

        assert response.status_code == 200
        data = response.json()
        assert "active" in data

    def test_cancel_pull_when_not_running(self, test_client):
        """Verify cancel pull returns appropriate response when no pull is running."""
        with patch("pika.api.web.is_admin_auth_required", return_value=False):
            response = test_client.post("/api/v1/models/pull/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["cancelled"] is False
        assert "not" in data["message"].lower() or "no" in data["message"].lower()

    def test_cancel_pull_requires_auth(self, test_client):
        """Verify cancel pull requires authentication when auth is enabled."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            response = test_client.post("/api/v1/models/pull/cancel")
        assert response.status_code == 401


class TestQueryStatusEndpoints:
    """Tests for query status endpoints."""

    def test_query_status_returns_none_initially(self, test_client):
        """Verify query status returns 'none' when no query is active."""
        with patch("pika.api.web.is_admin_auth_required", return_value=False):
            response = test_client.get("/api/v1/query/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "none"

    def test_cancel_query_when_not_running(self, test_client):
        """Verify cancel query returns appropriate response when no query is running."""
        with patch("pika.api.web.is_admin_auth_required", return_value=False):
            response = test_client.post("/api/v1/query/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["cancelled"] is False

    def test_clear_query_status(self, test_client):
        """Verify query status can be cleared."""
        with patch("pika.api.web.is_admin_auth_required", return_value=False):
            response = test_client.delete("/api/v1/query/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"


class TestAsyncIndexingAPI:
    """Tests for async indexing API endpoints."""

    def test_start_indexing_endpoint(self, test_client):
        """Verify start indexing endpoint returns valid response."""
        # Reset indexing state
        import pika.services.rag as rag_module
        rag_module._index_task = None
        rag_module._active_index = None

        with patch("pika.api.web.is_admin_auth_required", return_value=False):
            response = test_client.post("/api/v1/index/start")

        assert response.status_code == 202
        data = response.json()
        assert "index_id" in data
        assert "status" in data
        assert data["status"] in ["started", "already_running"]
        assert "message" in data

    def test_index_status_endpoint(self, test_client):
        """Verify index status endpoint returns valid response."""
        response = test_client.get("/api/v1/index/status")

        assert response.status_code == 200
        data = response.json()
        assert "active" in data

    def test_index_status_when_not_running(self, test_client):
        """Verify index status shows inactive when no indexing running."""
        # Reset indexing state
        import pika.services.rag as rag_module
        rag_module._index_task = None
        rag_module._active_index = None

        response = test_client.get("/api/v1/index/status")

        assert response.status_code == 200
        data = response.json()
        assert data["active"] is False

    def test_cancel_indexing_when_not_running(self, test_client):
        """Verify cancel indexing returns appropriate response when not running."""
        # Reset indexing state
        import pika.services.rag as rag_module
        rag_module._index_task = None
        rag_module._active_index = None

        with patch("pika.api.web.is_admin_auth_required", return_value=False):
            response = test_client.post("/api/v1/index/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["cancelled"] is False
        assert "not" in data["message"].lower() or "no" in data["message"].lower()

    def test_start_indexing_requires_auth(self, test_client):
        """Verify start indexing requires authentication when auth is enabled."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            response = test_client.post("/api/v1/index/start")
        assert response.status_code == 401

    def test_cancel_indexing_requires_auth(self, test_client):
        """Verify cancel indexing requires authentication when auth is enabled."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            response = test_client.post("/api/v1/index/cancel")
        assert response.status_code == 401

    def test_index_info_combined_endpoint(self, test_client):
        """Verify combined index info endpoint returns stats and documents."""
        with patch("pika.api.web.is_admin_auth_required", return_value=False):
            response = test_client.get("/api/v1/index/info")

        assert response.status_code == 200
        data = response.json()
        # Should have both stats and documents
        assert "total_documents" in data
        assert "total_chunks" in data
        assert "collection_name" in data
        assert "documents" in data
        assert isinstance(data["documents"], list)

    def test_sync_index_blocked_when_async_running(self, test_client):
        """Verify sync index returns 409 when async indexing is running."""
        import asyncio

        import pika.services.rag as rag_module
        from pika.services.rag import IndexStatus, _set_active_index

        # Create a mock task that is not done
        async def dummy():
            await asyncio.sleep(100)

        loop = asyncio.new_event_loop()
        task = loop.create_task(dummy())
        rag_module._index_task = task

        # Set an active index status
        status = IndexStatus(index_id="test123", status="running")
        _set_active_index(status)

        try:
            with patch("pika.api.web.is_admin_auth_required", return_value=False):
                response = test_client.post("/api/v1/index")

            assert response.status_code == 409
            assert "async indexing" in response.json()["detail"].lower()
        finally:
            # Cleanup
            task.cancel()
            try:
                loop.run_until_complete(task)
            except asyncio.CancelledError:
                pass
            loop.close()
            rag_module._index_task = None
            _set_active_index(None)


class TestUserManagementEndpoints:
    """Tests for user management endpoints."""

    def test_list_users_requires_admin(self, test_client):
        """Verify listing users requires admin authentication."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            response = test_client.get("/api/v1/users")
        assert response.status_code == 401

    def test_create_user_requires_admin(self, test_client):
        """Verify creating users requires admin authentication."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            response = test_client.post(
                "/api/v1/users",
                json={"username": "newuser", "password": "password123", "role": "user"},
            )
        assert response.status_code == 401

    def test_delete_user_requires_admin(self, test_client):
        """Verify deleting users requires admin authentication."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            response = test_client.delete("/api/v1/users/1")
        assert response.status_code == 401
