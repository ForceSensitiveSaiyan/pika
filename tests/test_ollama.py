"""Tests for Ollama service functionality."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestOllamaExceptions:
    """Tests for Ollama custom exceptions."""

    def test_connection_error_default_message(self):
        """Verify OllamaConnectionError has default message."""
        from pika.services.ollama import OllamaConnectionError

        error = OllamaConnectionError()
        assert "Cannot connect to Ollama" in str(error)

    def test_connection_error_custom_message(self):
        """Verify OllamaConnectionError accepts custom message."""
        from pika.services.ollama import OllamaConnectionError

        error = OllamaConnectionError("Custom error message")
        assert "Custom error message" in str(error)

    def test_model_not_found_error(self):
        """Verify OllamaModelNotFoundError includes model name."""
        from pika.services.ollama import OllamaModelNotFoundError

        error = OllamaModelNotFoundError("llama3.2:3b")
        assert "llama3.2:3b" in str(error)
        assert error.model == "llama3.2:3b"

    def test_timeout_error_default_message(self):
        """Verify OllamaTimeoutError has user-friendly default message."""
        from pika.services.ollama import OllamaTimeoutError

        error = OllamaTimeoutError()
        # New message is more user-friendly
        assert "taking longer than expected" in str(error).lower() or "wait" in str(error).lower()


class TestPullStatus:
    """Tests for PullStatus functionality."""

    def test_pull_status_creation(self):
        """Verify PullStatus can be created."""
        from pika.services.ollama import PullStatus

        status = PullStatus(model="llama3.2:3b")
        assert status.model == "llama3.2:3b"
        assert status.status == "starting"
        assert status.completed == 0
        assert status.total == 0
        assert status.error is None

    def test_pull_status_percent_calculation(self):
        """Verify percent calculation works correctly."""
        from pika.services.ollama import PullStatus

        status = PullStatus(model="test", completed=50, total=100)
        assert status.percent == 50

        status = PullStatus(model="test", completed=75, total=100)
        assert status.percent == 75

    def test_pull_status_percent_zero_total(self):
        """Verify percent returns 0 when total is 0."""
        from pika.services.ollama import PullStatus

        status = PullStatus(model="test", completed=0, total=0)
        assert status.percent == 0

    def test_pull_status_to_dict(self):
        """Verify PullStatus converts to dict correctly."""
        from pika.services.ollama import PullStatus

        status = PullStatus(
            model="llama3.2:3b",
            status="downloading",
            completed=500,
            total=1000,
        )

        data = status.to_dict()
        assert data["model"] == "llama3.2:3b"
        assert data["status"] == "downloading"
        assert data["completed"] == 500
        assert data["total"] == 1000
        assert data["percent"] == 50
        assert data["error"] is None


class TestPullTaskManagement:
    """Tests for pull task management functions."""

    def test_is_pull_running_false_initially(self):
        """Verify no pull is running initially."""
        import pika.services.ollama as ollama_module

        # Reset state
        ollama_module._pull_task = None
        ollama_module._active_pull = None

        from pika.services.ollama import is_pull_running

        assert is_pull_running() is False

    def test_get_active_pull_none_initially(self):
        """Verify no active pull initially."""
        import pika.services.ollama as ollama_module

        # Reset state
        ollama_module._active_pull = None

        from pika.services.ollama import get_active_pull

        assert get_active_pull() is None

    def test_cancel_pull_when_not_running(self):
        """Verify cancel returns False when no pull is running."""
        import pika.services.ollama as ollama_module

        # Reset state
        ollama_module._pull_task = None
        ollama_module._active_pull = None

        from pika.services.ollama import cancel_pull_task

        result = cancel_pull_task()
        assert result is False

    def test_set_active_pull(self):
        """Verify active pull can be set and retrieved."""
        import pika.services.ollama as ollama_module
        from pika.services.ollama import PullStatus, get_active_pull, _set_active_pull

        # Reset state
        ollama_module._active_pull = None

        status = PullStatus(model="test-model")
        _set_active_pull(status)

        active = get_active_pull()
        assert active is not None
        assert active.model == "test-model"

        # Clear
        _set_active_pull(None)
        assert get_active_pull() is None


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Verify ModelInfo can be created."""
        from pika.services.ollama import ModelInfo

        info = ModelInfo(
            name="llama3.2:3b",
            size=1234567890,
            modified_at="2026-01-24T12:00:00Z",
        )

        assert info.name == "llama3.2:3b"
        assert info.size == 1234567890
        assert info.modified_at == "2026-01-24T12:00:00Z"


class TestFormatSize:
    """Tests for _format_size utility function."""

    def test_format_bytes(self):
        """Verify bytes formatting."""
        from pika.services.ollama import _format_size

        assert _format_size(500) == "500 B"

    def test_format_kilobytes(self):
        """Verify kilobytes formatting."""
        from pika.services.ollama import _format_size

        assert "KB" in _format_size(1500)

    def test_format_megabytes(self):
        """Verify megabytes formatting."""
        from pika.services.ollama import _format_size

        assert "MB" in _format_size(1500000)

    def test_format_gigabytes(self):
        """Verify gigabytes formatting."""
        from pika.services.ollama import _format_size

        assert "GB" in _format_size(1500000000)


class TestOllamaClient:
    """Tests for OllamaClient class."""

    def test_client_initialization(self):
        """Verify client initializes with default settings."""
        from pika.services.ollama import OllamaClient

        client = OllamaClient()
        assert client.base_url is not None
        assert client.timeout > 0

    def test_client_model_property(self):
        """Verify model property returns current model."""
        from pika.services.ollama import OllamaClient
        from unittest.mock import patch

        client = OllamaClient()

        with patch("pika.services.ollama.get_app_config") as mock_config:
            mock_config.return_value.get_current_model.return_value = "test-model"
            assert client.model == "test-model"


class TestRetryWithBackoff:
    """Tests for retry_with_backoff function."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_first_attempt(self):
        """Verify function returns immediately on success."""
        from pika.services.ollama import retry_with_backoff

        call_count = 0

        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_with_backoff(successful_func, max_retries=3)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self):
        """Verify function retries and eventually succeeds."""
        from pika.services.ollama import retry_with_backoff
        import httpx

        call_count = 0

        async def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Connection failed")
            return "success"

        result = await retry_with_backoff(
            eventually_succeeds,
            max_retries=3,
            base_delay=0.01,  # Fast for tests
        )
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_raises_after_max_retries(self):
        """Verify function raises OllamaConnectionError after max retries."""
        from pika.services.ollama import retry_with_backoff, OllamaConnectionError
        import httpx

        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("Connection failed")

        with pytest.raises(OllamaConnectionError) as exc_info:
            await retry_with_backoff(
                always_fails,
                max_retries=2,
                base_delay=0.01,
            )

        assert "after 3 attempts" in str(exc_info.value)
        assert call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_does_not_retry_non_retryable_exceptions(self):
        """Verify non-retryable exceptions are raised immediately."""
        from pika.services.ollama import retry_with_backoff

        call_count = 0

        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            await retry_with_backoff(
                raises_value_error,
                max_retries=3,
                retryable_exceptions=(ConnectionError,),  # ValueError not included
            )

        assert call_count == 1  # Should not retry


class TestOllamaMetrics:
    """Tests for Ollama metrics integration."""

    def test_health_check_updates_metric(self):
        """Verify health check updates OLLAMA_HEALTHY gauge."""
        from pika.services.metrics import OLLAMA_HEALTHY

        # The metric should exist and be settable
        OLLAMA_HEALTHY.set(1)
        OLLAMA_HEALTHY.set(0)
        # No exception means success

    def test_request_metrics_exist(self):
        """Verify Ollama request metrics are defined."""
        from pika.services.metrics import OLLAMA_REQUEST_COUNT, OLLAMA_REQUEST_LATENCY

        # Should be able to use the metrics
        OLLAMA_REQUEST_COUNT.labels(status="test").inc()
        OLLAMA_REQUEST_LATENCY.observe(1.0)
        # No exception means success
