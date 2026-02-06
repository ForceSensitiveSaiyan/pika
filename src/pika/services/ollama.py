"""Ollama client service for LLM interactions."""

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator

import httpx

from pika.config import Settings, get_settings
from pika.services.app_config import get_app_config

logger = logging.getLogger(__name__)


# Custom exceptions for better error handling
class OllamaConnectionError(Exception):
    """Raised when Ollama is unreachable."""

    def __init__(self, message: str = "Cannot connect to Ollama. Please ensure Ollama is running."):
        self.message = message
        super().__init__(self.message)


class OllamaModelNotFoundError(Exception):
    """Raised when the requested model is not available."""

    def __init__(self, model: str):
        self.model = model
        self.message = f"Model '{model}' not found. Please pull it from Admin or select a different model."
        super().__init__(self.message)


class OllamaTimeoutError(Exception):
    """Raised when Ollama request times out."""

    def __init__(self, message: str = "The response is taking longer than expected. Try a shorter question or wait a moment."):
        self.message = message
        super().__init__(self.message)


class OllamaCircuitOpenError(Exception):
    """Raised when circuit breaker is open and requests are being rejected."""

    def __init__(self, message: str = "The AI service is recovering. You can still search your documents."):
        self.message = message
        super().__init__(self.message)


class CircuitState:
    """Circuit breaker states."""
    CLOSED = 0  # Normal operation
    HALF_OPEN = 1  # Testing if service recovered
    OPEN = 2  # Failing fast, no requests allowed


class CircuitBreaker:
    """Circuit breaker for Ollama connections.

    Prevents cascading failures by failing fast when Ollama is unavailable.
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Requests fail immediately without calling Ollama
    - HALF_OPEN: Allow one test request to check if Ollama recovered
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> int:
        """Get current circuit state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN

    def _update_metrics(self) -> None:
        """Update Prometheus metrics for circuit breaker state."""
        from pika.services.metrics import CIRCUIT_BREAKER_STATE
        CIRCUIT_BREAKER_STATE.set(self._state)

    async def is_available(self) -> bool:
        """Check if requests should be allowed through.

        Returns True if circuit is closed or half-open (test allowed).
        Returns False if circuit is open and recovery timeout hasn't elapsed.
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                import time
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    # Transition to half-open to allow a test request
                    self._state = CircuitState.HALF_OPEN
                    self._update_metrics()
                    logger.info(f"[CircuitBreaker] Transitioning to HALF_OPEN after {elapsed:.1f}s")
                    return True
                return False

            # HALF_OPEN - allow the test request
            return True

    async def record_success(self) -> None:
        """Record a successful request. Resets failure count and closes circuit."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info("[CircuitBreaker] Test request succeeded, closing circuit")
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            self._update_metrics()

    async def record_failure(self) -> None:
        """Record a failed request. May open the circuit."""
        import time
        from pika.services.metrics import CIRCUIT_BREAKER_TRIPS

        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Test request failed, back to open
                self._state = CircuitState.OPEN
                self._update_metrics()
                logger.warning("[CircuitBreaker] Test request failed, circuit remains OPEN")
            elif self._failure_count >= self.failure_threshold:
                # Too many failures, open the circuit
                self._state = CircuitState.OPEN
                self._update_metrics()
                CIRCUIT_BREAKER_TRIPS.inc()
                logger.warning(
                    f"[CircuitBreaker] Circuit OPEN after {self._failure_count} consecutive failures"
                )

    def reset(self) -> None:
        """Reset circuit breaker to closed state. For testing."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._update_metrics()


# Global circuit breaker instance
_circuit_breaker: CircuitBreaker | None = None
_circuit_breaker_lock = threading.Lock()


def get_circuit_breaker() -> CircuitBreaker:
    """Get or create the circuit breaker singleton."""
    global _circuit_breaker
    if _circuit_breaker is None:
        with _circuit_breaker_lock:
            if _circuit_breaker is None:
                from pika.config import get_settings
                settings = get_settings()
                _circuit_breaker = CircuitBreaker(
                    failure_threshold=settings.circuit_breaker_failure_threshold,
                    recovery_timeout=settings.circuit_breaker_recovery_timeout,
                )
    return _circuit_breaker


async def retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    retryable_exceptions: tuple = (httpx.ConnectError, httpx.ConnectTimeout),
):
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async callable to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        retryable_exceptions: Tuple of exceptions that trigger a retry
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Ollama request failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"Ollama request failed after {max_retries + 1} attempts: {e}")

    # Convert to our custom exception
    raise OllamaConnectionError(
        f"Failed to connect to Ollama after {max_retries + 1} attempts. "
        "Please check that Ollama is running and accessible."
    )


@dataclass
class ModelInfo:
    """Information about an Ollama model."""

    name: str
    size: int
    modified_at: str


@dataclass
class PullStatus:
    """Status of an active model pull."""

    model: str
    status: str = "starting"
    completed: int = 0
    total: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    error: str | None = None

    @property
    def percent(self) -> int:
        if self.total == 0:
            return 0
        return int((self.completed / self.total) * 100)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "status": self.status,
            "completed": self.completed,
            "total": self.total,
            "percent": self.percent,
            "error": self.error,
        }


# Global pull status tracker
_active_pull: PullStatus | None = None
_pull_task: "asyncio.Task | None" = None


def get_active_pull() -> PullStatus | None:
    """Get the currently active pull status, if any."""
    return _active_pull


def _set_active_pull(status: PullStatus | None) -> None:
    """Set the active pull status."""
    global _active_pull
    _active_pull = status


def is_pull_running() -> bool:
    """Check if a pull task is currently running."""
    return _pull_task is not None and not _pull_task.done()


def cancel_pull_task() -> bool:
    """Cancel the currently running pull task.

    Returns True if a task was cancelled, False if no task was running.
    """
    global _pull_task, _active_pull

    if _pull_task is None or _pull_task.done():
        return False

    _pull_task.cancel()

    if _active_pull:
        _active_pull.status = "cancelled"
        _active_pull.error = "Download cancelled by user"

    # Clear the active pull
    _set_active_pull(None)
    _pull_task = None

    logger.info("Model pull cancelled by user")
    return True


async def start_pull_task(client: "OllamaClient", model_name: str) -> None:
    """Start a background pull task if not already running."""
    global _pull_task

    if is_pull_running():
        return  # Already pulling

    async def run_pull():
        try:
            async for _ in client.pull_model(model_name):
                pass  # Status is updated inside pull_model
        except Exception as e:
            logger.error(f"Background pull failed: {e}")

    _pull_task = asyncio.create_task(run_pull())


# Export custom exceptions for use in other modules
__all__ = [
    "OllamaClient",
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "OllamaTimeoutError",
    "OllamaCircuitOpenError",
    "CircuitBreaker",
    "CircuitState",
    "get_circuit_breaker",
    "ModelInfo",
    "PullStatus",
    "get_ollama_client",
    "get_active_pull",
    "is_pull_running",
    "start_pull_task",
    "cancel_pull_task",
    "_format_size",
]


def _make_timeout(seconds: int, settings: Settings | None = None) -> httpx.Timeout:
    """Create httpx timeout with extended read timeout for LLM generation."""
    settings = settings or get_settings()
    return httpx.Timeout(
        connect=settings.http_connect_timeout,
        read=float(seconds),
        write=settings.http_write_timeout,
        pool=settings.http_pool_timeout,
    )


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


# Shared httpx client for connection pooling
_http_client: httpx.AsyncClient | None = None
_http_client_lock = asyncio.Lock()


async def get_http_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """Get or create the shared httpx client with connection pooling."""
    global _http_client
    async with _http_client_lock:
        if _http_client is None or _http_client.is_closed:
            # Create client with connection pooling
            _http_client = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
            logger.info("[Ollama] Created shared HTTP client with connection pooling")
        return _http_client


async def close_http_client() -> None:
    """Close the shared httpx client. Call during app shutdown."""
    global _http_client
    async with _http_client_lock:
        if _http_client is not None and not _http_client.is_closed:
            await _http_client.aclose()
            _http_client = None
            logger.info("[Ollama] Closed shared HTTP client")


class OllamaClient:
    """Async client for interacting with Ollama API."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.ollama_base_url
        self.timeout = self.settings.ollama_timeout

    @property
    def model(self) -> str:
        """Get the current model from app config."""
        return get_app_config().get_current_model()

    async def health_check(self) -> bool:
        """Check if Ollama is running and accessible."""
        from pika.services.metrics import OLLAMA_HEALTHY

        try:
            client = await get_http_client(timeout=5.0)
            response = await client.get(f"{self.base_url}/api/tags")
            is_healthy = response.status_code == 200
            OLLAMA_HEALTHY.set(1 if is_healthy else 0)
            return is_healthy
        except httpx.RequestError:
            OLLAMA_HEALTHY.set(0)
            return False

    async def list_models(self, max_retries: int = 2) -> list[ModelInfo]:
        """List available models in Ollama with retry logic."""

        async def _fetch_models():
            client = await get_http_client(timeout=self.timeout)
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json()

        try:
            data = await retry_with_backoff(
                _fetch_models,
                max_retries=max_retries,
                retryable_exceptions=(httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout),
            )
        except OllamaConnectionError:
            # Return empty list if Ollama is unreachable
            logger.warning("Could not connect to Ollama to list models")
            return []

        models = []
        for m in data.get("models", []):
            models.append(ModelInfo(
                name=m.get("name", ""),
                size=m.get("size", 0),
                modified_at=m.get("modified_at", ""),
            ))
        return models

    async def pull_model(self, model_name: str) -> AsyncIterator[dict]:
        """Pull a model from Ollama registry, yielding progress updates."""
        payload = {"name": model_name, "stream": True}

        # Use a very long timeout for model pulls (can take a while)
        timeout = httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)

        # Initialize pull status tracking
        pull_status = PullStatus(model=model_name)
        _set_active_pull(pull_status)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/pull",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            # Update pull status
                            if "status" in data:
                                pull_status.status = data["status"]
                            if "completed" in data:
                                pull_status.completed = data["completed"]
                            if "total" in data:
                                pull_status.total = data["total"]
                            yield data
        except Exception as e:
            pull_status.error = str(e)
            pull_status.status = "error"
            raise
        finally:
            # Clear active pull when done (success or error)
            if pull_status.status == "success" or pull_status.error:
                _set_active_pull(None)

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        max_retries: int = 3,
    ) -> str:
        """Generate a completion from Ollama with retry logic and circuit breaker.

        Uses streaming mode internally to work around buffering issues
        in Docker networking on macOS.

        Raises:
            OllamaCircuitOpenError: If circuit breaker is open
            OllamaConnectionError: If cannot connect after retries
            OllamaTimeoutError: If request times out
            OllamaModelNotFoundError: If model not available
        """
        import time as time_module
        from pika.config import get_settings

        settings = get_settings()

        # Check circuit breaker if enabled
        if settings.circuit_breaker_enabled:
            circuit = get_circuit_breaker()
            if not await circuit.is_available():
                logger.warning("[OLLAMA] Circuit breaker is OPEN, rejecting request")
                raise OllamaCircuitOpenError()

        model = model or self.model
        # Use non-streaming mode - streaming has issues with Docker networking on macOS
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # Non-streaming works better in Docker
        }
        if system:
            payload["system"] = system

        timeout = _make_timeout(self.timeout)
        prompt_len = len(prompt)
        system_len = len(system) if system else 0
        start_time = time_module.time()
        logger.info(f"[OLLAMA] Starting generate: timeout={self.timeout}s, prompt_len={prompt_len}, system_len={system_len}, model={model}")

        def sync_request():
            """Synchronous request using requests library - runs in thread pool."""
            import requests as req_lib

            request_start = time_module.time()
            logger.info(f"[OLLAMA] Sending request to {self.base_url}/api/generate")

            try:
                response = req_lib.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                elapsed = time_module.time() - request_start
                logger.info(f"[OLLAMA] Response: status={response.status_code}, elapsed={elapsed:.1f}s")

                if response.status_code == 404:
                    raise OllamaModelNotFoundError(model)
                response.raise_for_status()

                data = response.json()
                result = data.get("response", "")
                logger.info(f"[OLLAMA] Complete: {len(result)} chars in {elapsed:.1f}s")
                return result

            except req_lib.Timeout as e:
                elapsed = time_module.time() - request_start
                logger.error(f"[OLLAMA] Timeout after {elapsed:.1f}s: {e}")
                raise
            except req_lib.RequestException as e:
                elapsed = time_module.time() - request_start
                logger.error(f"[OLLAMA] Request error after {elapsed:.1f}s: {type(e).__name__}: {e}")
                raise
            except Exception as e:
                elapsed = time_module.time() - request_start
                logger.error(f"[OLLAMA] Error after {elapsed:.1f}s: {type(e).__name__}: {e}")
                raise

        async def make_request():
            # Run synchronous request in thread pool to avoid async networking issues
            return await asyncio.to_thread(sync_request)

        from pika.services.metrics import OLLAMA_REQUEST_COUNT, OLLAMA_REQUEST_LATENCY

        try:
            import requests as req_lib
            result = await retry_with_backoff(
                make_request,
                max_retries=max_retries,
                retryable_exceptions=(req_lib.ConnectionError, req_lib.Timeout),
            )
            total_elapsed = time_module.time() - start_time
            logger.info(f"[OLLAMA] Generate completed in {total_elapsed:.1f}s")

            # Record success metrics
            OLLAMA_REQUEST_COUNT.labels(status="success").inc()
            OLLAMA_REQUEST_LATENCY.observe(total_elapsed)

            # Record circuit breaker success
            if settings.circuit_breaker_enabled:
                await get_circuit_breaker().record_success()

            return result
        except asyncio.CancelledError:
            total_elapsed = time_module.time() - start_time
            logger.info(f"[OLLAMA] Request cancelled after {total_elapsed:.1f}s")
            OLLAMA_REQUEST_COUNT.labels(status="cancelled").inc()
            raise
        except OllamaConnectionError:
            total_elapsed = time_module.time() - start_time
            OLLAMA_REQUEST_COUNT.labels(status="connection_error").inc()
            OLLAMA_REQUEST_LATENCY.observe(total_elapsed)
            # Record circuit breaker failure
            if settings.circuit_breaker_enabled:
                await get_circuit_breaker().record_failure()
            raise
        except Exception as e:
            total_elapsed = time_module.time() - start_time
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                logger.error(f"[OLLAMA] Timeout after {total_elapsed:.1f}s")
                OLLAMA_REQUEST_COUNT.labels(status="timeout").inc()
                OLLAMA_REQUEST_LATENCY.observe(total_elapsed)
                # Record circuit breaker failure for timeouts
                if settings.circuit_breaker_enabled:
                    await get_circuit_breaker().record_failure()
                raise OllamaTimeoutError(
                    "The response is taking longer than expected. "
                    "Try a shorter question or wait a moment."
                )
            logger.error(f"[OLLAMA] Unexpected failure after {total_elapsed:.1f}s: {type(e).__name__}: {e}")
            OLLAMA_REQUEST_COUNT.labels(status="error").inc()
            OLLAMA_REQUEST_LATENCY.observe(total_elapsed)
            raise

    async def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        max_retries: int = 2,
    ) -> AsyncIterator[str]:
        """Stream a completion from Ollama with retry on connection failure and circuit breaker.

        Note: Retries only happen before streaming starts. Once tokens begin
        flowing, the stream cannot be restarted.

        Raises:
            OllamaCircuitOpenError: If circuit breaker is open
            OllamaConnectionError: If cannot connect after retries
            OllamaTimeoutError: If request times out
            OllamaModelNotFoundError: If model not available
        """
        from pika.config import get_settings
        from pika.services.metrics import OLLAMA_REQUEST_COUNT

        settings = get_settings()

        # Check circuit breaker if enabled
        if settings.circuit_breaker_enabled:
            circuit = get_circuit_breaker()
            if not await circuit.is_available():
                logger.warning("[OLLAMA] Circuit breaker is OPEN, rejecting stream request")
                raise OllamaCircuitOpenError()

        model = model or self.model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
        }
        if system:
            payload["system"] = system

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=_make_timeout(self.timeout)) as client:
                    async with client.stream(
                        "POST",
                        f"{self.base_url}/api/generate",
                        json=payload,
                    ) as response:
                        if response.status_code == 404:
                            OLLAMA_REQUEST_COUNT.labels(status="model_not_found").inc()
                            raise OllamaModelNotFoundError(model)
                        response.raise_for_status()

                        # Once we start yielding, no more retries
                        async for line in response.aiter_lines():
                            if line:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]

                        # Success - record metric and circuit breaker success
                        OLLAMA_REQUEST_COUNT.labels(status="success").inc()
                        if settings.circuit_breaker_enabled:
                            await get_circuit_breaker().record_success()
                        return

            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                last_error = e
                if attempt < max_retries:
                    delay = min(1.0 * (2 ** attempt), 5.0)
                    logger.warning(
                        f"[OLLAMA] Stream connection failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    OLLAMA_REQUEST_COUNT.labels(status="connection_error").inc()
                    # Record circuit breaker failure
                    if settings.circuit_breaker_enabled:
                        await get_circuit_breaker().record_failure()
                    raise OllamaConnectionError(
                        "The AI assistant is not responding. Please check that Ollama is running."
                    )
            except (OllamaModelNotFoundError, OllamaConnectionError):
                raise
            except httpx.TimeoutException:
                OLLAMA_REQUEST_COUNT.labels(status="timeout").inc()
                # Record circuit breaker failure for timeouts
                if settings.circuit_breaker_enabled:
                    await get_circuit_breaker().record_failure()
                raise OllamaTimeoutError()
            except Exception as e:
                OLLAMA_REQUEST_COUNT.labels(status="error").inc()
                logger.error(f"[OLLAMA] Stream error: {type(e).__name__}: {e}")
                raise

    async def embed(self, text: str, model: str | None = None) -> list[float]:
        """Generate embeddings for text."""
        model = model or self.model
        payload = {
            "model": model,
            "prompt": text,
        }

        async with httpx.AsyncClient(timeout=_make_timeout(self.timeout)) as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
            )
            response.raise_for_status()
            return response.json().get("embedding", [])


def get_ollama_client() -> OllamaClient:
    """Get Ollama client instance."""
    return OllamaClient()
