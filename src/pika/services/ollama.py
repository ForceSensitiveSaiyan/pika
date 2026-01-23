"""Ollama client service for LLM interactions."""

import asyncio
import json
import logging
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

    def __init__(self, message: str = "Request to Ollama timed out. The model may be overloaded."):
        self.message = message
        super().__init__(self.message)


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
    "ModelInfo",
    "PullStatus",
    "get_ollama_client",
    "get_active_pull",
    "is_pull_running",
    "start_pull_task",
    "_format_size",
]


def _make_timeout(seconds: int) -> httpx.Timeout:
    """Create httpx timeout with extended read timeout for LLM generation.

    Note: We add 30s buffer to the read timeout so the application-level
    timeout (asyncio.wait_for) triggers first, giving cleaner error handling.
    """
    return httpx.Timeout(
        connect=10.0,
        read=float(seconds) + 30.0,  # Buffer to let app timeout trigger first
        write=10.0,
        pool=10.0,
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
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except httpx.RequestError:
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List available models in Ollama."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
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
        """Generate a completion from Ollama with retry logic."""
        model = model or self.model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        timeout = _make_timeout(self.timeout)
        prompt_len = len(prompt)
        system_len = len(system) if system else 0
        logger.info(f"Calling Ollama generate with timeout={self.timeout}s, prompt_len={prompt_len}, system_len={system_len}")

        async def make_request():
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                # Log non-200 responses for debugging
                if response.status_code != 200:
                    logger.error(f"Ollama returned {response.status_code}: {response.text[:500]}")
                # Handle specific HTTP errors
                if response.status_code == 404:
                    raise OllamaModelNotFoundError(model)
                response.raise_for_status()
                return response.json().get("response", "")

        try:
            return await retry_with_backoff(
                make_request,
                max_retries=max_retries,
                retryable_exceptions=(httpx.ConnectError, httpx.ConnectTimeout),
            )
        except httpx.ReadTimeout:
            raise OllamaTimeoutError(
                f"Request timed out after {self.timeout}s. "
                "Try a shorter prompt or increase OLLAMA_TIMEOUT."
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise OllamaModelNotFoundError(model)
            raise

    async def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion from Ollama."""
        model = model or self.model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=_make_timeout(self.timeout)) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]

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
