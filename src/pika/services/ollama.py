"""Ollama client service for LLM interactions."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator

import httpx

from pika.config import Settings, get_settings
from pika.services.app_config import get_app_config

logger = logging.getLogger(__name__)


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


def get_active_pull() -> PullStatus | None:
    """Get the currently active pull status, if any."""
    return _active_pull


def _set_active_pull(status: PullStatus | None) -> None:
    """Set the active pull status."""
    global _active_pull
    _active_pull = status


def _make_timeout(seconds: int) -> httpx.Timeout:
    """Create httpx timeout with extended read timeout for LLM generation."""
    return httpx.Timeout(
        connect=10.0,
        read=float(seconds),
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
    ) -> str:
        """Generate a completion from Ollama."""
        model = model or self.model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        timeout = _make_timeout(self.timeout)
        logger.info(f"Calling Ollama generate with timeout={self.timeout}s (read={timeout.read}s)")
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            return response.json().get("response", "")

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
