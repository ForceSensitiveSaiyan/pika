"""Ollama client service for LLM interactions."""

import logging
from typing import AsyncIterator

import httpx

from pika.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _make_timeout(seconds: int) -> httpx.Timeout:
    """Create httpx timeout with extended read timeout for LLM generation."""
    return httpx.Timeout(
        connect=10.0,
        read=float(seconds),
        write=10.0,
        pool=10.0,
    )


class OllamaClient:
    """Async client for interacting with Ollama API."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.ollama_base_url
        self.model = self.settings.ollama_model
        self.timeout = self.settings.ollama_timeout

    async def health_check(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except httpx.RequestError:
            return False

    async def list_models(self) -> list[dict]:
        """List available models in Ollama."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])

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
                        import json
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
