"""Configuration management using pydantic-settings."""

import secrets
from functools import lru_cache

from pydantic_settings import BaseSettings


def _generate_secret() -> str:
    """Generate a secure random secret for session cookies."""
    return secrets.token_urlsafe(32)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App settings
    app_name: str = "PIKA"
    debug: bool = False

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout: int = 120

    # RAG settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5

    # Document settings
    documents_dir: str = "./documents"

    # Vector store settings
    chroma_persist_dir: str = "./data/chroma"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Confidence thresholds
    confidence_high: float = 0.7
    confidence_medium: float = 0.5
    confidence_low: float = 0.3

    # Authentication settings
    pika_admin_password: str | None = None  # Password for admin page
    pika_api_key: str | None = None  # API key for API endpoints
    pika_session_secret: str = ""  # Secret for session cookies (auto-generated if empty)

    # Security settings
    max_upload_size_mb: int = 50  # Maximum file upload size in MB
    rate_limit_auth: str = "5/minute"  # Rate limit for auth endpoints
    rate_limit_query: str = "30/minute"  # Rate limit for query endpoints

    def model_post_init(self, __context) -> None:
        """Generate session secret if not provided."""
        if not self.pika_session_secret:
            object.__setattr__(self, "pika_session_secret", _generate_secret())

    # Audit settings
    audit_log_path: str = "./data/audit.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
