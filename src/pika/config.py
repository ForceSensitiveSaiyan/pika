"""Configuration management using pydantic-settings."""

import secrets
from functools import lru_cache

from pydantic import field_validator
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
    ollama_model: str = "llama3.2:3b"
    ollama_timeout: int = 120

    # RAG settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    index_timeout: int = 600  # 10 minutes default for async indexing

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

    # Query queue settings
    max_concurrent_queries: int = 1  # Max queries running simultaneously (1 for CPU, 2+ for GPU)
    max_queued_per_user: int = 3  # Max pending queries per user
    queue_timeout: int = 300  # Seconds before queued query times out
    max_queue_size: int = 100  # Maximum total queue length

    # Validators for numeric settings
    @field_validator("max_concurrent_queries", "max_queued_per_user", "max_queue_size")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("queue_timeout", "ollama_timeout", "index_timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("timeout must be a positive integer")
        if v > 3600:
            raise ValueError("timeout cannot exceed 3600 seconds (1 hour)")
        return v

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v < 100:
            raise ValueError("chunk_size must be at least 100")
        if v > 10000:
            raise ValueError("chunk_size cannot exceed 10000")
        return v

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("top_k must be at least 1")
        if v > 50:
            raise ValueError("top_k cannot exceed 50")
        return v

    @field_validator("confidence_high", "confidence_medium", "confidence_low")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("confidence thresholds must be between 0 and 1")
        return v

    @field_validator("max_upload_size_mb")
    @classmethod
    def validate_upload_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_upload_size_mb must be positive")
        if v > 500:
            raise ValueError("max_upload_size_mb cannot exceed 500MB")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
