"""Configuration management using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings


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
    pika_session_secret: str = "change-me-in-production"  # Secret for session cookies

    # Audit settings
    audit_log_path: str = "./data/audit.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
