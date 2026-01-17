"""Persistent application configuration service."""

import json
import logging
from pathlib import Path
from threading import Lock

from pika.config import get_settings

logger = logging.getLogger(__name__)


class AppConfigService:
    """Service for managing persistent application configuration."""

    def __init__(self, config_path: Path | None = None):
        settings = get_settings()
        self.config_path = config_path or Path(settings.chroma_persist_dir).parent / "config.json"
        self._lock = Lock()
        self._config: dict = {}
        self._load()

    def _load(self) -> None:
        """Load configuration from disk."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    self._config = json.load(f)
                logger.info(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
                self._config = {}
        else:
            self._config = {}

    def _save(self) -> None:
        """Save configuration to disk."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Saved config to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def get(self, key: str, default=None):
        """Get a configuration value."""
        with self._lock:
            return self._config.get(key, default)

    def set(self, key: str, value) -> None:
        """Set a configuration value and persist it."""
        with self._lock:
            self._config[key] = value
            self._save()

    def get_current_model(self) -> str:
        """Get the current model, falling back to env setting."""
        model = self.get("current_model")
        if model:
            return model
        return get_settings().ollama_model

    def set_current_model(self, model: str) -> None:
        """Set the current model."""
        self.set("current_model", model)


# Singleton instance
_app_config: AppConfigService | None = None


def get_app_config() -> AppConfigService:
    """Get the app config singleton."""
    global _app_config
    if _app_config is None:
        _app_config = AppConfigService()
    return _app_config
