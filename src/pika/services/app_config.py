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

    def is_setup_complete(self) -> bool:
        """Check if initial setup has been completed."""
        return bool(self._config.get("setup_complete", False))

    def get_admin_credentials(self) -> dict | None:
        """Get stored admin credentials."""
        return self.get("admin_credentials")

    def set_admin_credentials(self, username: str, password_hash: str) -> None:
        """Store admin credentials (password should be hashed)."""
        self.set("admin_credentials", {
            "username": username,
            "password_hash": password_hash,
        })
        self.set("setup_complete", True)

    def get_users(self) -> list[dict]:
        """Get list of configured users."""
        users = self._config.get("users", [])
        if isinstance(users, list):
            return list(users)
        return []

    def set_users(self, users: list[dict]) -> None:
        """Persist list of configured users."""
        self.set("users", users)

    def add_user(self, username: str, password_hash: str, role: str) -> None:
        """Add a user to the config store."""
        users = self.get_users()
        users.append({
            "username": username,
            "password_hash": password_hash,
            "role": role,
        })
        self.set_users(users)

    def update_user_password(self, username: str, password_hash: str) -> None:
        """Update an existing user's password hash."""
        users = self.get_users()
        for user in users:
            if user.get("username") == username:
                user["password_hash"] = password_hash
                break
        self.set_users(users)

    def delete_user(self, username: str) -> None:
        """Delete a user by username."""
        users = [user for user in self.get_users() if user.get("username") != username]
        self.set_users(users)

    def get_api_key(self) -> str | None:
        """Get stored API key."""
        return self.get("api_key")

    def set_api_key(self, api_key: str) -> None:
        """Store API key."""
        self.set("api_key", api_key)


# Singleton instance
_app_config: AppConfigService | None = None
_app_config_lock = Lock()


def get_app_config() -> AppConfigService:
    """Get the app config singleton."""
    global _app_config
    if _app_config is None:
        with _app_config_lock:
            if _app_config is None:
                _app_config = AppConfigService()
    return _app_config
