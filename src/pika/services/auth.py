"""Authentication service for user management."""

import logging
import os
import secrets

import bcrypt

from pika.services.app_config import get_app_config
from pika.services.database import DatabaseService, get_database

logger = logging.getLogger(__name__)


class AuthService:
    """Service for authentication and user management."""

    def __init__(self, db: DatabaseService | None = None):
        self.db = db or get_database()
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Ensure the auth system is properly initialized on first run."""
        # Check if we need to migrate from config.json
        if not self.db.has_users():
            self._migrate_from_config()

        # Check for PIKA_ADMIN_PASSWORD env var
        if not self.db.has_users():
            self._create_from_env()

    def _migrate_from_config(self) -> None:
        """Migrate existing admin credentials from config.json to SQLite."""
        if self.db.is_migration_applied("config_migration"):
            return

        config = get_app_config()

        # Migrate admin credentials
        admin_creds = config.get_admin_credentials()
        if admin_creds:
            username = admin_creds.get("username", "admin")
            password_hash = admin_creds.get("password_hash", "")

            if password_hash:
                try:
                    self.db.create_user(username, password_hash, "admin")
                    logger.info(f"Migrated admin user '{username}' from config.json")
                except Exception as e:
                    logger.error(f"Failed to migrate admin user: {e}")

        # Migrate any additional users
        for user in config.get_users():
            username = user.get("username", "")
            password_hash = user.get("password_hash", "")
            role = user.get("role", "user")

            if username and password_hash:
                if not self.db.user_exists(username):
                    try:
                        self.db.create_user(username, password_hash, role)
                        logger.info(f"Migrated user '{username}' from config.json")
                    except Exception as e:
                        logger.error(f"Failed to migrate user '{username}': {e}")

        self.db.record_migration("config_migration")
        logger.info("Completed migration from config.json")

    def _create_from_env(self) -> None:
        """Create default admin from PIKA_ADMIN_PASSWORD env var."""
        admin_password = os.environ.get("PIKA_ADMIN_PASSWORD")
        if admin_password and not self.db.has_users():
            password_hash = self.hash_password(admin_password)
            self.db.create_user("admin", password_hash, "admin")
            logger.info("Created default admin user from PIKA_ADMIN_PASSWORD")

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode(), salt).decode()

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against a hash. Supports bcrypt and legacy SHA-256."""
        import hashlib

        # Check if it's a bcrypt hash (starts with $2b$, $2a$, or $2y$)
        if hashed.startswith(("$2b$", "$2a$", "$2y$")):
            try:
                return bcrypt.checkpw(password.encode(), hashed.encode())
            except Exception:
                return False
        else:
            # Legacy SHA-256 hash (64 hex characters)
            if len(hashed) == 64:
                legacy_hash = hashlib.sha256(password.encode()).hexdigest()
                return secrets.compare_digest(legacy_hash, hashed)
            return False

    def login(self, username: str, password: str) -> dict | None:
        """Verify credentials and return user data if valid.

        Returns user dict with id, username, role on success, None on failure.
        Also upgrades legacy password hashes to bcrypt.
        """
        user = self.db.get_user_by_username(username)
        if not user:
            return None

        if not user["is_active"]:
            logger.warning(f"Login attempt for disabled user: {username}")
            return None

        password_hash = user["password_hash"]
        if not self.verify_password(password, password_hash):
            return None

        # Auto-upgrade legacy SHA-256 to bcrypt
        if not password_hash.startswith(("$2b$", "$2a$", "$2y$")):
            new_hash = self.hash_password(password)
            self.db.update_password(user["id"], new_hash)
            logger.info(f"Upgraded password hash for user: {username}")

        # Update last login
        self.db.update_last_login(user["id"])

        return {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
        }

    def create_user(self, username: str, password: str, role: str = "user") -> dict:
        """Create a new user.

        Returns the created user data.
        Raises ValueError if username already exists.
        """
        if self.db.user_exists(username):
            raise ValueError(f"Username '{username}' already exists")

        if role not in ("admin", "user"):
            raise ValueError(f"Invalid role: {role}")

        password_hash = self.hash_password(password)
        user_id = self.db.create_user(username, password_hash, role)

        logger.info(f"Created user '{username}' with role '{role}'")

        return {
            "id": user_id,
            "username": username,
            "role": role,
        }

    def update_password(self, user_id: int, new_password: str) -> bool:
        """Update a user's password."""
        user = self.db.get_user_by_id(user_id)
        if not user:
            return False

        password_hash = self.hash_password(new_password)
        success = self.db.update_password(user_id, password_hash)

        if success:
            logger.info(f"Updated password for user ID {user_id}")

        return success

    def enable_user(self, user_id: int) -> bool:
        """Enable a disabled user."""
        success = self.db.set_user_active(user_id, True)
        if success:
            user = self.db.get_user_by_id(user_id)
            logger.info(f"Enabled user: {user['username'] if user else user_id}")
        return success

    def disable_user(self, user_id: int) -> bool:
        """Disable a user.

        Prevents disabling the last active admin.
        """
        user = self.db.get_user_by_id(user_id)
        if not user:
            return False

        # Prevent disabling the last admin
        if user["role"] == "admin" and user["is_active"]:
            if self.db.count_admins() <= 1:
                raise ValueError("Cannot disable the last admin user")

        success = self.db.set_user_active(user_id, False)
        if success:
            logger.info(f"Disabled user: {user['username']}")
        return success

    def delete_user(self, user_id: int) -> bool:
        """Delete a user.

        Prevents deleting the last active admin.
        """
        user = self.db.get_user_by_id(user_id)
        if not user:
            return False

        # Prevent deleting the last admin
        if user["role"] == "admin" and user["is_active"]:
            if self.db.count_admins() <= 1:
                raise ValueError("Cannot delete the last admin user")

        success = self.db.delete_user(user_id)
        if success:
            logger.info(f"Deleted user: {user['username']}")
        return success

    def list_users(self) -> list[dict]:
        """Get all users (without password hashes)."""
        return self.db.list_users()

    def get_user_by_username(self, username: str) -> dict | None:
        """Get a user by username (without password hash)."""
        user = self.db.get_user_by_username(username)
        if user:
            return {
                "id": user["id"],
                "username": user["username"],
                "role": user["role"],
                "is_active": user["is_active"],
                "created_at": user["created_at"],
                "last_login": user["last_login"],
            }
        return None

    def get_user_by_id(self, user_id: int) -> dict | None:
        """Get a user by ID (without password hash)."""
        user = self.db.get_user_by_id(user_id)
        if user:
            return {
                "id": user["id"],
                "username": user["username"],
                "role": user["role"],
                "is_active": user["is_active"],
                "created_at": user["created_at"],
                "last_login": user["last_login"],
            }
        return None

    def has_users(self) -> bool:
        """Check if any users exist."""
        return self.db.has_users()

    def is_setup_complete(self) -> bool:
        """Check if setup is complete (at least one user exists)."""
        return self.db.has_users()


# Singleton instance
_auth_service: AuthService | None = None


def get_auth_service() -> AuthService:
    """Get the auth service singleton."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
