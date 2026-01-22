"""SQLite database service for persistent storage."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock

from pika.config import get_settings

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for managing SQLite database operations."""

    def __init__(self, db_path: Path | None = None):
        settings = get_settings()
        self.db_path = db_path or Path(settings.chroma_persist_dir).parent / "pika.db"
        self._lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database and create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                cursor = conn.cursor()

                # Create users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL DEFAULT 'user',
                        is_active INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        last_login TEXT
                    )
                """)

                # Create migrations table for future schema changes
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS migrations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        applied_at TEXT NOT NULL
                    )
                """)

                # Create index on username for faster lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)
                """)

                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
            finally:
                conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def execute(self, query: str, params: tuple = ()) -> list[dict]:
        """Execute a query and return results as list of dicts."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

    def execute_one(self, query: str, params: tuple = ()) -> dict | None:
        """Execute a query and return a single result."""
        results = self.execute(query, params)
        return results[0] if results else None

    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute an insert and return the last row id."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an update and return the number of affected rows."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()

    # User operations
    def get_user_by_id(self, user_id: int) -> dict | None:
        """Get a user by ID."""
        return self.execute_one(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        )

    def get_user_by_username(self, username: str) -> dict | None:
        """Get a user by username."""
        return self.execute_one(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        )

    def list_users(self) -> list[dict]:
        """Get all users (without password hashes)."""
        rows = self.execute(
            "SELECT id, username, role, is_active, created_at, last_login FROM users ORDER BY created_at DESC"
        )
        return rows

    def create_user(self, username: str, password_hash: str, role: str = "user") -> int:
        """Create a new user and return the user ID."""
        now = datetime.utcnow().isoformat()
        return self.execute_insert(
            "INSERT INTO users (username, password_hash, role, is_active, created_at) VALUES (?, ?, ?, 1, ?)",
            (username, password_hash, role, now)
        )

    def update_password(self, user_id: int, password_hash: str) -> bool:
        """Update a user's password."""
        rows = self.execute_update(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash, user_id)
        )
        return rows > 0

    def update_last_login(self, user_id: int) -> bool:
        """Update a user's last login timestamp."""
        now = datetime.utcnow().isoformat()
        rows = self.execute_update(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (now, user_id)
        )
        return rows > 0

    def set_user_active(self, user_id: int, is_active: bool) -> bool:
        """Enable or disable a user."""
        rows = self.execute_update(
            "UPDATE users SET is_active = ? WHERE id = ?",
            (1 if is_active else 0, user_id)
        )
        return rows > 0

    def delete_user(self, user_id: int) -> bool:
        """Delete a user."""
        rows = self.execute_update(
            "DELETE FROM users WHERE id = ?",
            (user_id,)
        )
        return rows > 0

    def count_admins(self) -> int:
        """Count the number of active admin users."""
        result = self.execute_one(
            "SELECT COUNT(*) as count FROM users WHERE role = 'admin' AND is_active = 1"
        )
        return result["count"] if result else 0

    def user_exists(self, username: str) -> bool:
        """Check if a username already exists."""
        result = self.execute_one(
            "SELECT 1 FROM users WHERE username = ?",
            (username,)
        )
        return result is not None

    def has_users(self) -> bool:
        """Check if any users exist in the database."""
        result = self.execute_one("SELECT 1 FROM users LIMIT 1")
        return result is not None

    # Migration tracking
    def is_migration_applied(self, name: str) -> bool:
        """Check if a migration has been applied."""
        result = self.execute_one(
            "SELECT 1 FROM migrations WHERE name = ?",
            (name,)
        )
        return result is not None

    def record_migration(self, name: str) -> None:
        """Record that a migration has been applied."""
        now = datetime.utcnow().isoformat()
        self.execute_insert(
            "INSERT INTO migrations (name, applied_at) VALUES (?, ?)",
            (name, now)
        )


# Singleton instance
_database: DatabaseService | None = None


def get_database() -> DatabaseService:
    """Get the database singleton."""
    global _database
    if _database is None:
        _database = DatabaseService()
    return _database
