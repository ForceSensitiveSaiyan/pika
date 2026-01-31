"""Audit logging service for PIKA."""

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from pika.config import get_settings

logger = logging.getLogger(__name__)

# Log rotation settings
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_BACKUP_COUNT = 5  # Keep 5 backup files


class AuditLogger:
    """Service for logging audit events to a JSON lines file with rotation."""

    def __init__(self, log_path: Path | None = None):
        settings = get_settings()
        self.log_path = log_path or Path(settings.audit_log_path)
        self._lock = Lock()
        self._write_count = 0  # Track writes to avoid checking size every time
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Ensure the log directory exists."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _rotate_if_needed(self) -> None:
        """Rotate log file if it exceeds max size."""
        try:
            if not self.log_path.exists():
                return

            if self.log_path.stat().st_size < MAX_LOG_SIZE:
                return

            # Rotate: audit.log -> audit.log.1 -> audit.log.2 -> ...
            for i in range(MAX_BACKUP_COUNT - 1, 0, -1):
                old_path = self.log_path.with_suffix(f".log.{i}")
                new_path = self.log_path.with_suffix(f".log.{i + 1}")
                if old_path.exists():
                    if new_path.exists():
                        new_path.unlink()
                    old_path.rename(new_path)

            # Current log -> .log.1
            backup_path = self.log_path.with_suffix(".log.1")
            if backup_path.exists():
                backup_path.unlink()
            self.log_path.rename(backup_path)

            logger.info(f"[Audit] Rotated log file (exceeded {MAX_LOG_SIZE // (1024*1024)}MB)")

        except Exception as e:
            logger.error(f"Failed to rotate audit log: {e}")

    def _write_log(self, event: dict[str, Any]) -> None:
        """Write a log event to the file with rotation support."""
        event["timestamp"] = datetime.now().isoformat()
        try:
            with self._lock:
                # Check rotation every 100 writes to avoid stat() overhead
                self._write_count += 1
                if self._write_count >= 100:
                    self._rotate_if_needed()
                    self._write_count = 0

                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def log_query(
        self,
        question: str,
        model: str,
        confidence: str,
        sources: list[str],
        error: str | None = None,
    ) -> None:
        """Log a query event."""
        self._write_log({
            "event": "query",
            "question": question,
            "model": model,
            "confidence": confidence,
            "sources": sources,
            "error": error,
        })

    def log_admin_action(
        self,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an admin action event."""
        self._write_log({
            "event": "admin",
            "action": action,
            "details": details or {},
        })

    def log_auth(
        self,
        action: str,
        success: bool,
        ip_address: str | None = None,
    ) -> None:
        """Log an authentication event."""
        self._write_log({
            "event": "auth",
            "action": action,
            "success": success,
            "ip_address": ip_address,
        })

    def get_total_count(self) -> int:
        """Get the total number of log entries."""
        if not self.log_path.exists():
            return 0

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception as e:
            logger.error(f"Failed to count audit log entries: {e}")
            return 0

    def get_recent_logs(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Get log entries with pagination support.

        Args:
            limit: Maximum number of entries to return
            offset: Number of entries to skip (from most recent)

        Returns:
            List of log entries, most recent first
        """
        if not self.log_path.exists():
            return []

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Parse all valid entries (most recent first)
            all_logs = []
            for line in reversed(lines):
                line = line.strip()
                if line:
                    try:
                        all_logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            # Apply pagination
            start = offset
            end = offset + limit
            return all_logs[start:end]
        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")
            return []


# Singleton instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get the audit logger singleton."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
