"""Audit logging service for PIKA."""

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from pika.config import get_settings

logger = logging.getLogger(__name__)


class AuditLogger:
    """Service for logging audit events to a JSON lines file."""

    def __init__(self, log_path: Path | None = None):
        settings = get_settings()
        self.log_path = log_path or Path(settings.audit_log_path)
        self._lock = Lock()
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Ensure the log directory exists."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_log(self, event: dict[str, Any]) -> None:
        """Write a log event to the file."""
        event["timestamp"] = datetime.now().isoformat()
        try:
            with self._lock:
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

    def get_recent_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get the most recent log entries."""
        if not self.log_path.exists():
            return []

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Parse and return last N entries (most recent first)
            logs = []
            for line in reversed(lines[-limit:]):
                line = line.strip()
                if line:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            return logs
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
