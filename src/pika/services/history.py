"""Query history and feedback service."""

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock

from pika.config import get_settings

logger = logging.getLogger(__name__)


class HistoryService:
    """Service for managing query history and feedback."""

    def __init__(self, data_dir: Path | None = None):
        settings = get_settings()
        self.data_dir = data_dir or Path(settings.chroma_persist_dir).parent
        self.history_path = self.data_dir / "history.json"
        self.feedback_path = self.data_dir / "feedback.json"
        self._max_history_items = settings.max_history_items
        self._max_feedback_items = settings.max_feedback_items
        self._lock = Lock()
        self._history: list[dict] = []
        self._feedback: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Load history and feedback from disk."""
        # Load history
        if self.history_path.exists():
            try:
                with open(self.history_path) as f:
                    self._history = json.load(f)
                logger.info(f"Loaded {len(self._history)} history items")
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
                self._history = []

        # Load feedback
        if self.feedback_path.exists():
            try:
                with open(self.feedback_path) as f:
                    self._feedback = json.load(f)
                logger.info(f"Loaded {len(self._feedback)} feedback items")
            except Exception as e:
                logger.warning(f"Failed to load feedback: {e}")
                self._feedback = []

    def _save_history(self) -> None:
        """Save history to disk."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, "w") as f:
                json.dump(self._history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def _save_feedback(self) -> None:
        """Save feedback to disk."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.feedback_path, "w") as f:
                json.dump(self._feedback, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")

    def add_query(
        self,
        question: str,
        answer: str,
        confidence: str,
        sources: list[str],
        username: str | None = None,
    ) -> str:
        """Add a query to history. Returns the query ID."""
        with self._lock:
            query_id = f"q_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"

            entry = {
                "id": query_id,
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "sources": sources,
                "timestamp": datetime.utcnow().isoformat(),
                "username": username,
            }

            self._history.insert(0, entry)

            # Trim to max size
            if len(self._history) > self._max_history_items:
                self._history = self._history[:self._max_history_items]

            self._save_history()
            return query_id

    def get_history(self, limit: int = 20, username: str | None = None) -> list[dict]:
        """Get recent query history, optionally filtered by username."""
        with self._lock:
            if username:
                # Filter by username
                user_history = [h for h in self._history if h.get("username") == username]
                return user_history[:limit]
            return self._history[:limit]

    def clear_history(self, username: str | None = None) -> None:
        """Clear history, optionally only for a specific user."""
        with self._lock:
            if username:
                # Only clear history for this user
                self._history = [h for h in self._history if h.get("username") != username]
            else:
                self._history = []
            self._save_history()

    def add_feedback(
        self,
        query_id: str,
        question: str,
        answer: str,
        rating: str,  # "up" or "down"
    ) -> None:
        """Add feedback for a query."""
        with self._lock:
            entry = {
                "query_id": query_id,
                "question": question,
                "answer": answer,
                "rating": rating,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Check if feedback already exists for this query
            for i, fb in enumerate(self._feedback):
                if fb.get("query_id") == query_id:
                    self._feedback[i] = entry
                    self._save_feedback()
                    return

            self._feedback.append(entry)

            # Trim to max size (keep most recent)
            if len(self._feedback) > self._max_feedback_items:
                self._feedback = self._feedback[-self._max_feedback_items:]

            self._save_feedback()

    def get_feedback(self, limit: int = 100) -> list[dict]:
        """Get recent feedback."""
        with self._lock:
            return self._feedback[-limit:]

    def get_feedback_stats(self) -> dict:
        """Get feedback statistics."""
        with self._lock:
            up_count = sum(1 for fb in self._feedback if fb.get("rating") == "up")
            down_count = sum(1 for fb in self._feedback if fb.get("rating") == "down")
            return {
                "total": len(self._feedback),
                "positive": up_count,
                "negative": down_count,
            }


# Singleton instance
_history_service: HistoryService | None = None
_history_service_lock = Lock()


def get_history_service() -> HistoryService:
    """Get the history service singleton."""
    global _history_service
    if _history_service is None:
        with _history_service_lock:
            if _history_service is None:
                _history_service = HistoryService()
    return _history_service
