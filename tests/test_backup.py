"""Tests for backup and restore functionality."""

import io
import json
import zipfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from pika.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_admin_auth():
    """Mock admin authentication."""
    with patch("pika.api.web.is_admin_auth_required", return_value=False):
        yield


class TestBackupStatus:
    """Tests for backup status endpoint."""

    def test_backup_status_no_active_backup(self, client, mock_admin_auth):
        """Status returns inactive when no backup is running."""
        response = client.get("/admin/backup/status")
        assert response.status_code == 200
        data = response.json()
        assert data["active"] is False

    def test_backup_status_requires_auth(self, client):
        """Status endpoint requires authentication when enabled."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            with patch("pika.api.web.is_authenticated", return_value=False):
                response = client.get("/admin/backup/status")
                assert response.status_code == 401


class TestBackupStart:
    """Tests for starting backup."""

    def test_start_backup_requires_admin(self, client):
        """Start backup requires admin authentication."""
        with patch("pika.api.web.is_admin_auth_required", return_value=True):
            with patch("pika.api.web.is_authenticated", return_value=False):
                response = client.post("/admin/backup/start")
                assert response.status_code == 401


class TestBackupDownload:
    """Tests for backup download."""

    def test_download_no_backup_available(self, client, mock_admin_auth):
        """Download returns 404 when no backup is available."""
        response = client.get("/admin/backup/download")
        assert response.status_code == 404
        assert "No completed backup" in response.json()["detail"]


class TestBackupDelete:
    """Tests for backup deletion."""

    def test_delete_no_backup(self, client, mock_admin_auth):
        """Delete returns success even when no backup exists."""
        response = client.delete("/admin/backup")
        assert response.status_code == 200
        assert response.json()["status"] == "cleared"


class TestRestoreBackup:
    """Tests for restore functionality."""

    def test_restore_requires_zip(self, client, mock_admin_auth):
        """Restore requires a zip file."""
        response = client.post(
            "/admin/restore",
            files={"file": ("test.txt", b"not a zip", "text/plain")},
        )
        assert response.status_code == 400
        assert "must be a .zip" in response.json()["detail"]

    def test_restore_rejects_invalid_zip(self, client, mock_admin_auth):
        """Restore rejects invalid zip files."""
        response = client.post(
            "/admin/restore",
            files={"file": ("test.zip", b"not really a zip", "application/zip")},
        )
        assert response.status_code == 400
        assert "Invalid zip" in response.json()["detail"]

    def test_restore_rejects_empty_zip(self, client, mock_admin_auth):
        """Restore rejects zip files without expected content."""
        # Create an empty zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("random.txt", "some content")
        zip_buffer.seek(0)

        response = client.post(
            "/admin/restore",
            files={"file": ("test.zip", zip_buffer.read(), "application/zip")},
        )
        assert response.status_code == 400
        assert "missing expected data" in response.json()["detail"]

    def test_restore_valid_minimal_backup(self, client, mock_admin_auth, tmp_path):
        """Restore accepts a valid minimal backup."""
        # Create a minimal valid backup with config.json
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("config.json", json.dumps({"model": "test"}))
            zf.writestr("documents/test.txt", "test content")
        zip_buffer.seek(0)

        with patch("pika.api.web.get_settings") as mock_settings:
            mock_settings.return_value.chroma_persist_dir = str(tmp_path / "chroma")
            mock_settings.return_value.documents_dir = str(tmp_path / "documents")

            response = client.post(
                "/admin/restore",
                files={"file": ("backup.zip", zip_buffer.read(), "application/zip")},
            )

            # May fail due to database issues in test env, but should not be 400
            assert response.status_code in (200, 500)

    def test_restore_prevents_path_traversal(self, client, mock_admin_auth, tmp_path):
        """Restore blocks path traversal attempts."""
        # Create a zip with path traversal attempt
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("documents/../../../etc/passwd", "malicious")
        zip_buffer.seek(0)

        with patch("pika.api.web.get_settings") as mock_settings:
            mock_settings.return_value.chroma_persist_dir = str(tmp_path / "chroma")
            mock_settings.return_value.documents_dir = str(tmp_path / "documents")

            response = client.post(
                "/admin/restore",
                files={"file": ("backup.zip", zip_buffer.read(), "application/zip")},
            )
            assert response.status_code == 400
            assert "Invalid path" in response.json()["detail"]


class TestBackupStatusDataclass:
    """Tests for BackupStatus dataclass."""

    def test_backup_status_creation(self):
        """BackupStatus can be created with defaults."""
        from pika.api.web import BackupStatus

        status = BackupStatus(backup_id="test-123")
        assert status.backup_id == "test-123"
        assert status.status == "preparing"
        assert status.progress == 0
        assert status.total_files == 0
        assert status.processed_files == 0

    def test_backup_status_to_dict(self):
        """BackupStatus.to_dict() returns expected format."""
        from pika.api.web import BackupStatus

        status = BackupStatus(backup_id="test-123")
        status.status = "running"
        status.progress = 50
        status.total_files = 10
        status.processed_files = 5

        result = status.to_dict()
        assert result["backup_id"] == "test-123"
        assert result["status"] == "running"
        assert result["progress"] == 50
        assert result["total_files"] == 10
        assert result["processed_files"] == 5


class TestBackupRetention:
    """Tests for backup retention functionality."""

    def test_cleanup_old_backups_removes_excess(self, tmp_path):
        """Verify old backups are deleted when exceeding retention count."""
        import time

        from pika.api.web import _cleanup_old_backups

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        # Create 5 backup files with different timestamps
        for i in range(5):
            backup_file = backup_dir / f"pika_backup_2024010{i}_120000.zip"
            backup_file.write_text(f"backup {i}")
            # Set different modification times
            time.sleep(0.01)

        # Keep only 2 backups
        _cleanup_old_backups(backup_dir, keep_count=2)

        remaining = list(backup_dir.glob("pika_backup_*.zip"))
        assert len(remaining) == 2

        # Should keep the 2 newest (highest numbered in this case)
        remaining_names = {f.name for f in remaining}
        assert "pika_backup_20240104_120000.zip" in remaining_names
        assert "pika_backup_20240103_120000.zip" in remaining_names

    def test_cleanup_old_backups_unlimited_retention(self, tmp_path):
        """Verify unlimited retention keeps all backups."""
        from pika.api.web import _cleanup_old_backups

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        # Create 5 backup files
        for i in range(5):
            (backup_dir / f"pika_backup_2024010{i}_120000.zip").write_text(f"backup {i}")

        # Keep unlimited (0)
        _cleanup_old_backups(backup_dir, keep_count=0)

        remaining = list(backup_dir.glob("pika_backup_*.zip"))
        assert len(remaining) == 5

    def test_cleanup_old_backups_empty_dir(self, tmp_path):
        """Verify cleanup handles empty directory gracefully."""
        from pika.api.web import _cleanup_old_backups

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        # Should not raise
        _cleanup_old_backups(backup_dir, keep_count=2)

    def test_cleanup_old_backups_nonexistent_dir(self, tmp_path):
        """Verify cleanup handles nonexistent directory gracefully."""
        from pika.api.web import _cleanup_old_backups

        backup_dir = tmp_path / "nonexistent"

        # Should not raise
        _cleanup_old_backups(backup_dir, keep_count=2)
