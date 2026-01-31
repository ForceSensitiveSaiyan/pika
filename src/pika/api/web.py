"""Web interface routes for PIKA."""

import asyncio
import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock

import bcrypt
from fastapi import APIRouter, Depends, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from slowapi import Limiter
from slowapi.util import get_remote_address


# Cache-control headers to prevent browser from caching authenticated pages
NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate, private",
    "Pragma": "no-cache",
    "Expires": "0",
}

from pika.config import Settings, get_settings
from pika.services.app_config import get_app_config
from pika.services.audit import get_audit_logger
from pika.services.auth import AuthService, get_auth_service
from pika.services.documents import DocumentInfo, DocumentProcessor, get_document_processor

# Rate limiter for auth endpoints
limiter = Limiter(key_func=get_remote_address)

logger = logging.getLogger(__name__)

router = APIRouter()

# Templates directory
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Thread-safe session storage with expiration
_sessions: dict[str, dict] = {}
_sessions_lock = Lock()
SESSION_MAX_AGE = 86400  # 24 hours in seconds
SESSION_CLEANUP_INTERVAL = 300  # Run cleanup every 5 minutes

# CSRF token storage (maps token -> session_id for validation)
_csrf_tokens: dict[str, tuple[str, float]] = {}  # token -> (session_id, created_at)
_csrf_lock = Lock()
CSRF_TOKEN_MAX_AGE = 3600  # 1 hour

# Background cleanup task
_session_cleanup_task = None
_session_cleanup_stop = None


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_hashed_password(password: str, hashed: str) -> bool:
    """Verify a password against a hash. Supports both bcrypt and legacy SHA-256."""
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


def _cleanup_expired_sessions() -> int:
    """Remove expired sessions. Returns count of cleaned sessions."""
    current_time = time.time()
    expired = [
        sid for sid, data in _sessions.items()
        if current_time - data.get("created_at", 0) > SESSION_MAX_AGE
    ]
    for sid in expired:
        del _sessions[sid]
    return len(expired)


def _cleanup_expired_csrf_tokens() -> int:
    """Remove expired CSRF tokens. Returns count of cleaned tokens."""
    current_time = time.time()
    expired = [
        token for token, (_, created_at) in _csrf_tokens.items()
        if current_time - created_at > CSRF_TOKEN_MAX_AGE
    ]
    for token in expired:
        del _csrf_tokens[token]
    return len(expired)


async def _session_cleanup_loop() -> None:
    """Background task that periodically cleans up expired sessions and tokens."""
    import asyncio
    logger.info("[Sessions] Background cleanup task started")

    while True:
        try:
            # Check for stop signal
            if _session_cleanup_stop and _session_cleanup_stop.is_set():
                break

            # Run cleanup
            with _sessions_lock:
                sessions_cleaned = _cleanup_expired_sessions()
            with _csrf_lock:
                tokens_cleaned = _cleanup_expired_csrf_tokens()

            if sessions_cleaned or tokens_cleaned:
                logger.debug(f"[Sessions] Cleaned up {sessions_cleaned} sessions, {tokens_cleaned} CSRF tokens")

            await asyncio.sleep(SESSION_CLEANUP_INTERVAL)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[Sessions] Cleanup error: {e}")
            await asyncio.sleep(SESSION_CLEANUP_INTERVAL)

    logger.info("[Sessions] Background cleanup task stopped")


async def init_session_cleanup() -> None:
    """Initialize the background session cleanup task."""
    import asyncio
    global _session_cleanup_task, _session_cleanup_stop

    _session_cleanup_stop = asyncio.Event()
    _session_cleanup_task = asyncio.create_task(_session_cleanup_loop())


async def shutdown_session_cleanup() -> None:
    """Shutdown the background session cleanup task."""
    global _session_cleanup_task, _session_cleanup_stop

    if _session_cleanup_stop:
        _session_cleanup_stop.set()

    if _session_cleanup_task:
        _session_cleanup_task.cancel()
        try:
            await _session_cleanup_task
        except asyncio.CancelledError:
            pass  # Expected when task is cancelled
        except Exception as e:
            logger.warning(f"[Sessions] Error during cleanup task shutdown: {e}")
        _session_cleanup_task = None

    logger.info("[Sessions] Cleanup task shutdown complete")


def create_session(user_data: dict) -> str:
    """Create a new session with thread safety."""
    session_id = secrets.token_urlsafe(32)
    with _sessions_lock:
        _cleanup_expired_sessions()
        _sessions[session_id] = {
            **user_data,
            "created_at": time.time(),
        }
    return session_id


def get_session(session_id: str) -> dict | None:
    """Get session data with thread safety."""
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session:
            # Check if expired
            if time.time() - session.get("created_at", 0) > SESSION_MAX_AGE:
                del _sessions[session_id]
                return None
        return session


def delete_session(session_id: str) -> None:
    """Delete a session with thread safety."""
    with _sessions_lock:
        _sessions.pop(session_id, None)


def generate_csrf_token(session_id: str | None = None) -> str:
    """Generate a CSRF token tied to the current session."""
    token = secrets.token_urlsafe(32)
    with _csrf_lock:
        # Cleanup expired tokens
        current_time = time.time()
        expired = [t for t, (_, created) in _csrf_tokens.items()
                   if current_time - created > CSRF_TOKEN_MAX_AGE]
        for t in expired:
            del _csrf_tokens[t]
        # Store new token
        _csrf_tokens[token] = (session_id or "", current_time)
    return token


def validate_csrf_token(token: str | None, session_id: str | None = None) -> bool:
    """Validate a CSRF token."""
    if not token:
        return False
    with _csrf_lock:
        data = _csrf_tokens.get(token)
        if not data:
            return False
        stored_session, created_at = data
        # Check if expired
        if time.time() - created_at > CSRF_TOKEN_MAX_AGE:
            del _csrf_tokens[token]
            return False
        # Invalidate token after use (one-time use)
        del _csrf_tokens[token]
        # If session tracking enabled, verify session matches
        if session_id and stored_session and stored_session != session_id:
            return False
        return True


def is_setup_required() -> bool:
    """Check if initial setup is required."""
    auth = get_auth_service()
    # Setup is required if no users exist
    return not auth.is_setup_complete()


def is_admin_auth_required() -> bool:
    """Check if admin authentication is required."""
    auth = get_auth_service()
    # Auth required if any users exist
    return auth.is_setup_complete()


def is_authenticated(request: Request) -> bool:
    """Check if the request is authenticated."""
    if not is_admin_auth_required():
        return True
    session_id = request.cookies.get("pika_session")
    if not session_id:
        return False
    session = get_session(session_id)
    if not session:
        return False
    # Verify user still exists and is active in DB
    auth = get_auth_service()
    user = auth.get_user_by_username(session.get("username", ""))
    if not user or not user.get("is_active"):
        # Session is stale, delete it
        delete_session(session_id)
        return False
    return True


def get_current_user(request: Request) -> dict | None:
    """Get the current user from session."""
    session_id = request.cookies.get("pika_session")
    if not session_id:
        return None
    session = get_session(session_id)
    if not session:
        return None
    return {
        "username": session.get("username"),
        "role": session.get("role"),
    }


def is_admin(request: Request) -> bool:
    """Check if current user is an admin."""
    user = get_current_user(request)
    return user is not None and user.get("role") == "admin"


def verify_password(username: str, password: str) -> dict | None:
    """Verify username/password using the AuthService.

    Supports auto-upgrade of legacy SHA-256 hashes to bcrypt on successful login.
    """
    auth = get_auth_service()
    return auth.login(username, password)


def require_user_auth(request: Request) -> bool:
    """Dependency to require any authenticated user."""
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required")
    return True


def require_admin_auth(request: Request) -> bool:
    """Dependency to require admin authentication."""
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required")
    if not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")
    return True


def require_admin_or_api_auth(
    request: Request,
    x_api_key: str | None = Header(None),
) -> bool:
    """Allow session auth or valid API key when required."""
    if not is_admin_auth_required():
        return True
    if is_authenticated(request):
        return True

    settings = get_settings()
    if settings.pika_api_key is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key required")
    if not secrets.compare_digest(x_api_key, settings.pika_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat interface."""
    # Redirect to setup if first launch
    if is_setup_required():
        return RedirectResponse(url="/setup", status_code=302)
    if is_admin_auth_required() and not is_authenticated(request):
        return RedirectResponse(url="/admin/login", status_code=302)
    user = get_current_user(request)
    response = templates.TemplateResponse("index.html", {
        "request": request,
        "user": user,
        "is_admin": user and user.get("role") == "admin",
    })
    # Prevent browser from caching authenticated pages
    for header, value in NO_CACHE_HEADERS.items():
        response.headers[header] = value
    return response


@router.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    """Serve the admin interface."""
    # Redirect to setup if first launch
    if is_setup_required():
        return RedirectResponse(url="/setup", status_code=302)
    if is_admin_auth_required() and not is_authenticated(request):
        return RedirectResponse(url="/admin/login", status_code=302)
    # Only allow admin users to access the admin page
    if is_admin_auth_required() and not is_admin(request):
        return RedirectResponse(url="/", status_code=302)
    response = templates.TemplateResponse("admin.html", {
        "request": request,
        "auth_enabled": is_admin_auth_required(),
    })
    # Prevent browser from caching authenticated pages (fixes back-button after logout)
    for header, value in NO_CACHE_HEADERS.items():
        response.headers[header] = value
    return response


@router.get("/auth/check")
async def check_auth(request: Request):
    """Check if user is authenticated. Used by frontend to detect stale sessions."""
    if not is_admin_auth_required():
        return {"authenticated": True, "auth_required": False}
    return {
        "authenticated": is_authenticated(request),
        "auth_required": True,
    }


@router.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Serve the admin login page."""
    # Redirect to setup if first launch
    if is_setup_required():
        return RedirectResponse(url="/setup", status_code=302)
    if not is_admin_auth_required():
        return RedirectResponse(url="/admin", status_code=302)
    if is_authenticated(request):
        return RedirectResponse(url="/admin", status_code=302)
    csrf_token = generate_csrf_token()
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": None,
        "username": None,
        "csrf_token": csrf_token,
    })


@router.post("/admin/login")
@limiter.limit(lambda: get_settings().rate_limit_auth)
async def admin_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(...),
):
    """Handle admin login with rate limiting and CSRF protection."""
    audit = get_audit_logger()
    client_ip = get_client_ip(request)

    # Validate CSRF token
    if not validate_csrf_token(csrf_token):
        new_csrf = generate_csrf_token()
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid or expired form. Please try again.", "username": username, "csrf_token": new_csrf},
            status_code=400,
        )

    # Redirect to setup if first launch
    if is_setup_required():
        return RedirectResponse(url="/setup", status_code=302)

    if not is_admin_auth_required():
        return RedirectResponse(url="/admin", status_code=302)

    user = verify_password(username, password)
    if user:
        # Create session using thread-safe function
        session_id = create_session({
            "username": user["username"],
            "role": user["role"],
        })

        audit.log_auth("login", success=True, ip_address=client_ip)

        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            key="pika_session",
            value=session_id,
            httponly=True,
            samesite="lax",
            secure=not get_settings().debug,  # Secure in production
            max_age=SESSION_MAX_AGE,
        )
        return response

    audit.log_auth("login", success=False, ip_address=client_ip)
    new_csrf = generate_csrf_token()
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid username or password", "username": username, "csrf_token": new_csrf},
        status_code=401,
    )


@router.get("/admin/logout")
async def admin_logout(request: Request):
    """Handle admin logout."""
    session_id = request.cookies.get("pika_session")
    if session_id:
        delete_session(session_id)

    audit = get_audit_logger()
    audit.log_auth("logout", success=True, ip_address=get_client_ip(request))

    response = RedirectResponse(url="/admin/login", status_code=302)
    response.delete_cookie("pika_session")
    return response


@router.get("/setup", response_class=HTMLResponse)
async def setup_page(request: Request):
    """Serve the setup page for first launch."""
    if not is_setup_required():
        return RedirectResponse(url="/admin", status_code=302)
    csrf_token = generate_csrf_token()
    return templates.TemplateResponse("setup.html", {"request": request, "error": None, "csrf_token": csrf_token})


@router.post("/setup")
async def setup_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
    generate_api_key: bool = Form(False),
    csrf_token: str = Form(...),
):
    """Handle setup form submission with CSRF protection."""
    # Validate CSRF token first
    if not validate_csrf_token(csrf_token):
        new_csrf = generate_csrf_token()
        return templates.TemplateResponse(
            "setup.html",
            {"request": request, "error": "Invalid or expired form. Please try again.", "username": username, "csrf_token": new_csrf},
            status_code=400,
        )

    if not is_setup_required():
        return RedirectResponse(url="/admin", status_code=302)

    # Validate passwords match
    if password != password_confirm:
        new_csrf = generate_csrf_token()
        return templates.TemplateResponse(
            "setup.html",
            {"request": request, "error": "Passwords do not match", "username": username, "csrf_token": new_csrf},
            status_code=400,
        )

    # Validate password length
    if len(password) < 6:
        new_csrf = generate_csrf_token()
        return templates.TemplateResponse(
            "setup.html",
            {"request": request, "error": "Password must be at least 6 characters", "username": username, "csrf_token": new_csrf},
            status_code=400,
        )

    # Create admin user
    auth = get_auth_service()
    try:
        auth.create_user(username, password, "admin")
    except ValueError as e:
        new_csrf = generate_csrf_token()
        return templates.TemplateResponse(
            "setup.html",
            {"request": request, "error": str(e), "username": username, "csrf_token": new_csrf},
            status_code=400,
        )

    # Generate API key if requested
    config = get_app_config()
    api_key = None
    if generate_api_key:
        api_key = secrets.token_urlsafe(32)
        config.set_api_key(api_key)

    # Mark setup as complete in config (for backwards compatibility)
    config.set("setup_complete", True)

    # Log the setup
    audit = get_audit_logger()
    audit.log_admin_action("setup_complete", {"username": username, "api_key_generated": generate_api_key})

    # Create session and redirect to admin
    session_id = create_session({
        "username": username,
        "role": "admin",
    })

    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        key="pika_session",
        value=session_id,
        httponly=True,
        samesite="lax",
        secure=not get_settings().debug,
        max_age=SESSION_MAX_AGE,
    )
    return response


@router.get("/documents")
async def list_documents(
    processor: DocumentProcessor = Depends(get_document_processor),
    _: bool = Depends(require_admin_auth),
) -> list[dict]:
    """List all documents in the documents directory."""
    documents = processor.list_documents()
    return [
        {
            "filename": doc.filename,
            "path": doc.path,
            "size_bytes": doc.size_bytes,
            "modified_at": doc.modified_at.isoformat(),
            "file_type": doc.file_type,
        }
        for doc in documents
    ]


def _sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal attacks.

    Returns the basename only, stripping any directory components.
    Raises ValueError if filename is invalid.
    """
    # Get just the filename, no path components
    safe_name = Path(filename).name

    # Reject empty names
    if not safe_name:
        raise ValueError("Empty filename")

    # Reject hidden files (starting with .)
    if safe_name.startswith("."):
        raise ValueError("Hidden files not allowed")

    # Reject path traversal attempts
    if ".." in safe_name or "/" in safe_name or "\\" in safe_name:
        raise ValueError("Invalid filename characters")

    # Reject null bytes
    if "\x00" in safe_name:
        raise ValueError("Invalid filename")

    return safe_name


@router.post("/upload")
async def upload_document(
    file: UploadFile,
    settings: Settings = Depends(get_settings),
    _: bool = Depends(require_admin_or_api_auth),
) -> dict:
    """Upload a document to the documents directory."""
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename required")

    # Sanitize filename to prevent path traversal
    try:
        filename = _sanitize_filename(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {e}")

    # Validate file extension
    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
    file_ext = Path(filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {', '.join(allowed_extensions)}",
        )

    # Check file size by reading content length header or actual content
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()

    if len(content) > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.max_upload_size_mb}MB",
        )

    # Ensure documents directory exists
    docs_dir = Path(settings.documents_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Save file (content already read for size check)
    file_path = docs_dir / filename

    try:
        file_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    return {
        "filename": filename,
        "size_bytes": len(content),
        "status": "uploaded",
    }


@router.delete("/documents/{filename}")
async def delete_document(
    filename: str,
    settings: Settings = Depends(get_settings),
    _: bool = Depends(require_admin_or_api_auth),
) -> dict:
    """Delete a document from the documents directory."""
    docs_dir = Path(settings.documents_dir)
    file_path = docs_dir / filename

    # Security: ensure the file is within documents directory
    try:
        file_path = file_path.resolve()
        docs_dir = docs_dir.resolve()
        if not str(file_path).startswith(str(docs_dir)):
            raise HTTPException(status_code=400, detail="Invalid filename")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        file_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")

    # Audit log
    audit = get_audit_logger()
    audit.log_admin_action("delete_document", {"filename": filename})

    return {"filename": filename, "status": "deleted"}


@router.get("/admin/logs", response_class=HTMLResponse)
async def admin_logs_page(
    request: Request,
    page: int = 1,
    per_page: int = 50,
):
    """Serve the audit logs page with pagination."""
    if is_admin_auth_required() and not is_authenticated(request):
        return RedirectResponse(url="/admin/login", status_code=302)
    if is_admin_auth_required() and not is_admin(request):
        return RedirectResponse(url="/", status_code=302)

    # Validate pagination params
    page = max(1, page)
    per_page = max(10, min(100, per_page))  # Clamp between 10-100

    audit = get_audit_logger()
    total_count = audit.get_total_count()
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    page = min(page, total_pages)  # Don't exceed total pages

    offset = (page - 1) * per_page
    logs = audit.get_recent_logs(limit=per_page, offset=offset)

    response = templates.TemplateResponse(
        "logs.html",
        {
            "request": request,
            "logs": logs,
            "page": page,
            "per_page": per_page,
            "total_count": total_count,
            "total_pages": total_pages,
        },
    )
    # Prevent browser from caching authenticated pages
    for header, value in NO_CACHE_HEADERS.items():
        response.headers[header] = value
    return response


@router.get("/api/admin/logs")
async def get_audit_logs_api(
    request: Request,
    page: int = 1,
    per_page: int = 50,
):
    """Get audit logs as JSON with pagination."""
    if is_admin_auth_required() and not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required")
    if is_admin_auth_required() and not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")

    # Validate pagination params
    page = max(1, page)
    per_page = max(10, min(100, per_page))

    audit = get_audit_logger()
    total_count = audit.get_total_count()
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    page = min(page, total_pages)

    offset = (page - 1) * per_page
    logs = audit.get_recent_logs(limit=per_page, offset=offset)

    return {
        "logs": logs,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_count": total_count,
            "total_pages": total_pages,
        },
    }


# Background backup state
@dataclass
class BackupStatus:
    """Status of a backup operation."""
    backup_id: str
    status: str = "preparing"  # preparing, running, completed, error
    progress: int = 0  # 0-100
    total_files: int = 0
    processed_files: int = 0
    filename: str | None = None
    error: str | None = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "backup_id": self.backup_id,
            "status": self.status,
            "progress": self.progress,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "filename": self.filename,
            "error": self.error,
        }


_active_backup: BackupStatus | None = None
_backup_task: asyncio.Task | None = None
_backup_file_path: Path | None = None
_backup_lock: asyncio.Lock | None = None


def _get_backup_lock() -> asyncio.Lock:
    """Get or create the backup lock (must be created in async context)."""
    global _backup_lock
    if _backup_lock is None:
        _backup_lock = asyncio.Lock()
    return _backup_lock


def get_active_backup() -> BackupStatus | None:
    """Get the current backup status."""
    return _active_backup


def is_backup_running() -> bool:
    """Check if a backup is currently running."""
    return _backup_task is not None and not _backup_task.done()


async def _run_backup(settings: Settings) -> None:
    """Run backup in background thread."""
    global _active_backup, _backup_file_path
    import zipfile
    import tempfile
    import json
    from pika import __version__

    if _active_backup is None:
        return

    try:
        data_dir = Path(settings.chroma_persist_dir).parent
        docs_dir = Path(settings.documents_dir)
        chroma_dir = Path(settings.chroma_persist_dir)

        # Count total files for progress
        total_files = 0
        if docs_dir.exists():
            total_files += sum(1 for _ in docs_dir.rglob("*") if _.is_file())
        if chroma_dir.exists():
            total_files += sum(1 for _ in chroma_dir.rglob("*") if _.is_file())
        # Add config files
        total_files += 6  # config, audit, history, feedback, db, metadata

        _active_backup.total_files = total_files
        _active_backup.status = "running"
        processed = 0

        # Create temp file for the backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = data_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"pika_backup_{timestamp}.zip"
        _backup_file_path = backup_path

        def update_progress():
            nonlocal processed
            processed += 1
            _active_backup.processed_files = processed
            _active_backup.progress = int((processed / max(total_files, 1)) * 100)

        # Create ZIP file on disk (not in memory)
        with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add documents
            if docs_dir.exists():
                for file_path in docs_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = f"documents/{file_path.relative_to(docs_dir)}"
                        zf.write(file_path, arcname)
                        update_progress()
                        await asyncio.sleep(0)  # Yield to event loop

            # Add ChromaDB data
            if chroma_dir.exists():
                for file_path in chroma_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = f"chroma/{file_path.relative_to(chroma_dir)}"
                        zf.write(file_path, arcname)
                        update_progress()
                        await asyncio.sleep(0)  # Yield to event loop

            # Add config files
            config_path = data_dir / "config.json"
            if config_path.exists():
                zf.write(config_path, "config.json")
            update_progress()

            audit_path = Path(settings.audit_log_path)
            if audit_path.exists():
                zf.write(audit_path, "audit.log")
            update_progress()

            history_path = data_dir / "history.json"
            if history_path.exists():
                zf.write(history_path, "history.json")
            update_progress()

            feedback_path = data_dir / "feedback.json"
            if feedback_path.exists():
                zf.write(feedback_path, "feedback.json")
            update_progress()

            db_path = data_dir / "pika.db"
            if db_path.exists():
                zf.write(db_path, "pika.db")
            update_progress()

            # Add metadata
            metadata = {
                "version": __version__,
                "created_at": datetime.now().isoformat(),
                "includes": {
                    "documents": docs_dir.exists(),
                    "chroma": chroma_dir.exists(),
                    "config": config_path.exists(),
                    "database": db_path.exists(),
                    "history": history_path.exists(),
                    "feedback": feedback_path.exists(),
                }
            }
            zf.writestr("backup_meta.json", json.dumps(metadata, indent=2))
            update_progress()

        _active_backup.status = "completed"
        _active_backup.progress = 100
        _active_backup.filename = backup_path.name
        _active_backup.completed_at = datetime.now()
        logger.info(f"Backup completed: {backup_path}")

    except Exception as e:
        logger.error(f"Backup failed: {e}")
        _active_backup.status = "error"
        _active_backup.error = str(e)


@router.post("/admin/backup/start")
@limiter.limit(lambda: get_settings().rate_limit_admin)
async def start_backup(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Start a background backup operation."""
    global _active_backup, _backup_task, _backup_file_path

    if is_admin_auth_required() and not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required")
    if is_admin_auth_required() and not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")

    # Use lock to prevent race conditions when starting backups
    async with _get_backup_lock():
        if is_backup_running():
            return {"started": False, "message": "Backup already in progress", "backup": _active_backup.to_dict()}

        # Clean up any existing backup file before starting new one
        if _backup_file_path and _backup_file_path.exists():
            try:
                _backup_file_path.unlink()
                logger.info(f"Cleaned up old backup file: {_backup_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up old backup: {e}")
        _backup_file_path = None

        # Create new backup status
        backup_id = f"backup-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        _active_backup = BackupStatus(backup_id=backup_id)

        # Start background task
        _backup_task = asyncio.create_task(_run_backup(settings))

    # Audit log
    audit = get_audit_logger()
    audit.log_admin_action("backup_started", {"backup_id": backup_id})

    return {"started": True, "message": "Backup started", "backup": _active_backup.to_dict()}


@router.get("/admin/backup/status")
async def get_backup_status(request: Request):
    """Get the status of the current backup operation."""
    if is_admin_auth_required() and not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required")

    if _active_backup is None:
        return {"active": False}

    return {"active": True, "backup": _active_backup.to_dict()}


@router.get("/admin/backup/download")
async def download_backup_file(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Download a completed backup file."""
    from fastapi.responses import FileResponse

    if is_admin_auth_required() and not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required")
    if is_admin_auth_required() and not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")

    if _active_backup is None or _active_backup.status != "completed":
        raise HTTPException(status_code=404, detail="No completed backup available")

    if _backup_file_path is None or not _backup_file_path.exists():
        raise HTTPException(status_code=404, detail="Backup file not found")

    # Audit log
    audit = get_audit_logger()
    audit.log_admin_action("backup_download", {"filename": _active_backup.filename})

    return FileResponse(
        path=_backup_file_path,
        filename=_active_backup.filename,
        media_type="application/zip",
    )


@router.delete("/admin/backup")
@limiter.limit(lambda: get_settings().rate_limit_admin)
async def clear_backup(request: Request):
    """Clear the backup status and delete the backup file."""
    global _active_backup, _backup_file_path

    if is_admin_auth_required() and not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required")

    # Delete the backup file if it exists
    if _backup_file_path and _backup_file_path.exists():
        try:
            _backup_file_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete backup file: {e}")

    _active_backup = None
    _backup_file_path = None

    return {"status": "cleared"}


# Legacy endpoint for direct download (kept for compatibility but now just redirects)
@router.get("/admin/backup")
async def download_backup(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Download a backup - redirects to the new async flow if no backup ready."""
    from fastapi.responses import FileResponse, RedirectResponse

    if is_admin_auth_required() and not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required")
    if is_admin_auth_required() and not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")

    # If there's a completed backup, serve it
    if _active_backup and _active_backup.status == "completed" and _backup_file_path and _backup_file_path.exists():
        audit = get_audit_logger()
        audit.log_admin_action("backup_download", {"filename": _active_backup.filename})
        return FileResponse(
            path=_backup_file_path,
            filename=_active_backup.filename,
            media_type="application/zip",
        )

    # Otherwise, return error suggesting to use the async flow
    raise HTTPException(
        status_code=400,
        detail="No backup ready. Use POST /admin/backup/start to create a backup first."
    )


def _safe_extract_path(base_dir: Path, rel_path: str) -> Path | None:
    """Safely resolve a path ensuring it stays within base_dir (prevents Zip Slip).

    Returns None if the path would escape the base directory.
    """
    try:
        # Resolve both paths to absolute
        base_resolved = base_dir.resolve()
        dest_path = (base_dir / rel_path).resolve()
        # Use relative_to which raises ValueError if dest is not under base
        dest_path.relative_to(base_resolved)
        return dest_path
    except (ValueError, OSError):
        return None


@router.post("/admin/restore")
@limiter.limit(lambda: get_settings().rate_limit_admin)
async def restore_backup(
    request: Request,
    file: UploadFile,
    settings: Settings = Depends(get_settings),
):
    """Restore PIKA data from a backup zip."""
    import io
    import shutil
    import zipfile

    if is_admin_auth_required() and not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required")
    if is_admin_auth_required() and not is_admin(request):
        raise HTTPException(status_code=403, detail="Admin access required")

    # Prevent restore while backup is running
    async with _get_backup_lock():
        if is_backup_running():
            raise HTTPException(status_code=409, detail="Cannot restore while backup is in progress")

    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive")

    data_dir = Path(settings.chroma_persist_dir).parent
    docs_dir = Path(settings.documents_dir)
    chroma_dir = Path(settings.chroma_persist_dir)

    try:
        content = await file.read()
        zip_buffer = io.BytesIO(content)

        with zipfile.ZipFile(zip_buffer, "r") as zf:
            # Validate zip contents
            namelist = zf.namelist()
            has_documents = any(n.startswith("documents/") for n in namelist)
            has_chroma = any(n.startswith("chroma/") for n in namelist)

            if not has_documents and not has_chroma and "config.json" not in namelist:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid backup file: missing expected data",
                )

            # Validate all paths before extracting (Zip Slip prevention)
            for name in namelist:
                if name.startswith("documents/") and not name.endswith("/"):
                    rel_path = name[len("documents/"):]
                    if _safe_extract_path(docs_dir, rel_path) is None:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid path in archive: {name}",
                        )
                elif name.startswith("chroma/") and not name.endswith("/"):
                    rel_path = name[len("chroma/"):]
                    if _safe_extract_path(chroma_dir, rel_path) is None:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid path in archive: {name}",
                        )

            # Clear and restore documents
            if has_documents:
                # Clear contents but don't delete the directory (it may be a mount point)
                if docs_dir.exists():
                    for item in docs_dir.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                else:
                    docs_dir.mkdir(parents=True, exist_ok=True)

                for name in namelist:
                    if name.startswith("documents/") and not name.endswith("/"):
                        rel_path = name[len("documents/"):]
                        dest_path = _safe_extract_path(docs_dir, rel_path)
                        if dest_path:
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(name) as src, open(dest_path, "wb") as dst:
                                dst.write(src.read())

            # Clear and restore ChromaDB
            if has_chroma:
                # IMPORTANT: Shutdown RAG engine BEFORE touching ChromaDB files
                # This releases SQLite connections so we can overwrite the database
                from pika.services.rag import prepare_for_restore
                prepare_for_restore()

                # Small delay to ensure file handles are released
                await asyncio.sleep(0.5)

                # Clear contents but don't delete the directory (it may be a mount point)
                if chroma_dir.exists():
                    for item in chroma_dir.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                else:
                    chroma_dir.mkdir(parents=True, exist_ok=True)

                for name in namelist:
                    if name.startswith("chroma/") and not name.endswith("/"):
                        rel_path = name[len("chroma/"):]
                        dest_path = _safe_extract_path(chroma_dir, rel_path)
                        if dest_path:
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(name) as src, open(dest_path, "wb") as dst:
                                dst.write(src.read())

                # Fix file permissions - ensure all ChromaDB files are writable
                # This is necessary because zip extraction can lose write permissions
                import os
                import stat

                # First, fix the chroma directory itself
                chroma_dir.chmod(0o777)
                logger.info(f"[Restore] Set chroma_dir permissions: {chroma_dir}")

                # Then fix all subdirectories and files
                files_fixed = 0
                dirs_fixed = 0
                for root, dirs, files in os.walk(chroma_dir):
                    for d in dirs:
                        dir_path = Path(root) / d
                        dir_path.chmod(0o777)  # Full permissions for directories
                        dirs_fixed += 1
                    for f in files:
                        file_path = Path(root) / f
                        file_path.chmod(0o666)  # Read/write for files
                        files_fixed += 1

                logger.info(f"[Restore] ChromaDB restored to {chroma_dir}: {files_fixed} files, {dirs_fixed} dirs (permissions fixed)")

                # Delete SQLite journal files - these can cause "readonly database" errors
                # if they contain stale state from the backup source environment
                journal_extensions = ["-wal", "-shm", "-journal"]
                journals_deleted = 0
                for root, dirs, files in os.walk(chroma_dir):
                    for f in files:
                        if any(f.endswith(ext) for ext in journal_extensions):
                            journal_path = Path(root) / f
                            try:
                                journal_path.unlink()
                                journals_deleted += 1
                                logger.debug(f"[Restore] Deleted journal file: {journal_path}")
                            except Exception as e:
                                logger.warning(f"[Restore] Failed to delete journal file {journal_path}: {e}")
                if journals_deleted > 0:
                    logger.info(f"[Restore] Deleted {journals_deleted} SQLite journal file(s)")

            # Restore config.json
            if "config.json" in namelist:
                with zf.open("config.json") as src:
                    config_path = data_dir / "config.json"
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(config_path, "wb") as dst:
                        dst.write(src.read())

            # Restore history.json
            if "history.json" in namelist:
                with zf.open("history.json") as src:
                    with open(data_dir / "history.json", "wb") as dst:
                        dst.write(src.read())

            # Restore feedback.json
            if "feedback.json" in namelist:
                with zf.open("feedback.json") as src:
                    with open(data_dir / "feedback.json", "wb") as dst:
                        dst.write(src.read())

            # Restore user database
            if "pika.db" in namelist:
                with zf.open("pika.db") as src:
                    db_path = data_dir / "pika.db"
                    with open(db_path, "wb") as dst:
                        dst.write(src.read())
                # Reset database singleton to pick up restored DB
                import pika.services.database as db_module
                db_module._database = None

            # Note: We don't restore audit.log to preserve the current audit trail

            # Reset singletons to pick up restored data
            # RAG engine - only reset if we didn't already (prepare_for_restore handles it for chroma restores)
            if not has_chroma:
                import pika.services.rag as rag_module
                rag_module._rag_engine = None
                logger.info("[Restore] RAG engine singleton reset - will be recreated on next access")

            # Reset history service
            import pika.services.history as history_module
            history_module._history_service = None

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file")
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.exception(f"Restore failed: {e}")
        raise HTTPException(status_code=500, detail=f"Restore failed: {type(e).__name__}: {e}")

    # Audit log
    audit = get_audit_logger()
    audit.log_admin_action("backup_restore", {"filename": file.filename})

    # Always tell user to re-index after restore
    # (ChromaDB files are included in backup but can't be hot-loaded due to SQLite limitations)
    message = "Backup restored successfully. Click 'Refresh Index' to rebuild the search index."

    return {
        "status": "restored",
        "message": message,
    }
