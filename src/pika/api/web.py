"""Web interface routes for PIKA."""

import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from pika.config import Settings, get_settings
from pika.services.audit import get_audit_logger
from pika.services.documents import DocumentInfo, DocumentProcessor, get_document_processor

router = APIRouter()

# Templates directory
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Session storage (simple in-memory for basic security)
_sessions: dict[str, bool] = {}


def is_admin_auth_required() -> bool:
    """Check if admin authentication is required."""
    settings = get_settings()
    return settings.pika_admin_password is not None


def is_authenticated(request: Request) -> bool:
    """Check if the request is authenticated for admin access."""
    if not is_admin_auth_required():
        return True
    session_id = request.cookies.get("pika_session")
    return session_id is not None and _sessions.get(session_id, False)


def require_admin_auth(request: Request) -> bool:
    """Dependency to require admin authentication."""
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required")
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
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    """Serve the admin interface."""
    if is_admin_auth_required() and not is_authenticated(request):
        return RedirectResponse(url="/admin/login", status_code=302)
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "auth_enabled": is_admin_auth_required(),
    })


@router.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    """Serve the admin login page."""
    if not is_admin_auth_required():
        return RedirectResponse(url="/admin", status_code=302)
    if is_authenticated(request):
        return RedirectResponse(url="/admin", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@router.post("/admin/login")
async def admin_login(request: Request, password: str = Form(...)):
    """Handle admin login."""
    settings = get_settings()
    audit = get_audit_logger()
    client_ip = get_client_ip(request)

    if not is_admin_auth_required():
        return RedirectResponse(url="/admin", status_code=302)

    if secrets.compare_digest(password, settings.pika_admin_password):
        # Create session
        session_id = secrets.token_urlsafe(32)
        _sessions[session_id] = True

        audit.log_auth("login", success=True, ip_address=client_ip)

        response = RedirectResponse(url="/admin", status_code=302)
        response.set_cookie(
            key="pika_session",
            value=session_id,
            httponly=True,
            samesite="lax",
            max_age=86400,  # 24 hours
        )
        return response
    else:
        audit.log_auth("login", success=False, ip_address=client_ip)
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid password"},
            status_code=401,
        )


@router.get("/admin/logout")
async def admin_logout(request: Request):
    """Handle admin logout."""
    session_id = request.cookies.get("pika_session")
    if session_id and session_id in _sessions:
        del _sessions[session_id]

    audit = get_audit_logger()
    audit.log_auth("logout", success=True, ip_address=get_client_ip(request))

    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("pika_session")
    return response


@router.get("/documents")
async def list_documents(
    processor: DocumentProcessor = Depends(get_document_processor),
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


@router.post("/upload")
async def upload_document(
    file: UploadFile,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Upload a document to the documents directory."""
    # Validate file extension
    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {', '.join(allowed_extensions)}",
        )

    # Ensure documents directory exists
    docs_dir = Path(settings.documents_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    file_path = docs_dir / file.filename

    try:
        content = await file.read()
        file_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    return {
        "filename": file.filename,
        "size_bytes": len(content),
        "status": "uploaded",
    }


@router.delete("/documents/{filename}")
async def delete_document(
    filename: str,
    settings: Settings = Depends(get_settings),
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
async def admin_logs_page(request: Request):
    """Serve the audit logs page."""
    if is_admin_auth_required() and not is_authenticated(request):
        return RedirectResponse(url="/admin/login", status_code=302)

    audit = get_audit_logger()
    logs = audit.get_recent_logs(100)

    return templates.TemplateResponse(
        "logs.html",
        {"request": request, "logs": logs},
    )
