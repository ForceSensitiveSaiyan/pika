"""Web interface routes for PIKA."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pika.config import Settings, get_settings
from pika.services.documents import DocumentInfo, DocumentProcessor, get_document_processor

router = APIRouter()

# Templates directory
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    """Serve the admin interface."""
    return templates.TemplateResponse("admin.html", {"request": request})


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

    return {"filename": filename, "status": "deleted"}
