"""API routes for PIKA."""

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from pika.api.web import get_current_user, require_admin_auth, require_admin_or_api_auth, require_user_auth
from pika.services.auth import AuthService, get_auth_service
from pika.config import get_settings
from pika.services.app_config import AppConfigService, get_app_config
from pika.services.audit import get_audit_logger
from pika.services.history import HistoryService, get_history_service
from pika.services.ollama import (
    OllamaClient,
    _format_size,
    cancel_pull_task,
    get_active_pull,
    get_ollama_client,
    is_pull_running,
    start_pull_task,
)
from pika.services.rag import (
    Confidence,
    RAGEngine,
    cancel_query,
    clear_query_status,
    get_active_query,
    get_rag_engine,
    is_query_running,
    start_query_task,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limiter for API endpoints
limiter = Limiter(key_func=get_remote_address)


class GenerateRequest(BaseModel):
    """Request model for text generation."""

    prompt: str
    model: str | None = None
    system: str | None = None
    stream: bool = False


class GenerateResponse(BaseModel):
    """Response model for text generation."""

    response: str
    model: str


class OllamaHealthResponse(BaseModel):
    """Health status for Ollama."""

    connected: bool
    current_model: str | None = None
    model_loaded: bool = False
    error: str | None = None


class IndexHealthResponse(BaseModel):
    """Health status for the index."""

    document_count: int
    chunk_count: int


class DiskHealthResponse(BaseModel):
    """Health status for disk space."""

    data_dir: str
    free_bytes: int
    free_gb: float
    warning: bool = False  # True if < 1GB free


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str  # healthy, degraded, unhealthy
    ollama: OllamaHealthResponse
    index: IndexHealthResponse
    disk: DiskHealthResponse


@router.get("/health", response_model=HealthResponse)
async def health_check(
    ollama: OllamaClient = Depends(get_ollama_client),
    rag: RAGEngine = Depends(get_rag_engine),
    config: AppConfigService = Depends(get_app_config),
) -> HealthResponse:
    """Check the health of PIKA and its dependencies."""
    import shutil
    from pathlib import Path

    settings = get_settings()

    # Check Ollama
    ollama_health = OllamaHealthResponse(connected=False)
    try:
        ollama_ok = await ollama.health_check()
        current_model = config.get_current_model()
        model_loaded = False

        if ollama_ok:
            # Check if current model is available
            models = await ollama.list_models()
            model_names = [m.name for m in models]
            model_loaded = current_model in model_names

        ollama_health = OllamaHealthResponse(
            connected=ollama_ok,
            current_model=current_model,
            model_loaded=model_loaded,
        )
    except Exception as e:
        ollama_health = OllamaHealthResponse(
            connected=False,
            error=str(e),
        )

    # Check index
    try:
        stats = rag.get_stats()
        index_health = IndexHealthResponse(
            document_count=stats.total_documents,
            chunk_count=stats.total_chunks,
        )
    except Exception:
        index_health = IndexHealthResponse(document_count=0, chunk_count=0)

    # Check disk space
    data_dir = Path(settings.chroma_persist_dir).parent
    try:
        disk_usage = shutil.disk_usage(data_dir)
        free_gb = disk_usage.free / (1024**3)
        disk_health = DiskHealthResponse(
            data_dir=str(data_dir),
            free_bytes=disk_usage.free,
            free_gb=round(free_gb, 2),
            warning=free_gb < 1.0,
        )
    except Exception:
        disk_health = DiskHealthResponse(
            data_dir=str(data_dir),
            free_bytes=0,
            free_gb=0,
            warning=True,
        )

    # Determine overall status
    if not ollama_health.connected:
        status = "unhealthy"
    elif not ollama_health.model_loaded or disk_health.warning:
        status = "degraded"
    else:
        status = "healthy"

    return HealthResponse(
        status=status,
        ollama=ollama_health,
        index=index_health,
        disk=disk_health,
    )


class ModelResponse(BaseModel):
    """Response model for a single model."""

    name: str
    size: str
    size_bytes: int
    is_current: bool


class CurrentModelResponse(BaseModel):
    """Response model for the current model."""

    model: str


class SetModelRequest(BaseModel):
    """Request model for setting the current model."""

    model: str


class PullModelRequest(BaseModel):
    """Request model for pulling a new model."""

    model: str


@router.get("/models", response_model=list[ModelResponse])
async def list_models(
    ollama: OllamaClient = Depends(get_ollama_client),
    config: AppConfigService = Depends(get_app_config),
) -> list[ModelResponse]:
    """List available Ollama models."""
    try:
        models = await ollama.list_models()
        current_model = config.get_current_model()

        # Check if current model exists in available models
        model_names = [m.name for m in models]
        if current_model not in model_names and models:
            # Auto-select first available model
            current_model = models[0].name
            config.set_current_model(current_model)
            logger.info(f"Auto-selected model: {current_model}")

        return [
            ModelResponse(
                name=m.name,
                size=_format_size(m.size),
                size_bytes=m.size,
                is_current=(m.name == current_model),
            )
            for m in models
        ]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")


@router.get("/models/current", response_model=CurrentModelResponse)
async def get_current_model(
    config: AppConfigService = Depends(get_app_config),
) -> CurrentModelResponse:
    """Get the currently active model."""
    return CurrentModelResponse(model=config.get_current_model())


@router.post("/models/current", response_model=CurrentModelResponse)
async def set_current_model(
    request: SetModelRequest,
    config: AppConfigService = Depends(get_app_config),
    _: bool = Depends(require_admin_or_api_auth),
) -> CurrentModelResponse:
    """Set the current model."""
    old_model = config.get_current_model()
    config.set_current_model(request.model)

    # Audit log
    audit = get_audit_logger()
    audit.log_admin_action(
        "change_model",
        {
            "old_model": old_model,
            "new_model": request.model,
        },
    )

    return CurrentModelResponse(model=request.model)


class PullModelResponse(BaseModel):
    """Response model for starting a pull."""

    started: bool
    message: str


@router.post("/models/pull", response_model=PullModelResponse)
async def pull_model(
    request: PullModelRequest,
    ollama: OllamaClient = Depends(get_ollama_client),
    _: bool = Depends(require_admin_or_api_auth),
) -> PullModelResponse:
    """Pull a new model from Ollama registry."""
    if is_pull_running():
        pull = get_active_pull()
        return PullModelResponse(
            started=False,
            message=f"Already pulling {pull.model if pull else 'a model'}",
        )

    # Audit log
    audit = get_audit_logger()
    audit.log_admin_action("pull_model", {"model": request.model})

    # Start background task
    await start_pull_task(ollama, request.model)
    return PullModelResponse(started=True, message=f"Started pulling {request.model}")


class PullStatusResponse(BaseModel):
    """Response model for pull status."""

    active: bool
    model: str | None = None
    status: str | None = None
    completed: int = 0
    total: int = 0
    percent: int = 0
    error: str | None = None


@router.get("/models/pull/status", response_model=PullStatusResponse)
async def get_pull_status() -> PullStatusResponse:
    """Get the status of any active model pull."""
    pull = get_active_pull()
    if pull is None:
        return PullStatusResponse(active=False)
    return PullStatusResponse(
        active=True,
        model=pull.model,
        status=pull.status,
        completed=pull.completed,
        total=pull.total,
        percent=pull.percent,
        error=pull.error,
    )


class CancelPullResponse(BaseModel):
    """Response model for cancel pull."""

    cancelled: bool
    message: str


@router.post("/models/pull/cancel", response_model=CancelPullResponse)
async def cancel_pull(
    _: bool = Depends(require_admin_or_api_auth),
) -> CancelPullResponse:
    """Cancel an active model pull."""
    if not is_pull_running():
        return CancelPullResponse(
            cancelled=False,
            message="No model pull is currently running",
        )

    cancelled = cancel_pull_task()
    if cancelled:
        # Audit log
        audit = get_audit_logger()
        pull = get_active_pull()
        audit.log_admin_action("cancel_pull", {"model": pull.model if pull else "unknown"})

        return CancelPullResponse(
            cancelled=True,
            message="Model pull cancelled",
        )
    return CancelPullResponse(
        cancelled=False,
        message="Failed to cancel model pull",
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    ollama: OllamaClient = Depends(get_ollama_client),
) -> GenerateResponse | StreamingResponse:
    """Generate text using Ollama."""
    try:
        if request.stream:
            return StreamingResponse(
                ollama.generate_stream(
                    prompt=request.prompt,
                    model=request.model,
                    system=request.system,
                ),
                media_type="text/event-stream",
            )

        response = await ollama.generate(
            prompt=request.prompt,
            model=request.model,
            system=request.system,
        )
        return GenerateResponse(
            response=response,
            model=request.model or ollama.model,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Generation failed: {e}")


# --- RAG Endpoints ---


class IndexResponse(BaseModel):
    """Response model for index operations."""

    status: str
    total_documents: int
    total_chunks: int


class IndexStatsResponse(BaseModel):
    """Response model for index statistics."""

    total_documents: int
    total_chunks: int
    collection_name: str


class IndexedDocumentResponse(BaseModel):
    """Response model for an indexed document."""

    filename: str
    chunk_count: int


class QueryRequest(BaseModel):
    """Request model for RAG queries."""

    question: str
    top_k: int | None = None


class SourceResponse(BaseModel):
    """Response model for a source document."""

    filename: str
    chunk_index: int
    content: str
    similarity: float


class QueryResponse(BaseModel):
    """Response model for RAG query results."""

    answer: str
    sources: list[SourceResponse]
    confidence: Confidence


@router.post("/index", response_model=IndexResponse)
async def index_documents(
    rag: RAGEngine = Depends(get_rag_engine),
    _: bool = Depends(require_admin_or_api_auth),
) -> IndexResponse:
    """Reindex all documents from the documents directory."""
    try:
        stats = rag.index_documents()

        # Audit log
        audit = get_audit_logger()
        audit.log_admin_action(
            "index_documents",
            {
                "total_documents": stats.total_documents,
                "total_chunks": stats.total_chunks,
            },
        )

        return IndexResponse(
            status="indexed",
            total_documents=stats.total_documents,
            total_chunks=stats.total_chunks,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


@router.get("/index/stats", response_model=IndexStatsResponse)
async def get_index_stats(
    rag: RAGEngine = Depends(get_rag_engine),
    _: bool = Depends(require_admin_auth),
) -> IndexStatsResponse:
    """Get statistics about the current index."""
    try:
        stats = rag.get_stats()
        return IndexStatsResponse(
            total_documents=stats.total_documents,
            total_chunks=stats.total_chunks,
            collection_name=stats.collection_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")


@router.get("/documents", response_model=list[IndexedDocumentResponse])
async def get_indexed_documents(
    rag: RAGEngine = Depends(get_rag_engine),
    _: bool = Depends(require_admin_or_api_auth),
) -> list[IndexedDocumentResponse]:
    """Get list of indexed documents with their chunk counts."""
    try:
        documents = rag.get_indexed_documents()
        return [
            IndexedDocumentResponse(
                filename=doc.filename,
                chunk_count=doc.chunk_count,
            )
            for doc in documents
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {e}")


class QueryStartResponse(BaseModel):
    """Response model for starting a query."""

    query_id: str
    status: str


class QueryStatusResponse(BaseModel):
    """Response model for query status."""

    query_id: str | None
    question: str | None
    status: str  # pending, running, completed, error, none
    result: QueryResponse | None = None
    error: str | None = None


@router.post("/query", response_model=QueryStartResponse)
@limiter.limit(lambda: get_settings().rate_limit_query)
async def query_documents(
    request: Request,
    query: QueryRequest,
    _: bool = Depends(require_admin_or_api_auth),
) -> QueryStartResponse:
    """Start a background query to the RAG system with rate limiting."""
    query_id = str(uuid.uuid4())[:8]
    user = get_current_user(request)
    username = user.get("username") if user else None

    try:
        await start_query_task(
            question=query.question,
            query_id=query_id,
            top_k=query.top_k,
            username=username,
        )
        return QueryStartResponse(query_id=query_id, status="running")
    except Exception as e:
        logger.exception(f"Failed to start query: {query.question[:50]}...")
        raise HTTPException(status_code=500, detail=f"Failed to start query: {e}")


@router.get("/query/status", response_model=QueryStatusResponse)
async def get_query_status(
    request: Request,
    _: bool = Depends(require_admin_or_api_auth),
) -> QueryStatusResponse:
    """Get the status of the current or most recent query for the current user."""
    user = get_current_user(request)
    username = user.get("username") if user else None
    query = get_active_query(username)

    if query is None:
        return QueryStatusResponse(
            query_id=None,
            question=None,
            status="none",
        )

    result = None
    if query.result:
        result = QueryResponse(
            answer=query.result.answer,
            sources=[
                SourceResponse(
                    filename=s.filename,
                    chunk_index=s.chunk_index,
                    content=s.content,
                    similarity=s.similarity,
                )
                for s in query.result.sources
            ],
            confidence=query.result.confidence,
        )

    return QueryStatusResponse(
        query_id=query.query_id,
        question=query.question,
        status=query.status,
        result=result,
        error=query.error,
    )


@router.delete("/query/status")
async def clear_query(
    request: Request,
    _: bool = Depends(require_admin_or_api_auth),
) -> dict:
    """Clear the current query status for the current user."""
    user = get_current_user(request)
    username = user.get("username") if user else None
    clear_query_status(username)
    return {"status": "cleared"}


class CancelQueryResponse(BaseModel):
    """Response model for cancel query."""

    cancelled: bool
    message: str


@router.post("/query/cancel", response_model=CancelQueryResponse)
async def cancel_running_query(
    request: Request,
    _: bool = Depends(require_admin_or_api_auth),
) -> CancelQueryResponse:
    """Cancel the currently running query for the current user."""
    user = get_current_user(request)
    username = user.get("username") if user else None

    if not is_query_running(username):
        return CancelQueryResponse(
            cancelled=False,
            message="No query is currently running",
        )

    cancelled = cancel_query(username)
    if cancelled:
        return CancelQueryResponse(
            cancelled=True,
            message="Query cancelled successfully",
        )
    return CancelQueryResponse(
        cancelled=False,
        message="Failed to cancel query",
    )


# --- History & Feedback Endpoints ---


class HistoryEntry(BaseModel):
    """Response model for a history entry."""

    id: str
    question: str
    answer: str
    confidence: str
    sources: list[str]
    timestamp: str


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback."""

    query_id: str
    question: str
    answer: str
    rating: str  # "up" or "down"


@router.get("/history", response_model=list[HistoryEntry])
async def get_history(
    request: Request,
    limit: int = 20,
    history: HistoryService = Depends(get_history_service),
    _: bool = Depends(require_admin_or_api_auth),
) -> list[HistoryEntry]:
    """Get recent query history for the current user."""
    user = get_current_user(request)
    username = user.get("username") if user else None
    entries = history.get_history(limit=limit, username=username)
    return [
        HistoryEntry(
            id=e["id"],
            question=e["question"],
            answer=e["answer"],
            confidence=e["confidence"],
            sources=e["sources"],
            timestamp=e["timestamp"],
        )
        for e in entries
    ]


@router.delete("/history")
async def clear_history(
    request: Request,
    history: HistoryService = Depends(get_history_service),
    _: bool = Depends(require_admin_or_api_auth),
) -> dict:
    """Clear query history for the current user."""
    user = get_current_user(request)
    username = user.get("username") if user else None
    history.clear_history(username=username)
    return {"status": "cleared"}


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    history: HistoryService = Depends(get_history_service),
    _: bool = Depends(require_admin_or_api_auth),
) -> dict:
    """Submit feedback for a query."""
    if request.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="Rating must be 'up' or 'down'")

    history.add_feedback(
        query_id=request.query_id,
        question=request.question,
        answer=request.answer,
        rating=request.rating,
    )
    return {"status": "received", "rating": request.rating}


# --- User Management Endpoints ---


class UserResponse(BaseModel):
    """Response model for a user."""

    id: int
    username: str
    role: str
    is_active: bool
    created_at: str
    last_login: str | None


class CreateUserRequest(BaseModel):
    """Request model for creating a user."""

    username: str
    password: str
    role: str = "user"


class CreateUserResponse(BaseModel):
    """Response model for creating a user."""

    id: int
    username: str
    role: str


class UpdatePasswordRequest(BaseModel):
    """Request model for updating a password."""

    password: str


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    auth: AuthService = Depends(get_auth_service),
    _: bool = Depends(require_admin_auth),
) -> list[UserResponse]:
    """List all users (admin only)."""
    users = auth.list_users()
    return [
        UserResponse(
            id=u["id"],
            username=u["username"],
            role=u["role"],
            is_active=bool(u["is_active"]),
            created_at=u["created_at"],
            last_login=u["last_login"],
        )
        for u in users
    ]


@router.post("/users", response_model=CreateUserResponse)
async def create_user(
    request: CreateUserRequest,
    auth: AuthService = Depends(get_auth_service),
    _: bool = Depends(require_admin_auth),
) -> CreateUserResponse:
    """Create a new user (admin only)."""
    # Validate role
    if request.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'user'")

    # Validate password length
    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    # Validate username
    if len(request.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")

    try:
        user = auth.create_user(request.username, request.password, request.role)

        # Audit log
        audit = get_audit_logger()
        audit.log_admin_action("create_user", {"username": request.username, "role": request.role})

        return CreateUserResponse(
            id=user["id"],
            username=user["username"],
            role=user["role"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/users/{user_id}/password")
async def update_user_password(
    user_id: int,
    request: UpdatePasswordRequest,
    auth: AuthService = Depends(get_auth_service),
    _: bool = Depends(require_admin_auth),
) -> dict:
    """Update a user's password (admin only)."""
    # Validate password length
    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    # Check user exists
    user = auth.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    success = auth.update_password(user_id, request.password)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update password")

    # Audit log
    audit = get_audit_logger()
    audit.log_admin_action("update_password", {"user_id": user_id, "username": user["username"]})

    return {"status": "updated", "user_id": user_id}


@router.put("/users/{user_id}/toggle")
async def toggle_user(
    user_id: int,
    auth: AuthService = Depends(get_auth_service),
    _: bool = Depends(require_admin_auth),
) -> dict:
    """Enable/disable a user (admin only)."""
    # Check user exists
    user = auth.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        if user["is_active"]:
            auth.disable_user(user_id)
            new_status = False
        else:
            auth.enable_user(user_id)
            new_status = True

        # Audit log
        audit = get_audit_logger()
        audit.log_admin_action(
            "toggle_user",
            {"user_id": user_id, "username": user["username"], "is_active": new_status},
        )

        return {"status": "toggled", "user_id": user_id, "is_active": new_status}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    auth: AuthService = Depends(get_auth_service),
    _: bool = Depends(require_admin_auth),
) -> dict:
    """Delete a user (admin only)."""
    # Check user exists
    user = auth.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        success = auth.delete_user(user_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete user")

        # Audit log
        audit = get_audit_logger()
        audit.log_admin_action("delete_user", {"user_id": user_id, "username": user["username"]})

        return {"status": "deleted", "user_id": user_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
