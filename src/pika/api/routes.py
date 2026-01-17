"""API routes for PIKA."""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pika.services.app_config import get_app_config, AppConfigService
from pika.services.ollama import OllamaClient, get_ollama_client, _format_size, get_active_pull
from pika.services.rag import RAGEngine, get_rag_engine, Confidence, IndexedDocument

logger = logging.getLogger(__name__)

router = APIRouter()


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


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    ollama_connected: bool


@router.get("/health", response_model=HealthResponse)
async def health_check(
    ollama: OllamaClient = Depends(get_ollama_client),
) -> HealthResponse:
    """Check the health of PIKA and its dependencies."""
    ollama_ok = await ollama.health_check()
    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        ollama_connected=ollama_ok,
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
) -> CurrentModelResponse:
    """Set the current model."""
    config.set_current_model(request.model)
    return CurrentModelResponse(model=request.model)


@router.post("/models/pull")
async def pull_model(
    request: PullModelRequest,
    ollama: OllamaClient = Depends(get_ollama_client),
) -> StreamingResponse:
    """Pull a new model from Ollama registry."""
    async def generate_progress():
        try:
            async for progress in ollama.pull_model(request.model):
                yield f"data: {json.dumps(progress)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
    )


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
) -> IndexResponse:
    """Reindex all documents from the documents directory."""
    try:
        stats = rag.index_documents()
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


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    rag: RAGEngine = Depends(get_rag_engine),
) -> QueryResponse:
    """Query the RAG system with a question."""
    try:
        result = await rag.query(
            question=request.question,
            top_k=request.top_k,
        )
        return QueryResponse(
            answer=result.answer,
            sources=[
                SourceResponse(
                    filename=s.filename,
                    chunk_index=s.chunk_index,
                    content=s.content,
                    similarity=s.similarity,
                )
                for s in result.sources
            ],
            confidence=result.confidence,
        )
    except Exception as e:
        logger.exception(f"Query failed for question: {request.question[:50]}...")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
