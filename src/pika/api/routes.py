"""API routes for PIKA."""

import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pika.services.ollama import OllamaClient, get_ollama_client
from pika.services.rag import RAGEngine, get_rag_engine, Confidence

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


@router.get("/models")
async def list_models(
    ollama: OllamaClient = Depends(get_ollama_client),
) -> list[dict]:
    """List available Ollama models."""
    try:
        return await ollama.list_models()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")


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
