"""API routes for PIKA."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pika.services.ollama import OllamaClient, get_ollama_client

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
