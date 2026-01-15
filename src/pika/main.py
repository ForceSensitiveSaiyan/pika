"""PIKA - FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from pika import __version__
from pika.api.routes import router as api_router
from pika.api.web import router as web_router
from pika.config import get_settings

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()
    print(f"Starting {settings.app_name} v{__version__}")
    print(f"Ollama URL: {settings.ollama_base_url}")
    yield
    print("Shutting down PIKA")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Self-hosted RAG system for small businesses",
        version=__version__,
        lifespan=lifespan,
    )

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Include API routes
    app.include_router(api_router, prefix="/api/v1", tags=["api"])

    # Include web routes (must be after API routes)
    app.include_router(web_router, tags=["web"])

    return app


app = create_app()
