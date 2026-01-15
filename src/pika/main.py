"""PIKA - FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from pika import __version__
from pika.api.routes import router
from pika.config import get_settings


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

    app.include_router(router, prefix="/api/v1", tags=["api"])

    @app.get("/")
    async def root():
        return {
            "name": settings.app_name,
            "version": __version__,
            "docs": "/docs",
        }

    return app


app = create_app()
