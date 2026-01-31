"""PIKA - FastAPI application entry point."""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from pika import __version__
from pika.api.routes import router as api_router
from pika.api.web import router as web_router
from pika.config import get_settings

# Configure logging
settings = get_settings()
log_level = logging.DEBUG if settings.debug else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # Override any existing logging config
)

# Explicitly set log level for pika modules to ensure they're visible
logging.getLogger("pika").setLevel(log_level)
logging.getLogger("pika.services.rag").setLevel(log_level)
logging.getLogger("pika.services.ollama").setLevel(log_level)

logger = logging.getLogger(__name__)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"

# Rate limiter instance (shared across modules)
limiter = Limiter(key_func=get_remote_address)

# Graceful shutdown flag
_shutdown_event: asyncio.Event | None = None


def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded: {exc.detail}",
            "retry_after": getattr(exc, "retry_after", 60),
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler with graceful shutdown support."""
    global _shutdown_event
    from pika.services.rag import init_queue_processor, shutdown_queue_processor, preload_embedding_model
    from pika.api.web import init_session_cleanup, shutdown_session_cleanup
    from pika.services.ollama import close_http_client

    _shutdown_event = asyncio.Event()

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("Received shutdown signal, initiating graceful shutdown...")
        if _shutdown_event:
            _shutdown_event.set()

    # Register signal handlers (Unix-like systems)
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, shutdown_handler)
    except (NotImplementedError, RuntimeError):
        # NotImplementedError: Windows doesn't support add_signal_handler
        # RuntimeError: Can't run in non-main thread (e.g., during testing)
        pass

    logger.info(f"Starting {settings.app_name} v{__version__}")
    logger.info(f"Ollama URL: {settings.ollama_base_url}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Max concurrent queries: {settings.max_concurrent_queries}")

    # Pre-load embedding model (runs in background to not block startup)
    asyncio.create_task(preload_embedding_model())

    # Start background tasks
    await init_queue_processor()
    await init_session_cleanup()

    yield

    logger.info("Shutting down PIKA gracefully...")
    # Shutdown background tasks
    await shutdown_session_cleanup()
    await shutdown_queue_processor()
    # Close HTTP client pool
    await close_http_client()
    # Allow pending tasks to complete
    await asyncio.sleep(0.5)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="Self-hosted RAG system for small businesses",
        version=__version__,
        lifespan=lifespan,
    )

    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Include API routes
    app.include_router(api_router, prefix="/api/v1", tags=["api"])

    # Include web routes (must be after API routes)
    app.include_router(web_router, tags=["web"])

    return app


# Export limiter for use in route decorators
def get_limiter() -> Limiter:
    """Get the rate limiter instance."""
    return limiter


app = create_app()
