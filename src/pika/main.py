"""PIKA - FastAPI application entry point."""

import asyncio
import logging
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pythonjsonlogger import jsonlogger
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from pika import __version__
from pika.api.routes import router as api_router
from pika.api.web import router as web_router
from pika.config import get_settings
from pika.services.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    set_app_info,
)

# Configure structured JSON logging
settings = get_settings()
log_level = logging.DEBUG if settings.debug else logging.INFO


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = self.formatTime(record)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        if not log_record.get("message"):
            log_record["message"] = record.getMessage()


# Use JSON logging in production, human-readable in debug
if settings.debug:
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
else:
    formatter = CustomJsonFormatter()

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

logging.basicConfig(
    level=log_level,
    handlers=[handler],
    force=True,
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

    # Initialize metrics
    set_app_info(__version__, settings.ollama_model)

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

    # Add metrics middleware
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """Record request metrics for Prometheus."""
        # Skip metrics endpoint itself to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        # Normalize endpoint path for metrics (avoid high cardinality)
        endpoint = request.url.path
        # Group dynamic paths
        if endpoint.startswith("/api/v1/"):
            endpoint = "/api/v1/" + endpoint.split("/")[3] if len(endpoint.split("/")) > 3 else endpoint

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

        return response

    # Security headers middleware
    @app.middleware("http")
    async def security_headers_middleware(request: Request, call_next):
        """Add security headers to all responses."""
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    # Prometheus metrics endpoint
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        """Prometheus metrics endpoint."""
        return PlainTextResponse(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

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
