"""
app/main.py

Application entry point.

Responsibilities:
  1. Configure basic logging.
  2. Initialise FastAPI with metadata and lifespan.
  3. Register CORS middleware.
  4. Register global exception handlers.
  5. Mount API routers.
  6. Expose a /health endpoint.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.exceptions import (
    AppException,
    DocumentNotFoundException,
    DocumentProcessingException,
    VectorStoreException,
    EmbeddingException,
    RAGException,
    InvalidFileTypeException,
    FileTooLargeException,
)
from app.routes import document_routes, query_routes

# ── Logging ───────────────────────────────────────────────────────────────────
# Standard Python logging setup.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

settings = get_settings()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Startup and shutdown logic."""
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)

    # Ensure data directories exist
    settings.chroma_persist_path.mkdir(parents=True, exist_ok=True)
    settings.upload_path.mkdir(parents=True, exist_ok=True)

    yield

    logger.info("Shutting down %s", settings.APP_NAME)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Intelligent Document Analysis System — "
        "upload documents, ask questions, extract structured data, "
        "and perform semantic search powered by RAG and LangChain."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── CORS ──────────────────────────────────────────────────────────────────────
# Allows the frontend (running on a different port) to call this API.

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception Handlers ────────────────────────────────────────────────────────

@app.exception_handler(DocumentNotFoundException)
async def document_not_found_handler(request: Request, exc: DocumentNotFoundException) -> JSONResponse:
    return JSONResponse(status_code=404, content=_error_body(404, exc.message, exc.details))


@app.exception_handler(InvalidFileTypeException)
@app.exception_handler(FileTooLargeException)
async def upload_validation_handler(request: Request, exc: AppException) -> JSONResponse:
    return JSONResponse(status_code=422, content=_error_body(422, exc.message, exc.details))


@app.exception_handler(DocumentProcessingException)
@app.exception_handler(VectorStoreException)
@app.exception_handler(EmbeddingException)
@app.exception_handler(RAGException)
async def processing_error_handler(request: Request, exc: AppException) -> JSONResponse:
    logger.error("Processing error: %s", exc.message, exc_info=True)
    return JSONResponse(status_code=500, content=_error_body(500, exc.message, exc.details))


@app.exception_handler(AppException)
async def generic_app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    logger.error("Unexpected app error: %s", exc.message, exc_info=True)
    return JSONResponse(status_code=500, content=_error_body(500, exc.message, exc.details))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.critical("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content=_error_body(500, "An unexpected error occurred.", None))


def _error_body(status_code: int, message: str, details) -> dict:
    """Consistent error response shape."""
    return {
        "success": False,
        "status_code": status_code,
        "error": message,
        "details": details,
    }


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Basic health check endpoint."""
    return {
        "success": True,
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(document_routes.router, prefix=settings.API_V1_PREFIX, tags=["Documents"])
app.include_router(query_routes.router, prefix=settings.API_V1_PREFIX, tags=["Query"])
