"""
app/core/exceptions.py

Custom exception hierarchy for the application.

Design principle:
  - All domain errors inherit from AppException.
  - HTTP-layer helpers (not_found_error, etc.) convert domain errors
    into proper FastAPI HTTPExceptions with structured JSON bodies.
  - Routes catch domain exceptions and call the HTTP helpers —
    keeping business logic completely decoupled from HTTP concerns.
"""

from typing import Any, Optional

from fastapi import HTTPException, status


# ── Base ──────────────────────────────────────────────────────────────────────

class AppException(Exception):
    """
    Base class for all application-level exceptions.
    Carry a human-readable message and optional structured details.
    """

    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        self.message = message
        self.details = details
        super().__init__(message)


# ── Domain Exceptions ─────────────────────────────────────────────────────────

class DocumentNotFoundException(AppException):
    """Raised when a requested document does not exist in the store."""


class DocumentProcessingException(AppException):
    """Raised when PDF extraction, cleaning, or chunking fails."""


class VectorStoreException(AppException):
    """Raised when Chroma read/write operations fail."""


class EmbeddingException(AppException):
    """Raised when the embedding model fails to encode text."""


class RAGException(AppException):
    """Raised when the LangChain RAG chain encounters an error."""


class InvalidFileTypeException(AppException):
    """Raised when an uploaded file has an unsupported extension."""


class FileTooLargeException(AppException):
    """Raised when an uploaded file exceeds MAX_FILE_SIZE_MB."""


# ── HTTP Exception Factories ──────────────────────────────────────────────────

def not_found_error(resource: str, identifier: str) -> HTTPException:
    """404 — resource not found."""
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "NOT_FOUND", "resource": resource, "identifier": identifier},
    )
