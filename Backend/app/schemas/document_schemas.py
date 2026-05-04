"""
app/schemas/document_schemas.py

Pydantic request/response models for document endpoints.
These are the API contracts -- they define exactly what the client
sends and receives. FastAPI uses these for automatic validation,
serialization, and Swagger documentation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Response Models ───────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    """Metadata for a single document (used in list and detail responses)."""
    document_id: str
    filename: str
    extension: str
    file_size_bytes: int
    pages: int
    chunks: int
    status: str = "ready"  # "processing" | "ready" | "failed"
    uploaded_at: str


class UploadResponse(BaseModel):
    """Response after successfully uploading and processing a document."""
    success: bool = True
    message: str
    document_id: str
    filename: str
    pages: int
    chunks: int


class DocumentListResponse(BaseModel):
    """Response for GET /documents."""
    success: bool = True
    count: int
    documents: List[DocumentInfo]


class DocumentDetailResponse(BaseModel):
    """Response for GET /documents/{id}."""
    success: bool = True
    document: DocumentInfo


class DeleteResponse(BaseModel):
    """Response after deleting a document."""
    success: bool = True
    message: str
    document_id: str
