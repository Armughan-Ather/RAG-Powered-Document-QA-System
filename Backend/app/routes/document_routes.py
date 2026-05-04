"""
app/routes/document_routes.py

Document endpoints:
  POST   /documents/upload   -- Upload and process a document.
  GET    /documents           -- List all uploaded documents.
  GET    /documents/{id}      -- Get details of one document.
  DELETE /documents/{id}      -- Delete a document and its chunks.

Routes are thin controllers:
  - Parse the request.
  - Call the service.
  - Return the response.
  No business logic here.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.config import Settings, get_settings
from app.schemas.document_schemas import (
    DeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    UploadResponse,
)
from app.services import document_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Settings injected via FastAPI Depends — same pattern, no separate file needed
SettingsDep = Annotated[Settings, Depends(get_settings)]


# ── POST /documents/upload ────────────────────────────────────────────────────

@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    summary="Upload and process a document",
    description=(
        "Uploads a file (PDF, TXT), extracts text, cleans it, "
        "splits into chunks, generates embeddings, and stores "
        "in the vector database. Returns document metadata."
    ),
)
async def upload_document(
    settings: SettingsDep,
    file: UploadFile = File(
        ...,
        description="The document file to upload (.pdf, .txt).",
    ),
):
    """
    Full document ingestion pipeline:
      Validate -> Extract -> Clean -> Chunk -> Embed -> Store
    """
    result = await document_service.process_upload(file, settings)
    return UploadResponse(success=True, **result)


# ── GET /documents ────────────────────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
    description="Returns metadata for all documents that have been uploaded.",
)
async def list_documents():
    """Returns all document metadata sorted by upload date (newest first)."""
    result = document_service.list_documents()
    return DocumentListResponse(success=True, **result)


# ── GET /documents/{document_id} ──────────────────────────────────────────────

@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details",
    description="Returns detailed metadata for a single document by its ID.",
)
async def get_document(document_id: str):
    """Returns metadata for a specific document."""
    doc = document_service.get_document_detail(document_id)
    return DocumentDetailResponse(success=True, document=doc)


# ── DELETE /documents/{document_id} ───────────────────────────────────────────

@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    summary="Delete a document",
    description=(
        "Deletes a document from the metadata store and removes "
        "all its chunks from the vector database."
    ),
)
async def delete_document(document_id: str):
    """Removes a document and all its vector store chunks."""
    result = document_service.delete_document(document_id)
    return DeleteResponse(success=True, **result)
