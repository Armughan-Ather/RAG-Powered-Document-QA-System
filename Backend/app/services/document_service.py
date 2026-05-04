"""
app/services/document_service.py

Business logic for document operations.
Orchestrates the full pipeline: validate -> load -> clean -> chunk -> store.

This is the layer between routes and the lower-level utilities.
Routes call services. Services call utils/vectorstore/repository.
"""

import logging
import uuid
from typing import List

from fastapi import UploadFile

from app.core.config import Settings
from app.core.exceptions import DocumentNotFoundException, DocumentProcessingException
from app.repository import document_repository
from app.rag.ingestion import load_uploaded_file, clean_documents, split_documents
from app.utils.file_validator import validate_upload
from app.vectorstore import chroma_store

logger = logging.getLogger(__name__)


async def process_upload(file: UploadFile, settings: Settings) -> dict:
    """
    Full document upload pipeline.

    Steps:
      1. Validate file (extension + size).
      2. Generate a unique document_id.
      3. Save metadata as "processing" in repository.
      4. Load file into LangChain Documents (PyPDFLoader / TextLoader).
      5. Clean the documents (remove noise, ligatures, etc.).
      6. Chunk into overlapping pieces with metadata.
      7. Embed and store chunks in Chroma.
      8. Update metadata status to "ready".

    Args:
        file:     FastAPI UploadFile from the request.
        settings: App settings.

    Returns:
        Dict with upload result metadata:
        {
            "document_id": "abc123",
            "filename": "invoice.pdf",
            "pages": 5,
            "chunks": 12,
            "message": "Document processed successfully."
        }
    """
    # Step 1: Validate
    file_info = await validate_upload(file, settings)
    filename = file_info["filename"]

    # Step 2: Generate ID
    document_id = str(uuid.uuid4())

    # Step 3: Save initial metadata (status=processing)
    document_repository.save_document({
        "document_id": document_id,
        "filename": filename,
        "extension": file_info["extension"],
        "file_size_bytes": file_info["size_bytes"],
        "pages": 0,
        "chunks": 0,
        "status": "processing",
    })

    try:
        # Step 4: Load into LangChain Documents
        doc_id, documents = await load_uploaded_file(
            file=file,
            settings=settings,
            document_id=document_id,
        )
        pages_count = len(documents)

        # Step 5: Clean
        documents = clean_documents(documents)

        # Step 6: Chunk
        chunks = split_documents(documents, settings)
        chunks_count = len(chunks)

        if chunks_count == 0:
            raise DocumentProcessingException(
                message=f"No chunks produced from '{filename}'. File may be empty.",
                details={"filename": filename, "document_id": document_id},
            )

        # Step 7: Embed and store in Chroma
        chroma_store.add_documents(chunks)

        # Step 8: Update metadata to ready with final counts
        document_repository.update_document(document_id, {
            "status": "ready",
            "pages": pages_count,
            "chunks": chunks_count,
        })

        logger.info(
            "Upload complete: '%s' -> %d pages, %d chunks",
            filename, pages_count, chunks_count,
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "pages": pages_count,
            "chunks": chunks_count,
            "message": "Document processed successfully.",
        }

    except DocumentProcessingException:
        document_repository.update_status(document_id, "failed")
        raise

    except Exception as exc:
        document_repository.update_status(document_id, "failed")
        raise DocumentProcessingException(
            message=f"Failed to process '{filename}': {exc}",
            details={"filename": filename, "document_id": document_id, "error": str(exc)},
        )


def list_documents() -> dict:
    """
    Lists all uploaded documents with their metadata.

    Returns:
        {"count": int, "documents": List[dict]}
    """
    documents = document_repository.get_all_documents()
    return {
        "count": len(documents),
        "documents": documents,
    }


def get_document_detail(document_id: str) -> dict:
    """
    Gets detailed metadata for a single document.

    Args:
        document_id: The document UUID.

    Returns:
        Document metadata dict.

    Raises:
        DocumentNotFoundException if not found.
    """
    return document_repository.get_document(document_id)


def delete_document(document_id: str) -> dict:
    """
    Deletes a document from both the metadata store and Chroma.

    Args:
        document_id: The document UUID.

    Returns:
        {"message": str, "document_id": str}
    """
    # Verify it exists first
    doc = document_repository.get_document(document_id)

    # Delete from Chroma (vector store)
    chroma_store.delete_document(document_id)

    # Delete from metadata store
    document_repository.delete_document(document_id)

    logger.info("Document fully deleted: %s (%s)", document_id, doc["filename"])

    return {
        "message": f"Document '{doc['filename']}' deleted successfully.",
        "document_id": document_id,
    }
