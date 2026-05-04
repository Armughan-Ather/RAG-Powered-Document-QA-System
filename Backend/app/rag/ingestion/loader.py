"""
app/rag/ingestion/loader.py

Loads files into LangChain Document objects using LangChain's built-in loaders.
"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from app.core.config import Settings
from app.core.exceptions import DocumentProcessingException

logger = logging.getLogger(__name__)


# ── Loader Registry ───────────────────────────────────────────────────────────
# Maps file extension → LangChain loader class.
# Add new formats here — the rest of the code adapts automatically.

LOADER_REGISTRY: dict = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


async def save_upload_to_disk(
    file: UploadFile,
    upload_dir: Path,
    document_id: str,
) -> Path:
    """
    Saves an uploaded file to disk with a unique name.

    Naming: {document_id}_{original_filename}
    This avoids collisions if two users upload "invoice.pdf".

    Args:
        file:        FastAPI UploadFile object.
        upload_dir:  Directory to save into (from settings.UPLOAD_DIR).
        document_id: Unique document ID (UUID).

    Returns:
        Path to the saved file on disk.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_name = file.filename or "unknown"
    safe_name = f"{document_id}_{original_name}"
    file_path = upload_dir / safe_name

    # Write the uploaded file to disk
    content = await file.read()
    file_path.write_bytes(content)
    await file.seek(0)  # Rewind in case anyone reads again

    logger.info("File saved: %s (%d bytes)", file_path.name, len(content))
    return file_path


def load_document_from_path(
    file_path: Path,
    document_id: str,
    original_filename: str,
) -> List[Document]:
    """
    Loads a file from disk into LangChain Document objects using
    the appropriate loader based on file extension.

    Extra metadata is injected into every Document:
      - document_id:   Unique ID for this upload
      - document_name: Original filename
      - source:        Full path to the file on disk

    Args:
        file_path:         Path to the file on disk.
        document_id:       Unique document ID.
        original_filename: Original upload filename.

    Returns:
        List[Document] — one per page (PDF) or one for entire file (TXT).

    Raises:
        DocumentProcessingException: If loader fails or format unsupported.
    """
    ext = file_path.suffix.lower()

    loader_class = LOADER_REGISTRY.get(ext)
    if loader_class is None:
        raise DocumentProcessingException(
            message=f"No loader available for '{ext}' files.",
            details={"filename": original_filename, "extension": ext},
        )

    try:
        loader = loader_class(str(file_path))
        documents: List[Document] = loader.load()
    except Exception as exc:
        raise DocumentProcessingException(
            message=f"Failed to load '{original_filename}': {exc}",
            details={"filename": original_filename, "error": str(exc)},
        )

    if not documents:
        raise DocumentProcessingException(
            message=f"No content extracted from '{original_filename}'.",
            details={"filename": original_filename},
        )

    # Enrich every Document with our custom metadata
    for i, doc in enumerate(documents):
        doc.metadata.update({
            "document_id": document_id,
            "document_name": original_filename,
            # PyPDFLoader sets "page" automatically (0-indexed),
            # convert to 1-indexed for human readability
            "page": doc.metadata.get("page", i) + 1,
        })

    logger.info(
        "Loaded '%s': %d document(s), %d total chars",
        original_filename,
        len(documents),
        sum(len(d.page_content) for d in documents),
    )

    return documents


async def load_uploaded_file(
    file: UploadFile,
    settings: Settings,
    document_id: str | None = None,
) -> tuple[str, List[Document]]:
    """
    Full upload → load pipeline. This is the main entry point.

    1. Generates a document_id (UUID) if not provided.
    2. Saves the file to disk.
    3. Loads it into LangChain Documents via the appropriate loader.

    Args:
        file:        FastAPI UploadFile.
        settings:    App settings.
        document_id: Optional — generated if not provided.

    Returns:
        Tuple of (document_id, List[Document])
    """
    if document_id is None:
        document_id = str(uuid.uuid4())

    original_filename = file.filename or "unknown"

    # Save to disk
    file_path = await save_upload_to_disk(
        file=file,
        upload_dir=settings.upload_path,
        document_id=document_id,
    )

    # Load via LangChain loader
    documents = load_document_from_path(
        file_path=file_path,
        document_id=document_id,
        original_filename=original_filename,
    )

    return document_id, documents
