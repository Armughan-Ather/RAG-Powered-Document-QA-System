"""
app/utils/file_validator.py

Validates uploaded files BEFORE any processing begins.
This is the first gate — reject bad files early, with clear error messages.

Checks:
  1. File extension is in ALLOWED_EXTENSIONS (.pdf, .txt, .docx)
  2. File size is below MAX_FILE_SIZE_MB

Why a separate module?
  - Routes stay thin (just call validate_upload()).
  - Easy to add new checks later (magic bytes, virus scan, etc.)
  - Easy to unit test in isolation.
"""

import logging
from pathlib import Path

from fastapi import UploadFile

from app.core.config import Settings
from app.core.exceptions import FileTooLargeException, InvalidFileTypeException

logger = logging.getLogger(__name__)


def validate_file_extension(filename: str, settings: Settings) -> str:
    """
    Checks that the file extension is in the allowed list.

    Args:
        filename: Original filename from the upload (e.g. "invoice.pdf").
        settings: App settings (carries ALLOWED_EXTENSIONS).

    Returns:
        The normalised extension (e.g. ".pdf").

    Raises:
        InvalidFileTypeException: If the extension is not allowed.
    """
    ext = Path(filename).suffix.lower()

    if not ext:
        raise InvalidFileTypeException(
            message="File has no extension. Allowed types: "
                    f"{settings.ALLOWED_EXTENSIONS}",
            details={"filename": filename},
        )

    if ext not in settings.ALLOWED_EXTENSIONS:
        raise InvalidFileTypeException(
            message=f"File type '{ext}' is not supported. "
                    f"Allowed: {settings.ALLOWED_EXTENSIONS}",
            details={"filename": filename, "extension": ext},
        )

    logger.debug("Extension check passed: %s", ext)
    return ext


async def validate_file_size(file: UploadFile, settings: Settings) -> int:
    """
    Reads the full file content to measure size, then rewinds the cursor.

    We MUST read to measure because UploadFile.size is unreliable
    (some clients don't send Content-Length).

    Args:
        file: The FastAPI UploadFile object.
        settings: App settings (carries MAX_FILE_SIZE_MB).

    Returns:
        File size in bytes.

    Raises:
        FileTooLargeException: If the file exceeds the limit.
    """
    # Read all content to measure
    content = await file.read()
    size_bytes = len(content)

    # Rewind so downstream code can read the file again
    await file.seek(0)

    if size_bytes > settings.max_file_size_bytes:
        size_mb = round(size_bytes / (1024 * 1024), 2)
        raise FileTooLargeException(
            message=f"File is {size_mb}MB — exceeds the "
                    f"{settings.MAX_FILE_SIZE_MB}MB limit.",
            details={
                "filename": file.filename,
                "size_mb": size_mb,
                "limit_mb": settings.MAX_FILE_SIZE_MB,
            },
        )

    logger.debug("Size check passed: %d bytes", size_bytes)
    return size_bytes


async def validate_upload(file: UploadFile, settings: Settings) -> dict:
    """
    Full upload validation — call this from routes.

    Returns a dict with validated metadata:
        {
            "filename": "invoice.pdf",
            "extension": ".pdf",
            "size_bytes": 102400,
        }
    """
    filename = file.filename or "unknown"

    ext = validate_file_extension(filename, settings)
    size_bytes = await validate_file_size(file, settings)

    logger.info(
        "Upload validated: %s (%s, %d bytes)",
        filename, ext, size_bytes,
    )

    return {
        "filename": filename,
        "extension": ext,
        "size_bytes": size_bytes,
    }
