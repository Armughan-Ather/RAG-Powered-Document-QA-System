"""
app/core/rag_logger.py

Single-file RAG debug logger.

Creates one file: logs/rag_debug.log
Overwritten at the start of every query request.

The duplication fix: attach the handler to ONE top-level logger ("app")
only, and let propagation do the rest naturally. Never attach to both
a parent and its children — that causes every record to be written twice.
"""

import logging
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent.parent / "logs" / "rag_debug.log"

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Noisy third-party libraries — keep them at WARNING so they don't pollute the file
_NOISY_LOGGERS = [
    "httpcore", "httpx", "sentence_transformers", "chromadb",
    "openai", "langchain", "langsmith", "urllib3", "uvicorn", "fastapi",
]

_file_handler: logging.FileHandler | None = None


def reset_rag_log() -> None:
    """
    Call once at the start of every query/search/extract request.
    Wipes the previous log and starts fresh.
    """
    global _file_handler

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Remove old handler from the single attachment point
    app_logger = logging.getLogger("app")
    if _file_handler is not None:
        app_logger.removeHandler(_file_handler)
        _file_handler.close()

    # Fresh file — 'w' wipes previous request's content
    _file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(_FORMATTER)

    # Attach to "app" only — all app.* loggers propagate up to this one.
    # Never attach to children too, that causes duplicate lines.
    app_logger.addHandler(_file_handler)
    app_logger.setLevel(logging.DEBUG)

    # Silence noisy third-party libraries
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
