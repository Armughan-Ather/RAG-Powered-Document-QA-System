"""
app/rag/ingestion/chunker.py

Splits LangChain Document objects into overlapping chunks using
RecursiveCharacterTextSplitter.
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings

logger = logging.getLogger(__name__)


def create_text_splitter(settings: Settings) -> RecursiveCharacterTextSplitter:
    """
    Creates a text splitter configured from app settings.

    Uses character count (len) as the length function.
    For token-based splitting, swap len for a tokenizer function.

    Args:
        settings: App settings (CHUNK_SIZE, CHUNK_OVERLAP).

    Returns:
        Configured RecursiveCharacterTextSplitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )


def split_documents(
    documents: List[Document],
    settings: Settings,
    extra_metadata: Optional[dict] = None,
) -> List[Document]:
    """
    Splits a list of LangChain Documents into smaller chunks.

    This is the main entry point -- call this from services.

    What it does:
      1. Creates a RecursiveCharacterTextSplitter from settings.
      2. Calls splitter.split_documents(documents).
         - This preserves ALL existing metadata from the parent Document.
         - Each chunk gets its own page_content (the chunk text).
      3. Adds chunk_index and chunk_id to each chunk's metadata.

    Args:
        documents:      List[Document] from document_loader.py.
        settings:       App settings (chunk size/overlap).
        extra_metadata: Optional dict to attach to every chunk
                        (e.g., {"category": "invoice"}).

    Returns:
        List[Document] -- each chunk is a Document with:
          .page_content = chunk text
          .metadata = {
              ...parent metadata (document_id, page, source, etc.),
              "chunk_index": 0,
              "chunk_id": "abc123_5_0",
              "char_count": 487,
              ...extra_metadata
          }
    """
    splitter = create_text_splitter(settings)

    # split_documents preserves metadata from parent Documents
    chunks: List[Document] = splitter.split_documents(documents)

    # Enrich each chunk with index and unique ID
    for i, chunk in enumerate(chunks):
        doc_id = chunk.metadata.get("document_id", "unknown")
        page = chunk.metadata.get("page", 0)

        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_id"] = f"{doc_id}_{page}_{i}"
        chunk.metadata["char_count"] = len(chunk.page_content)

        # Merge any extra metadata
        if extra_metadata:
            chunk.metadata.update(extra_metadata)

    doc_name = documents[0].metadata.get("document_name", "unknown") if documents else "unknown"
    logger.info(
        "Chunking complete: '%s' -- %d documents -> %d chunks "
        "(size=%d, overlap=%d)",
        doc_name,
        len(documents),
        len(chunks),
        settings.CHUNK_SIZE,
        settings.CHUNK_OVERLAP,
    )

    return chunks
