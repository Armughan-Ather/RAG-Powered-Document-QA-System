"""
rag/ingestion -- Document ingestion pipeline.

These modules handle the ingestion side of RAG:
    load → clean → chunk

Modules:
    loader.py   -- Load files into LangChain Documents (PDF, TXT, DOCX)
    cleaner.py  -- Clean raw extracted text (noise, ligatures, artifacts)
    chunker.py  -- Split Documents into overlapping chunks
"""

from app.rag.ingestion.loader import load_uploaded_file
from app.rag.ingestion.cleaner import clean_documents
from app.rag.ingestion.chunker import split_documents

__all__ = [
    "load_uploaded_file",
    "clean_documents",
    "split_documents",
]
