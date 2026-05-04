"""
vectorstore -- Chroma vector store and embedding model.

Modules:
    embedding_manager.py -- Singleton HuggingFace embedding model
    chroma_store.py      -- Chroma operations (add, search, delete, stats)
"""

from app.vectorstore.embedding_manager import get_embedding_model
from app.vectorstore.chroma_store import (
    get_vector_store,
    add_documents,
    similarity_search,
    mmr_search,
    search_with_filters,
    delete_document,
    get_document_ids,
    get_collection_stats,
)

__all__ = [
    "get_embedding_model",
    "get_vector_store",
    "add_documents",
    "similarity_search",
    "mmr_search",
    "search_with_filters",
    "delete_document",
    "get_document_ids",
    "get_collection_stats",
]
