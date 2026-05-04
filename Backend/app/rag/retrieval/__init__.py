"""
rag/retrieval -- Retrieval pipeline.

These modules handle the query side of RAG:
    retrieve → rerank

Modules:
    retriever.py -- Creates LangChain Retriever from Chroma vector store
    reranker.py  -- Cross-encoder reranker for precision improvement
"""

from app.rag.retrieval.retriever import get_retriever, get_filtered_retriever
from app.rag.retrieval.reranker import rerank_documents

__all__ = [
    "get_retriever",
    "get_filtered_retriever",
    "rerank_documents",
]
