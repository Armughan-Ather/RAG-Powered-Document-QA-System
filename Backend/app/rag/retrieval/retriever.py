"""
app/rag/retrieval/retriever.py

Hybrid retriever: combines vector search (MMR) + BM25 keyword search.

Why hybrid?
  Pure vector search is semantic — it finds chunks with similar *meaning*.
  This works well for conceptual questions but fails for proper nouns,
  acronyms, and exact terms.

  BM25 is a classic keyword ranking algorithm — it finds exact word matches
  regardless of meaning, guaranteeing exact-term recall.

  Hybrid = vector score + BM25 score combined via RRF (Reciprocal Rank Fusion).
  RRF merges two ranked lists into one without needing to normalize scores.

  Result: proper nouns, acronyms, and exact terms are always found,
  while semantic understanding is preserved for conceptual queries.

BM25 limitation:
  BM25 needs all documents in memory as a list. We rebuild it from Chroma
  on every retriever creation. For millions of chunks, you'd use a dedicated
  search engine like Elasticsearch.
"""

import logging
from typing import Optional

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.config import Settings, get_settings
from app.vectorstore.chroma_store import get_vector_store

logger = logging.getLogger(__name__)


def _get_all_documents() -> list[Document]:
    """
    Fetches all stored chunks from Chroma as LangChain Documents.
    Used to build the BM25 index over the full corpus.
    """
    store = get_vector_store()
    collection = store._collection
    result = collection.get(include=["documents", "metadatas"])

    docs = []
    for text, metadata in zip(result["documents"], result["metadatas"]):
        docs.append(Document(page_content=text, metadata=metadata or {}))

    return docs


def get_retriever(
    k: Optional[int] = None,
    filter_dict: Optional[dict] = None,
    settings: Optional[Settings] = None,
) -> EnsembleRetriever:
    """
    Creates a hybrid retriever combining MMR vector search and BM25.

    The two retrievers are combined with equal weight (0.5 / 0.5) using
    Reciprocal Rank Fusion — each retriever contributes its ranked list
    and RRF merges them into a single ranked result.

    Args:
        k:           Number of results each retriever returns before merging.
                     Default: settings.RETRIEVAL_TOP_K.
        filter_dict: Metadata filter for the vector retriever only.
                     BM25 does its own filtering by document_id if provided.
        settings:    App settings. Auto-loaded if not provided.

    Returns:
        EnsembleRetriever (vector + BM25 combined).
    """
    if settings is None:
        settings = get_settings()

    k = k or settings.RETRIEVAL_TOP_K
    fetch_k = settings.RETRIEVAL_K
    lambda_mult = settings.MMR_LAMBDA

    store = get_vector_store()

    # ── Vector retriever (MMR) ────────────────────────────────────────────────
    search_kwargs: dict = {
        "k": k,
        "fetch_k": fetch_k,
        "lambda_mult": lambda_mult,
    }
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    vector_retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs,
    )

    # ── BM25 retriever (keyword) ──────────────────────────────────────────────
    all_docs = _get_all_documents()

    # If filtering by document_id, restrict BM25 to that document's chunks too
    if filter_dict and "document_id" in filter_dict:
        doc_id = filter_dict["document_id"]
        all_docs = [d for d in all_docs if d.metadata.get("document_id") == doc_id]

    if not all_docs:
        # No documents yet — return vector-only retriever
        logger.warning("No documents in store, falling back to vector-only retriever.")
        return vector_retriever  # type: ignore[return-value]

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = k

    # ── Ensemble (RRF fusion) ─────────────────────────────────────────────────
    # weights=[0.5, 0.5] means equal contribution from both retrievers.
    # Increase vector weight (e.g. [0.3, 0.7]) if semantic queries dominate.
    # Increase BM25 weight (e.g. [0.7, 0.3]) if exact-match queries dominate.
    hybrid = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    logger.debug(
        "Hybrid retriever created: k=%d, filter=%s, corpus=%d docs",
        k, filter_dict, len(all_docs),
    )

    return hybrid


def get_filtered_retriever(
    document_id: Optional[str] = None,
    document_name: Optional[str] = None,
    search_type: str = "mmr",   # kept for API compatibility, hybrid always used
    k: Optional[int] = None,
) -> EnsembleRetriever:
    """
    Creates a hybrid retriever scoped to a specific document.

    Args:
        document_id:   Only search within this document.
        document_name: Only search docs with this filename.
        k:             Number of results.

    Returns:
        EnsembleRetriever with document filter applied.
    """
    conditions = []
    if document_id:
        conditions.append({"document_id": document_id})
    if document_name:
        conditions.append({"document_name": document_name})

    filter_dict = None
    if len(conditions) == 1:
        filter_dict = conditions[0]
    elif len(conditions) > 1:
        filter_dict = {"$and": conditions}

    return get_retriever(k=k, filter_dict=filter_dict)
