"""
app/vectorstore/chroma_store.py

Wraps LangChain's Chroma vector store with all operations needed
by the application.

Architecture:
  - Uses langchain_chroma.Chroma (official LangChain integration).
  - Chroma instance is a lazy singleton (created on first use).
  - Persistence: data stored on disk at CHROMA_PERSIST_DIR.
  - Embedding: uses the singleton from embedding_manager.py.

Operations:
  add_documents()       -- Embed + store List[Document] into Chroma.
  similarity_search()   -- Top-k most similar chunks to a query.
  mmr_search()          -- MMR: balances relevance + diversity.
  search_with_filters() -- Search scoped by metadata (e.g., document_id).
  delete_document()     -- Remove all chunks for a given document.
  get_document_ids()    -- List all unique document IDs in the store.
  get_collection_stats()-- Count of documents, collection name, etc.

All search methods return List[Document] -- the same format that
went in, so Phase 4 (RAG) can consume them directly.
"""

import logging
from functools import lru_cache
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.exceptions import VectorStoreException
from app.vectorstore.embedding_manager import get_embedding_model

logger = logging.getLogger(__name__)


# ── Singleton Vector Store ────────────────────────────────────────────────────

@lru_cache()
def get_vector_store() -> Chroma:
    """
    Returns the singleton Chroma vector store instance.

    Created on first call, reused forever after.
    Reads config from settings (collection name, persist directory).
    Uses the embedding model from embedding_manager.py.

    Returns:
        Chroma instance connected to the persistent store.
    """
    settings = get_settings()
    embeddings = get_embedding_model()

    logger.info(
        "Initializing Chroma vector store: collection='%s', dir='%s'",
        settings.CHROMA_COLLECTION_NAME,
        settings.CHROMA_PERSIST_DIR,
    )

    try:
        vector_store = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )
    except Exception as exc:
        raise VectorStoreException(
            message=f"Failed to initialize Chroma: {exc}",
            details={"error": str(exc)},
        )

    logger.info("Chroma vector store ready.")
    return vector_store


# ── Add Documents ─────────────────────────────────────────────────────────────

def add_documents(documents: List[Document]) -> List[str]:
    """
    Embeds and stores a list of Documents into Chroma.

    Chroma handles embedding automatically using the embedding_function
    we passed during initialization.

    Args:
        documents: List[Document] from Phase 2 (chunked, cleaned).

    Returns:
        List of Chroma IDs assigned to each document.

    Raises:
        VectorStoreException: If the add operation fails.
    """
    if not documents:
        logger.warning("add_documents called with empty list -- skipping.")
        return []

    store = get_vector_store()

    try:
        ids = store.add_documents(documents)
    except Exception as exc:
        raise VectorStoreException(
            message=f"Failed to add documents to Chroma: {exc}",
            details={"count": len(documents), "error": str(exc)},
        )

    doc_name = documents[0].metadata.get("document_name", "unknown")
    logger.info(
        "Added %d chunks to Chroma from '%s'.",
        len(ids), doc_name,
    )

    return ids


# ── Similarity Search ─────────────────────────────────────────────────────────

def similarity_search(
    query: str,
    k: int = 5,
    filter_dict: Optional[dict] = None,
) -> List[Document]:
    """
    Finds the top-k chunks most semantically similar to the query.

    How it works:
      1. Query text is embedded into a vector (same model as storage).
      2. Chroma finds the k nearest vectors by cosine similarity.
      3. Returns the corresponding Documents with metadata.

    Args:
        query:       Natural language question or search text.
        k:           Number of results to return (default: 5).
        filter_dict: Optional metadata filter (e.g., {"document_id": "abc123"}).

    Returns:
        List[Document] -- the k most relevant chunks.
    """
    store = get_vector_store()

    try:
        results = store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict,
        )
    except Exception as exc:
        raise VectorStoreException(
            message=f"Similarity search failed: {exc}",
            details={"query": query[:100], "error": str(exc)},
        )

    logger.info(
        "Similarity search: '%s' -- returned %d results.",
        query[:80], len(results),
    )

    return results


def similarity_search_with_scores(
    query: str,
    k: int = 5,
    filter_dict: Optional[dict] = None,
) -> List[tuple[Document, float]]:
    """
    Same as similarity_search but also returns the similarity score
    for each result. Useful for debugging retrieval quality.

    Returns:
        List of (Document, score) tuples. Lower score = more similar.
    """
    store = get_vector_store()

    try:
        results = store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict,
        )
    except Exception as exc:
        raise VectorStoreException(
            message=f"Similarity search with scores failed: {exc}",
            details={"query": query[:100], "error": str(exc)},
        )

    logger.info(
        "Similarity search (scored): '%s' -- %d results, top score: %.4f",
        query[:80],
        len(results),
        results[0][1] if results else 0.0,
    )

    return results


# ── MMR Search ────────────────────────────────────────────────────────────────

def mmr_search(
    query: str,
    k: int = 5,
    fetch_k: int = 15,
    lambda_mult: float = 0.5,
    filter_dict: Optional[dict] = None,
) -> List[Document]:
    """
    Max Marginal Relevance search -- balances relevance and diversity.

    Why use MMR instead of plain similarity?
      Without MMR, the top 5 results might all be from the SAME paragraph
      (slightly different chunks, same content). MMR penalizes redundancy.

    How it works:
      1. Fetch fetch_k candidates by similarity (e.g., 15).
      2. From those, select k results (e.g., 5) that are:
         - Relevant to the query (similarity)
         - Diverse from each other (not redundant)
      3. lambda_mult controls the balance:
         - 1.0 = pure relevance (same as similarity_search)
         - 0.0 = pure diversity (maximum variety)
         - 0.5 = balanced (recommended default)

    Args:
        query:       Natural language question.
        k:           Final number of results to return.
        fetch_k:     Candidates to fetch before MMR filtering.
        lambda_mult: Balance between relevance (1) and diversity (0).
        filter_dict: Optional metadata filter.

    Returns:
        List[Document] -- diverse, relevant chunks.
    """
    store = get_vector_store()

    try:
        results = store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter_dict,
        )
    except Exception as exc:
        raise VectorStoreException(
            message=f"MMR search failed: {exc}",
            details={"query": query[:100], "error": str(exc)},
        )

    logger.info(
        "MMR search: '%s' -- %d results (fetch_k=%d, lambda=%.2f).",
        query[:80], len(results), fetch_k, lambda_mult,
    )

    return results


# ── Filtered Search ───────────────────────────────────────────────────────────

def search_with_filters(
    query: str,
    document_id: Optional[str] = None,
    document_name: Optional[str] = None,
    page: Optional[int] = None,
    k: int = 5,
    use_mmr: bool = True,
) -> List[Document]:
    """
    Convenience method -- builds a metadata filter dict and runs search.

    Examples:
      # Only search within a specific document:
      search_with_filters("refund policy", document_id="abc123")

      # Only search a specific page:
      search_with_filters("total amount", document_id="abc123", page=5)

    Args:
        query:         Natural language question.
        document_id:   Filter to a specific document.
        document_name: Filter by original filename.
        page:          Filter to a specific page number.
        k:             Number of results.
        use_mmr:       Use MMR (True) or plain similarity (False).

    Returns:
        List[Document] -- filtered results.
    """
    settings = get_settings()

    # Build the filter dict (Chroma uses $and for multiple conditions)
    conditions = []
    if document_id:
        conditions.append({"document_id": document_id})
    if document_name:
        conditions.append({"document_name": document_name})
    if page is not None:
        conditions.append({"page": page})

    filter_dict = None
    if len(conditions) == 1:
        filter_dict = conditions[0]
    elif len(conditions) > 1:
        filter_dict = {"$and": conditions}

    if use_mmr:
        return mmr_search(
            query=query,
            k=k,
            fetch_k=settings.RETRIEVAL_K,
            lambda_mult=settings.MMR_LAMBDA,
            filter_dict=filter_dict,
        )
    else:
        return similarity_search(
            query=query,
            k=k,
            filter_dict=filter_dict,
        )


# ── Delete ────────────────────────────────────────────────────────────────────

def delete_document(document_id: str) -> bool:
    """
    Removes ALL chunks belonging to a document from Chroma.

    Args:
        document_id: The document UUID to delete.

    Returns:
        True if deletion was successful.

    Raises:
        VectorStoreException: If the deletion fails.
    """
    store = get_vector_store()

    try:
        # Get the underlying Chroma collection
        collection = store._collection

        # Find all chunk IDs with this document_id
        results = collection.get(
            where={"document_id": document_id},
            include=[],  # we only need the IDs
        )

        ids_to_delete = results.get("ids", [])

        if not ids_to_delete:
            logger.warning(
                "No chunks found for document_id='%s' -- nothing to delete.",
                document_id,
            )
            return False

        collection.delete(ids=ids_to_delete)

        logger.info(
            "Deleted %d chunks for document_id='%s'.",
            len(ids_to_delete), document_id,
        )
        return True

    except Exception as exc:
        raise VectorStoreException(
            message=f"Failed to delete document '{document_id}': {exc}",
            details={"document_id": document_id, "error": str(exc)},
        )


# ── Utility Methods ──────────────────────────────────────────────────────────

def get_document_ids() -> List[str]:
    """
    Returns a list of all unique document IDs stored in Chroma.
    Used by the GET /documents endpoint.
    """
    store = get_vector_store()

    try:
        collection = store._collection
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        # Extract unique document IDs
        doc_ids = list({
            m.get("document_id")
            for m in metadatas
            if m.get("document_id")
        })

        return sorted(doc_ids)

    except Exception as exc:
        raise VectorStoreException(
            message=f"Failed to list document IDs: {exc}",
            details={"error": str(exc)},
        )


def get_collection_stats() -> dict:
    """
    Returns statistics about the Chroma collection.
    Useful for health checks and admin dashboards.

    Returns:
        {
            "collection_name": "documents",
            "total_chunks": 1234,
            "unique_documents": 15,
        }
    """
    store = get_vector_store()
    settings = get_settings()

    try:
        collection = store._collection
        count = collection.count()

        return {
            "collection_name": settings.CHROMA_COLLECTION_NAME,
            "total_chunks": count,
            "unique_documents": len(get_document_ids()),
        }

    except Exception as exc:
        raise VectorStoreException(
            message=f"Failed to get collection stats: {exc}",
            details={"error": str(exc)},
        )
