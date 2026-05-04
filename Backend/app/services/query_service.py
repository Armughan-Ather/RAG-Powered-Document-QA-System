"""
app/services/query_service.py

Business logic for question answering and semantic search.
Delegates to the RAG chain (Phase 4) and formats results for the API.

Routes call these functions. These functions call rag_chain.
"""

import logging
from typing import Any, Optional

from app.core.exceptions import RAGException
from app.rag.rag_chain import run_qa_chain, run_search_chain
from app.repository import document_repository

logger = logging.getLogger(__name__)


async def ask_question(
    question: str,
    document_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Answers a natural language question using RAG.

    Steps:
      1. Optionally verify the document exists (if scoped).
      2. Run the QA chain (retrieve chunks -> prompt -> LLM -> answer).
      3. Return answer + source citations.

    Args:
        question:    The user's question.
        document_id: Optional -- scope to one document.
        k:           Number of chunks to retrieve.

    Returns:
        {
            "answer": "The refund policy allows...",
            "sources": [...],
            "query": "What is the refund policy?",
        }

    Raises:
        RAGException: If the chain fails.
    """
    # Validate document exists if scoped
    if document_id:
        document_repository.get_document(document_id)  # raises if not found

    try:
        result = await run_qa_chain(
            question=question,
            document_id=document_id,
        )
    except Exception as exc:
        raise RAGException(
            message=f"Failed to answer question: {exc}",
            details={"question": question[:200], "error": str(exc)},
        )

    logger.info(
        "Question answered: '%s' -> %d sources",
        question[:80],
        len(result.get("sources", [])),
    )

    return result


async def semantic_search(
    query: str,
    document_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Performs semantic search and summarizes the results.
    """
    if document_id:
        document_repository.get_document(document_id)

    try:
        result = await run_search_chain(
            query=query,
            document_id=document_id,
        )
    except Exception as exc:
        raise RAGException(
            message=f"Search failed: {exc}",
            details={"query": query[:200], "error": str(exc)},
        )

    logger.info(
        "Search completed: '%s' -> %d sources",
        query[:80],
        len(result.get("sources", [])),
    )

    return result
