"""
app/rag/retrieval/reranker.py

Cross-encoder reranker for improving retrieval precision.

Why reranking?
  Vector search (MMR) is fast but approximate. It finds chunks that are
  *semantically similar* to the query, but similarity is not the same as
  relevance. A chunk can be similar in topic but not actually answer the
  question.

  A cross-encoder reranker solves this. It takes each (query, chunk) pair
  and scores them TOGETHER — it reads both at the same time and decides
  how well the chunk answers the query. This is much more accurate than
  comparing vectors independently.

Two-stage retrieval (the standard RAG pattern):
  Stage 1 — Vector search (fast):
      Fetch 10-15 candidates from Chroma using MMR.
      Fast because it's just vector math.

  Stage 2 — Reranking (precise):
      Score all 10-15 candidates with the cross-encoder.
      Keep only the top 3-5 by rerank score.
      Slower but much more accurate.

  The LLM only sees the top 3-5 reranked chunks — not all 10-15.
  This means less noise in the context and better answers.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Runs locally, no API key needed
  - Small and fast (~80MB)
  - Trained on MS MARCO passage ranking dataset
  - Returns a relevance score (higher = more relevant)
"""

import logging
from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache()
def _get_cross_encoder() -> CrossEncoder:
    """
    Returns the singleton CrossEncoder model.

    Loaded once on first use, reused for every request.
    The model takes (query, passage) pairs and scores their relevance.
    """
    settings = get_settings()
    model_name = settings.RERANKER_MODEL_NAME

    logger.info("Loading reranker model: '%s' ...", model_name)
    model = CrossEncoder(model_name)
    logger.info("Reranker model loaded.")
    return model


def rerank_documents(
    query: str,
    documents: List[Document],
) -> List[Document]:
    """
    Reranks a list of retrieved documents by relevance to the query.
    Keeps top_n chunks as configured in settings (RERANKER_TOP_N).
    """
    if not documents:
        return []

    settings = get_settings()
    top_n = settings.RERANKER_TOP_N

    # If fewer docs than top_n, just return them all
    if len(documents) <= top_n:
        return documents

    model = _get_cross_encoder()

    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    scored = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True,
    )

    logger.info("RERANKER scores for all %d candidates:", len(documents))
    for i, (score, doc) in enumerate(scored):
        marker = "  ✓ KEPT" if i < top_n else "  ✗ dropped"
        logger.info(
            "%s [%d] score=%.4f | doc='%s' page=%s | text: %s",
            marker,
            i + 1,
            score,
            doc.metadata.get("document_name", "?"),
            doc.metadata.get("page", "?"),
            doc.page_content.replace("\n", " "),
        )

    top_docs = [doc for _, doc in scored[:top_n]]
    return top_docs
