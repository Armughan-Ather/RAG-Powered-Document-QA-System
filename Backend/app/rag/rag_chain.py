"""
app/rag/rag_chain.py

LCEL chains that connect retriever -> prompt -> LLM -> output.

What is LCEL?
  LangChain Expression Language -- the modern way to compose chains.
  Uses the pipe operator (|) to connect components:
      chain = retriever | prompt | llm | parser

  Each component transforms data and passes it to the next:
      retriever: query -> List[Document]
      prompt:    {context, question} -> ChatMessages
      llm:       ChatMessages -> AIMessage
      parser:    AIMessage -> str

Chains provided:
  create_qa_chain()     -- Question answering with citations.
                           Returns: {"answer": str, "sources": [...], "query": str}

  create_search_chain() -- Semantic search with summary.
                           Returns: {"summary": str, "sources": [...], "query": str}

Both chains return the source documents alongside the answer so the
API can include citations in the response.
"""

import logging
from functools import lru_cache
from typing import Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.rag.prompt_builder import (
    QA_PROMPT,
    SEARCH_PROMPT,
    QUERY_REWRITE_PROMPT,
    format_documents_as_context,
)
from app.rag.retrieval.retriever import get_retriever, get_filtered_retriever
from app.rag.retrieval.reranker import rerank_documents

logger = logging.getLogger(__name__)


# ── LLM Singleton ─────────────────────────────────────────────────────────────

@lru_cache()
def get_llm() -> ChatOpenAI:
    """
    Returns the singleton ChatOpenAI instance.

    Created once, reused for every request.
    Temperature=0.0 because document QA must be deterministic
    (same question + same docs = same answer).
    """
    settings = get_settings()

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=settings.OPENAI_TEMPERATURE,
        max_tokens=settings.OPENAI_MAX_TOKENS,
        api_key=settings.OPENAI_API_KEY,
    )

    logger.info("LLM initialized: %s (temp=%.1f)", settings.OPENAI_MODEL, settings.OPENAI_TEMPERATURE)
    return llm


# ── QA Chain ──────────────────────────────────────────────────────────────────

async def run_qa_chain(
    question: str,
    document_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Runs the full QA pipeline:
      1. Retrieve relevant chunks from Chroma.
      2. Format them as context.
      3. Inject into the QA prompt.
      4. Send to GPT-4o-mini.
      5. Return answer + source documents.

    Args:
        question:    The user's natural language question.
        document_id: Optional -- scope search to one document.
        k:           Number of chunks to retrieve (default from settings).
        search_type: "mmr" (default) or "similarity".

    Returns:
        {
            "answer": "The refund policy allows...",
            "sources": [
                {"document_name": "policy.pdf", "page": 3, "chunk_id": "..."},
                ...
            ],
            "query": "What is the refund policy?",
        }
    """
    llm = get_llm()

    # Step 1: Rewrite the question into a better search query
    rewrite_chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
    search_query = await rewrite_chain.ainvoke({"question": question})
    search_query = search_query.strip()

    logger.info("Query rewritten: '%s' → '%s'", question[:80], search_query)

    # Step 2: Get the right retriever
    if document_id:
        retriever = get_filtered_retriever(document_id=document_id)
    else:
        retriever = get_retriever()

    # Step 3: Retrieve using the rewritten query
    source_documents = await retriever.ainvoke(search_query)

    logger.info("=" * 60)
    logger.info("QA PIPELINE — question: '%s'", question)
    logger.info("Search query (rewritten): '%s'", search_query)
    logger.info("Scope: %s", f"document_id={document_id}" if document_id else "ALL documents")
    logger.info("=" * 60)
    logger.info("STEP 1 — MMR retrieved %d chunks:", len(source_documents))
    for i, doc in enumerate(source_documents):
        logger.info(
            "  [%d] doc='%s' page=%s chunk_id=%s\n      text: %s",
            i + 1,
            doc.metadata.get("document_name", "?"),
            doc.metadata.get("page", "?"),
            doc.metadata.get("chunk_id", "?"),
            doc.page_content.replace("\n", " "),
        )

    # Step 4: Rerank using the rewritten query
    source_documents = rerank_documents(search_query, source_documents)

    logger.info("STEP 2 — After reranking, kept %d chunks:", len(source_documents))
    for i, doc in enumerate(source_documents):
        logger.info(
            "  [%d] doc='%s' page=%s chunk_id=%s\n      text: %s",
            i + 1,
            doc.metadata.get("document_name", "?"),
            doc.metadata.get("page", "?"),
            doc.metadata.get("chunk_id", "?"),
            doc.page_content.replace("\n", " "),
        )

    # Step 4: Format context
    context = format_documents_as_context(source_documents)
    logger.info("STEP 3 — Context built (%d chars), sending to LLM...", len(context))

    # Step 4: Build and invoke the chain
    # Chain: prompt -> llm -> string output
    chain = QA_PROMPT | llm | StrOutputParser()
    answer = await chain.ainvoke({
        "context": context,
        "question": question,
    })

    sources = _extract_sources(source_documents)

    logger.info("STEP 4 — LLM answer received (%d chars)", len(answer))
    logger.info("ANSWER:\n%s", answer)
    logger.info("SOURCES: %s", [(s["document_name"], f"page {s['page']}") for s in sources])
    logger.info("=" * 60)

    return {
        "answer": answer,
        "sources": sources,
        "query": question,
    }


# ── Search Chain ──────────────────────────────────────────────────────────────

async def run_search_chain(
    query: str,
    document_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Runs the semantic search pipeline:
      1. Retrieve relevant chunks.
      2. Summarize them using the LLM.
      3. Return summary + source documents.

    Args:
        query:       Search query text.
        document_id: Optional -- scope to one document.
        k:           Number of chunks to retrieve.
        search_type: "mmr" or "similarity".

    Returns:
        {
            "summary": "The documents contain information about...",
            "sources": [...],
            "query": "payment terms",
        }
    """
    llm = get_llm()

    # Rewrite the query for better retrieval
    rewrite_chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
    search_query = await rewrite_chain.ainvoke({"question": query})
    search_query = search_query.strip()

    logger.info("Query rewritten: '%s' → '%s'", query[:80], search_query)

    # Step 1: Retrieve
    if document_id:
        retriever = get_filtered_retriever(document_id=document_id)
    else:
        retriever = get_retriever()

    source_documents = await retriever.ainvoke(search_query)

    logger.info("=" * 60)
    logger.info("SEARCH PIPELINE — query: '%s'", query)
    logger.info("Scope: %s", f"document_id={document_id}" if document_id else "ALL documents")
    logger.info("=" * 60)
    logger.info("STEP 1 — MMR retrieved %d chunks:", len(source_documents))
    for i, doc in enumerate(source_documents):
        logger.info(
            "  [%d] doc='%s' page=%s chunk_id=%s\n      text: %s",
            i + 1,
            doc.metadata.get("document_name", "?"),
            doc.metadata.get("page", "?"),
            doc.metadata.get("chunk_id", "?"),
            doc.page_content.replace("\n", " "),
        )

    # Rerank before summarizing
    source_documents = rerank_documents(search_query, source_documents)

    logger.info("STEP 2 — After reranking, kept %d chunks:", len(source_documents))
    for i, doc in enumerate(source_documents):
        logger.info(
            "  [%d] doc='%s' page=%s chunk_id=%s | text: %s",
            i + 1,
            doc.metadata.get("document_name", "?"),
            doc.metadata.get("page", "?"),
            doc.metadata.get("chunk_id", "?"),
            doc.page_content.replace("\n", " "),
        )

    context = format_documents_as_context(source_documents)
    logger.info("STEP 3 — Context built (%d chars), sending to LLM...", len(context))

    chain = SEARCH_PROMPT | llm | StrOutputParser()

    summary = await chain.ainvoke({
        "context": context,
        "query": query,
    })

    sources = _extract_sources(source_documents)

    logger.info("STEP 4 — LLM summary received (%d chars)", len(summary))
    logger.info("SUMMARY:\n%s", summary)
    logger.info("SOURCES: %s", [(s["document_name"], f"page {s['page']}") for s in sources])
    logger.info("=" * 60)

    return {
        "summary": summary,
        "sources": sources,
        "query": query,
    }


# ── Helper ────────────────────────────────────────────────────────────────────


def _extract_sources(documents) -> list[dict]:
    """
    Extracts citation metadata from source documents.

    Returns a list of dicts with the info needed for API response:
        [
            {
                "document_name": "invoice.pdf",
                "document_id": "abc123",
                "page": 5,
                "chunk_id": "abc123_5_2",
                "text_preview": "Payment must be received within..."
            },
            ...
        ]
    """
    sources = []
    seen_chunk_ids = set()  # deduplicate

    for doc in documents:
        chunk_id = doc.metadata.get("chunk_id", "")

        # Skip duplicates (MMR may still return near-duplicates)
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        sources.append({
            "document_name": doc.metadata.get("document_name", "Unknown"),
            "document_id": doc.metadata.get("document_id", ""),
            "page": doc.metadata.get("page"),
            "chunk_id": chunk_id,
            "text_preview": doc.page_content[:150] + "..."
                if len(doc.page_content) > 150
                else doc.page_content,
        })

    return sources
