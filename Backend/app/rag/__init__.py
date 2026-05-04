"""
rag -- RAG (Retrieval-Augmented Generation) pipeline.

Modules:
    ingestion/        -- load → clean → chunk
    retrieval/        -- hybrid retriever (MMR + BM25) + cross-encoder reranker
    prompt_builder.py -- QA, Search, and Query Rewrite prompt templates
    rag_chain.py      -- full pipelines wiring everything together
"""

from app.rag.rag_chain import run_qa_chain, run_search_chain

__all__ = [
    "run_qa_chain",
    "run_search_chain",
]
