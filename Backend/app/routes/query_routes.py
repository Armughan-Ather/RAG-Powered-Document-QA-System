"""
app/routes/query_routes.py

Query endpoints:
  POST /query   -- Ask a question (RAG-based QA with citations).
  POST /search  -- Semantic search with summary.
"""

import logging

from fastapi import APIRouter

from app.core.rag_logger import reset_rag_log
from app.schemas.query_schemas import (
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)
from app.services import query_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse, summary="Ask a question")
async def query_documents(request: QueryRequest):
    reset_rag_log()
    result = await query_service.ask_question(
        question=request.question,
        document_id=request.document_id,
    )
    return QueryResponse(success=True, **result)


@router.post("/search", response_model=SearchResponse, summary="Semantic search")
async def search_documents(request: SearchRequest):
    reset_rag_log()
    result = await query_service.semantic_search(
        query=request.query,
        document_id=request.document_id,
    )
    return SearchResponse(success=True, **result)
