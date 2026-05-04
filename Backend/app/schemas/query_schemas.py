"""
app/schemas/query_schemas.py

Pydantic request/response models for query and search endpoints.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    document_id: Optional[str] = Field(default=None)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    document_id: Optional[str] = Field(default=None)


class SourceInfo(BaseModel):
    document_name: str
    document_id: str
    page: Optional[int] = None
    chunk_id: str = ""
    text_preview: str = ""


class QueryResponse(BaseModel):
    success: bool = True
    answer: str
    sources: List[SourceInfo]
    query: str


class SearchResponse(BaseModel):
    success: bool = True
    summary: str
    sources: List[SourceInfo]
    query: str
