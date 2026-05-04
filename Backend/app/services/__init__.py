"""
services -- Business logic layer.

Services sit between routes and the lower-level modules.
Routes call services. Services call utils, vectorstore, rag, and repository.

Modules:
    document_service.py -- Upload pipeline orchestration + document CRUD
    query_service.py    -- QA and search delegation to RAG chains
"""
