# Intelligent Document Analysis System

A backend API for uploading documents and asking questions about them using RAG (Retrieval-Augmented Generation). Built with FastAPI, LangChain, and ChromaDB.

Upload a PDF, ask a question in plain English, and get a grounded answer with citations — the system only answers from what's in your documents, never from the LLM's prior knowledge.

> **Note:** This is a learning project for RAG systems. There is no frontend included. For testing, use the interactive API docs at `/docs` after starting the server.

---

## Features

- **Document Upload** — PDF, TXT, and DOCX support. Files are extracted, cleaned, chunked, embedded, and stored in a vector database.
- **Question Answering** (`POST /query`) — Ask a natural language question, get a direct answer with source citations.
- **Semantic Search** (`POST /search`) — Search for a topic and get an organized summary of what the documents say about it.
- **Multi-document** — All endpoints search across every uploaded document by default. Optionally scope to a single document.
- **Debug Logging** — Every query writes a full pipeline trace to `logs/rag_debug.log` showing exactly which chunks were retrieved, reranker scores, and the final answer.

---

## RAG Pipeline

This project implements a production-grade two-stage retrieval pipeline, going beyond the naive "embed and search" approach most tutorials show.

### Ingestion (Upload Time)

```
PDF / TXT / DOCX
      ↓
Text Extraction     (PyPDFLoader / TextLoader / Docx2txtLoader)
      ↓
Cleaning            (remove ligatures, page artifacts, unicode noise, hyphenated line breaks)
      ↓
Chunking            (RecursiveCharacterTextSplitter, size=500, overlap=100)
      ↓
Embedding           (all-MiniLM-L6-v2, 384-dimensional vectors, runs locally)
      ↓
Vector Store        (ChromaDB, persisted to disk)
```

### Query Time

```
User Question
      ↓
Query Rewriting     (LLM rewrites the question into a keyword-rich search query)
      ↓
Hybrid Retrieval    (BM25 keyword search + MMR vector search run in parallel)
      ↓
RRF Fusion          (Reciprocal Rank Fusion merges both ranked lists)
      ↓
Cross-Encoder Reranking  (ms-marco-MiniLM-L-6-v2 scores each chunk against the query)
      ↓
LLM Answer          (GPT-4o-mini answers using only the top-ranked chunks as context)
      ↓
Response with Citations
```

### Why Each Step Exists

**Query Rewriting** — Natural language questions are often poorly phrased for retrieval. "What is the relationship between X and Y?" is rewritten to "X Y connection role context" before searching, improving recall significantly.

**Hybrid Retrieval (BM25 + MMR)** — Pure vector search fails on proper nouns, acronyms, and exact terms because embeddings capture meaning, not spelling. BM25 guarantees exact keyword matches. MMR adds diversity to prevent returning the same paragraph five times. RRF combines both ranked lists.

**Cross-Encoder Reranking** — Vector similarity finds chunks that are *topically similar* to the query. The cross-encoder reads each (query, chunk) pair together and scores how well the chunk actually *answers* the question. Much more precise, used to select the final chunks sent to the LLM.

---

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI |
| RAG Framework | LangChain |
| Vector Database | ChromaDB (local, persistent) |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace (runs locally, no API needed) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers (runs locally) |
| LLM | GPT-4o-mini via OpenAI API |
| Keyword Search | BM25 via `rank_bm25` |
| PDF Parsing | PyPDF via LangChain community loaders |

---

## Project Structure

```
Backend/
├── app/
│   ├── core/
│   │   ├── config.py          # All settings, validated at startup via Pydantic
│   │   ├── exceptions.py      # Custom exception hierarchy
│   │   └── rag_logger.py      # Debug log — overwrites on every request
│   ├── rag/
│   │   ├── ingestion/
│   │   │   ├── loader.py      # Load files → LangChain Documents
│   │   │   ├── cleaner.py     # Clean raw extracted text
│   │   │   └── chunker.py     # Split into overlapping chunks
│   │   ├── retrieval/
│   │   │   ├── retriever.py   # Hybrid BM25 + MMR retriever
│   │   │   └── reranker.py    # Cross-encoder reranker
│   │   ├── prompt_builder.py  # QA, Search, and Query Rewrite prompts
│   │   └── rag_chain.py       # Full pipelines wiring everything together
│   ├── vectorstore/
│   │   ├── chroma_store.py    # ChromaDB operations
│   │   └── embedding_manager.py  # HuggingFace embedding model singleton
│   ├── repository/
│   │   └── document_repository.py  # Document metadata (JSON file store)
│   ├── services/
│   │   ├── document_service.py  # Upload pipeline orchestration
│   │   └── query_service.py     # QA and search business logic
│   ├── routes/
│   │   ├── document_routes.py   # Upload, list, detail, delete endpoints
│   │   └── query_routes.py      # Query and search endpoints
│   ├── schemas/
│   │   ├── document_schemas.py  # Pydantic models for document endpoints
│   │   └── query_schemas.py     # Pydantic models for query endpoints
│   └── main.py                  # FastAPI app, middleware, exception handlers
├── data/
│   ├── chroma_db/             # Vector store (auto-created, gitignored)
│   └── uploads/               # Uploaded files (auto-created, gitignored)
├── logs/
│   └── rag_debug.log          # Pipeline trace for last request (gitignored)
├── .env.example               # Copy to .env and fill in your keys
└── requirements.txt
```

---

## Setup

**1. Clone and create a virtual environment**

```bash
git clone <your-repo-url>
cd Backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment**

```bash
cp .env.example .env
```

Open `.env` and set your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

Everything else has sensible defaults and works out of the box.

**4. Run the server**

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

---

## API Endpoints

### Documents

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/documents/upload` | Upload and process a document |
| `GET` | `/api/v1/documents` | List all uploaded documents |
| `GET` | `/api/v1/documents/{id}` | Get document details |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document and its chunks |

### Query

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/query` | Ask a question, get a direct answer with citations |
| `POST` | `/api/v1/search` | Search a topic, get a summarized overview |

### Health

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |

---

## Example Usage

**Upload a document**

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@contract.pdf"
```

**Ask a question**

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the notice period for termination?"}'
```

Response:
```json
{
  "success": true,
  "answer": "The notice period for termination is 30 days written notice by either party, as stated in Section 4 of the contract.",
  "sources": [
    {
      "document_name": "contract.pdf",
      "page": 4,
      "text_preview": "Either party may terminate this agreement with 30 days written notice..."
    }
  ],
  "query": "What is the notice period for termination?"
}
```

**Semantic search**

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "payment terms"}'
```

---

## Configuration

All settings are in `.env`. Key options:

| Setting | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key. |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model to use. |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | HuggingFace embedding model (runs locally). |
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker (runs locally). |
| `CHUNK_SIZE` | `500` | Max characters per chunk. |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks. |
| `RETRIEVAL_K` | `15` | Candidates fetched by vector search before MMR. |
| `RETRIEVAL_TOP_K` | `8` | Chunks kept after MMR. |
| `RERANKER_TOP_N` | `5` | Chunks kept after reranking (sent to LLM). |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload file size. |

---

## Debug Logging

Every query request overwrites `logs/rag_debug.log` with a full pipeline trace:

```
STEP 1 — Hybrid retrieval: 9 chunks retrieved
  [1] doc='contract.pdf' page=4 | text: Either party may terminate...
  [2] doc='contract.pdf' page=7 | text: Termination clauses apply...
  ...

RERANKER scores:
  ✓ KEPT [1] score=8.23 | doc='contract.pdf' page=4
  ✓ KEPT [2] score=7.11 | doc='contract.pdf' page=7
  ✗ dropped [3] score=1.20 | doc='contract.pdf' page=2
  ...

STEP 2 — After reranking, kept 5 chunks
STEP 3 — Context built (1847 chars), sending to LLM...
STEP 4 — LLM answer received
ANSWER: The notice period is 30 days...
```

This makes it easy to understand exactly why the system returned a particular answer.
