"""
app/rag/prompt_builder.py

Prompt templates for the RAG pipeline.

Why separate prompts into their own file?
  - Easy to iterate on prompt engineering without touching chain logic.
  - Each template is a ChatPromptTemplate -- type-safe, validates variables.
  - Templates are constants (created once, reused for every request).

Templates:
  QA_PROMPT      -- Main question-answering prompt with citation rules.
  SEARCH_PROMPT  -- Semantic search: summarize relevant sections.
"""

from langchain_core.prompts import ChatPromptTemplate


# ── QA Prompt ─────────────────────────────────────────────────────────────────
# Used by: POST /query
# Input variables: {context}, {question}

QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an intelligent document analysis assistant. "
        "Your job is to answer questions accurately based ONLY on the "
        "provided document context.\n\n"
        "RULES:\n"
        "1. Use ONLY the information in the context below. "
        "Do NOT use prior knowledge.\n"
        "2. If the answer is not found in the context, say: "
        "\"I could not find this information in the provided documents.\"\n"
        "3. Always cite your sources at the end of your answer using this format:\n"
        "   Sources: [Document: <name>, Page: <number>]\n"
        "4. Be concise and direct. Do not repeat the question.\n"
        "5. If multiple documents contain relevant information, "
        "synthesize them into a single coherent answer.\n"
        "6. Preserve exact numbers, dates, and proper nouns from the context.\n"
        "7. Answer the INTENT of the question, not just the literal wording. "
        "If someone asks about a 'relationship' between two things, look for "
        "any connection — direct or indirect — present in the context."
    ),
    (
        "human",
        "Context:\n"
        "---\n"
        "{context}\n"
        "---\n\n"
        "Question: {question}"
    ),
])


# ── Search Prompt ─────────────────────────────────────────────────────────────
# Used by: POST /search
# Input variables: {context}, {query}
# Purpose: Summarize the most relevant sections found across documents.

SEARCH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a document search assistant. "
        "Given a search query and relevant document sections, "
        "provide a clear summary of the information found.\n\n"
        "RULES:\n"
        "1. Summarize the relevant information from the sections below.\n"
        "2. Organize by topic if multiple topics are covered.\n"
        "3. Cite each piece of information with "
        "[Document: <name>, Page: <number>].\n"
        "4. If the sections don't contain relevant information, "
        "say: \"No relevant information found for this query.\""
    ),
    (
        "human",
        "Document sections:\n"
        "---\n"
        "{context}\n"
        "---\n\n"
        "Search query: {query}"
    ),
])


# ── Helper: Format Documents into Context String ──────────────────────────────

def format_documents_as_context(documents) -> str:
    """
    Converts a list of LangChain Document objects into a single
    formatted string for injection into prompts.

    Each chunk is labeled with its source info so the LLM can cite it.

    Example output:
        [Document: invoice.pdf, Page: 1]
        Invoice Number: INV-2026-001. Date: April 28, 2026...

        [Document: invoice.pdf, Page: 2]
        Payment Terms: Payment must be received within 30 days...

    Args:
        documents: List[Document] from the retriever.

    Returns:
        Formatted context string.
    """
    if not documents:
        return "No relevant documents found."

    parts = []
    for doc in documents:
        doc_name = doc.metadata.get("document_name", "Unknown")
        page = doc.metadata.get("page", "?")
        header = f"[Document: {doc_name}, Page: {page}]"
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(parts)


# ── Query Rewrite Prompt ──────────────────────────────────────────────────────
# Used by: run_qa_chain() and run_search_chain() before retrieval
# Purpose: Rewrite the user's natural language question into a keyword-rich
#          search query that retrieves better chunks from the vector store.
#
# Why?
#   Natural language questions are often ambiguous or poorly phrased for
#   retrieval. "What is the relationship between X and Y?" implies a direct
#   connection that may not exist in the document. Rewriting it as
#   "X Y connection context" retrieves the relevant chunks regardless.
#
#   The rewriter is document-agnostic — it works for contracts,
#   research papers, medical reports, or any other document type.

QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a search query optimizer. "
        "Your job is to rewrite a user's question into a better search query "
        "for retrieving relevant passages from documents.\n\n"
        "RULES:\n"
        "1. Output ONLY the rewritten query — no explanation, no punctuation at end.\n"
        "2. Make it keyword-rich and specific.\n"
        "3. If the question asks about a 'relationship' between two things, "
        "rewrite it to search for both things together with relevant context words.\n"
        "4. If the question is vague or uses pronouns, make it more explicit.\n"
        "5. Keep it under 20 words.\n\n"
        "Examples:\n"
        "  Q: What is the relationship between the author and the company?\n"
        "  A: author company role position employment connection\n\n"
        "  Q: What did they agree on?\n"
        "  A: agreement terms conditions obligations parties\n\n"
        "  Q: Tell me about the main findings\n"
        "  A: main findings results conclusions key outcomes\n\n"
        "  Q: What happened in section 3?\n"
        "  A: section 3 content details information"
    ),
    (
        "human",
        "Rewrite this question as a search query: {question}"
    ),
])
