"""
app/rag/ingestion/cleaner.py

Cleans raw extracted text from LangChain Document objects before chunking.
"""

import logging
import re
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── Individual Cleaning Steps ─────────────────────────────────────────────────

def remove_unicode_noise(text: str) -> str:
    """
    Strips invisible/control characters that break embeddings.
    Keeps standard printable chars, newlines, and tabs.
    """
    # Remove zero-width chars, BOM, soft hyphens, etc.
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", text)
    # Replace non-breaking spaces with regular spaces
    text = text.replace("\u00a0", " ")
    return text


def fix_ligatures(text: str) -> str:
    """
    Replaces common PDF ligature characters with their ASCII equivalents.
    PDF extractors sometimes emit these as single Unicode chars.
    """
    ligatures = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb00": "ff",
    }
    for lig, replacement in ligatures.items():
        text = text.replace(lig, replacement)
    return text


def fix_hyphenated_line_breaks(text: str) -> str:
    """
    Rejoins words split across lines by hyphens.
    "docu-\\nment" -> "document"

    Only joins when the next line starts with a lowercase letter
    (avoids breaking "end-of-\\nChapter" type splits).
    """
    return re.sub(r"(\w)-\s*\n\s*([a-z])", r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """
    - Collapses multiple spaces/tabs into one space.
    - Collapses 3+ newlines into 2 (keeps paragraph breaks).
    - Strips leading/trailing whitespace from each line.
    """
    # Collapse multiple spaces/tabs (but not newlines) into one space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Strip each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Collapse 3+ newlines into 2 (preserve paragraph structure)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_page_artifacts(text: str) -> str:
    """
    Removes common header/footer patterns from PDF pages.
    Patterns:
      - "Page 1 of 12", "Page 1", "- 1 -"
      - "CONFIDENTIAL", "DRAFT"
      - Standalone numbers (page numbers at top/bottom)
    """
    patterns = [
        r"(?i)^\s*page\s+\d+\s*(of\s+\d+)?\s*$",   # "Page 1 of 12"
        r"^\s*-\s*\d+\s*-\s*$",                       # "- 1 -"
        r"(?i)^\s*(confidential|draft)\s*$",           # "CONFIDENTIAL"
        r"^\s*\d{1,3}\s*$",                            # Standalone page numbers
    ]
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if any(re.match(p, line) for p in patterns):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def remove_excessive_bullets(text: str) -> str:
    """
    Normalises bullet-point markers to a consistent dash "- ".
    Common PDF artifacts: bullet chars.
    """
    text = re.sub(
        r"^\s*[\u2022\u2023\u25e6\u2043\u2219]\s+",
        "- ",
        text,
        flags=re.MULTILINE,
    )
    return text


# ── Main Cleaning Pipeline ────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Applies all cleaning steps in sequence to a single text string.
    Order matters -- ligatures before whitespace, hyphens before whitespace.

    Args:
        text: Raw extracted text from a Document's page_content.

    Returns:
        Cleaned text ready for chunking.
    """
    if not text or not text.strip():
        return ""

    text = remove_unicode_noise(text)
    text = fix_ligatures(text)
    text = fix_hyphenated_line_breaks(text)
    text = remove_page_artifacts(text)
    text = remove_excessive_bullets(text)
    text = normalize_whitespace(text)

    return text


def clean_documents(documents: List[Document]) -> List[Document]:
    """
    Cleans the page_content of every Document in the list.

    Works with LangChain Document objects -- modifies .page_content in place.
    Empty documents (after cleaning) are filtered out.

    Args:
        documents: List[Document] from the document loader.

    Returns:
        List[Document] with cleaned page_content.
        Documents that are empty after cleaning are removed.
    """
    cleaned: List[Document] = []

    for doc in documents:
        cleaned_text = clean_text(doc.page_content)

        if not cleaned_text:
            page = doc.metadata.get("page", "?")
            logger.debug("Page %s empty after cleaning -- skipped.", page)
            continue

        # Update the document's content (metadata stays intact)
        doc.page_content = cleaned_text
        cleaned.append(doc)

    original_count = len(documents)
    final_count = len(cleaned)

    if final_count < original_count:
        logger.info(
            "Cleaning: %d/%d documents retained (%d empty removed).",
            final_count, original_count, original_count - final_count,
        )

    return cleaned
