"""
app/vectorstore/embedding_manager.py

Manages the embedding model as a singleton.

Why a singleton?
  The HuggingFace model (~90MB) takes 2-3 seconds to load into RAM.
  We load it ONCE and reuse the same instance for every request.
  Without this, every API call would wait 2-3 seconds for model loading.

Why a separate file?
  The vector store (chroma_store.py) doesn't need to know WHICH embedding
  model is being used. If you swap to OpenAI embeddings tomorrow, you
  change this one file and nothing else breaks.

Swapping to OpenAI (if you ever want to):
  Replace get_embedding_model() body with:
      from langchain_openai import OpenAIEmbeddings
      return OpenAIEmbeddings(model="text-embedding-3-small")
  Everything else stays identical.
"""

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache()
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns the singleton HuggingFace embedding model.

    lru_cache ensures the model is loaded exactly once.
    First call: ~2-3 seconds (downloads model if needed, loads into RAM).
    All subsequent calls: instant (returns cached instance).

    The model converts text into 384-dimensional vectors.
    These vectors capture semantic meaning -- similar text = similar vectors.

    Returns:
        HuggingFaceEmbeddings instance ready for use.
    """
    settings = get_settings()

    logger.info(
        "Loading embedding model: '%s' (device: %s) ...",
        settings.EMBEDDING_MODEL_NAME,
        settings.EMBEDDING_DEVICE,
    )

    model = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,  # L2 normalize for cosine similarity
            "batch_size": 64,              # process 64 texts at once (faster)
        },
    )

    # Verify it works with a test embedding
    test_vector = model.embed_query("test")
    logger.info(
        "Embedding model loaded successfully. Vector dimension: %d",
        len(test_vector),
    )

    return model
