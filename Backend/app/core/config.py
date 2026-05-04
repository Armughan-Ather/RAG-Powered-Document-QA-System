"""
app/core/config.py

Centralised application configuration using Pydantic-Settings.
All values are read from environment variables or the .env file.
Validated at startup — the app will REFUSE to boot with bad config.
"""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: Backend/
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Single source of truth for all application configuration.
    Uses lru_cache (see get_settings) so it is instantiated only once.
    """

    # ── App ──────────────────────────────────────────────────────
    APP_NAME: str = "Intelligent Document Analysis System"
    APP_VERSION: str = "1.0.0"
    ENV: str = Field(default="development")
    DEBUG: bool = False

    # ── API ──────────────────────────────────────────────────────
    API_V1_PREFIX: str = "/api/v1"

    # ── Security ─────────────────────────────────────────────────
    SECRET_KEY: str = Field(default="change-me-in-production")
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
    ]

    # ── OpenAI ───────────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_TOKENS: int = 2048
    OPENAI_TEMPERATURE: float = 0.0

    # ── Embeddings ────────────────────────────────────────────────
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"  # "cuda" for GPU acceleration

    # ── Reranker ──────────────────────────────────────────────────
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_TOP_N: int = 5      # chunks kept after reranking

    # ── Chunking ─────────────────────────────────────────────────
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # ── Retrieval ────────────────────────────────────────────────
    RETRIEVAL_K: int = 15        # candidates fetched from vector store (wider net)
    RETRIEVAL_TOP_K: int = 8     # final chunks after MMR reranking
    MMR_LAMBDA: float = 0.5      # 0.0=max diversity, 1.0=max relevance

    # ── Vector Store ─────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma_db")
    CHROMA_COLLECTION_NAME: str = "documents"

    # ── File Upload ───────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".txt", ".docx"]
    UPLOAD_DIR: str = str(BASE_DIR / "data" / "uploads")

    # ── Pydantic-Settings config ──────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",          # silently discard unknown env vars
    )

    # ── Validators ────────────────────────────────────────────────

    @field_validator("ENV")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"ENV must be one of {allowed}, got '{v}'")
        return v

    @field_validator("OPENAI_TEMPERATURE")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("OPENAI_TEMPERATURE must be between 0.0 and 2.0")
        return v

    @field_validator("MAX_FILE_SIZE_MB")
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        if not 1 <= v <= 500:
            raise ValueError("MAX_FILE_SIZE_MB must be between 1 and 500")
        return v

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_overlap(cls, v: int) -> int:
        if v < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative")
        return v

    @model_validator(mode="after")
    def validate_chunk_overlap_lt_chunk_size(self) -> "Settings":
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be less than "
                f"CHUNK_SIZE ({self.CHUNK_SIZE})"
            )
        return self

    @model_validator(mode="after")
    def warn_missing_openai_key(self) -> "Settings":
        if not self.OPENAI_API_KEY and self.ENV == "production":
            raise ValueError("OPENAI_API_KEY is required in production")
        return self

    # ── Computed properties ───────────────────────────────────────

    @property
    def max_file_size_bytes(self) -> int:
        """File size limit in bytes (used in upload validation)."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def chroma_persist_path(self) -> Path:
        """Resolved Path object for Chroma persistence directory."""
        return Path(self.CHROMA_PERSIST_DIR)

    @property
    def upload_path(self) -> Path:
        """Resolved Path object for file uploads directory."""
        return Path(self.UPLOAD_DIR)

    @property
    def is_production(self) -> bool:
        return self.ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.ENV == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Returns the singleton Settings instance.
    lru_cache ensures .env is read exactly once — fast and safe.
    Use as a FastAPI dependency: Depends(get_settings)
    """
    return Settings()
