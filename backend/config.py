"""
config.py — Central configuration using Pydantic Settings.
Reads from environment variables / .env file.

LLM PROVIDER OPTIONS:
  - "groq"   → FREE. Uses Groq cloud API (llama-3.3-70b). Get key at console.groq.com
  - "openai" → Paid. Uses OpenAI GPT-4o-mini. Requires billing credits.

EMBEDDING OPTIONS:
  - "local"  → FREE. Uses sentence-transformers running on your CPU (no API needed).
  - "openai" → Paid. Uses text-embedding-3-small via OpenAI API.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings

_ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")


class Settings(BaseSettings):
    # ── LLM Provider: "groq" (free) or "openai" (paid) ──────────────────────
    llm_provider: str = "groq"

    # ── Groq (FREE) ──────────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"   # best free Groq model

    # ── OpenAI (PAID, optional fallback) ─────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Embeddings: "local" (free) or "openai" (paid) ────────────────────────
    embedding_provider: str = "local"
    # Best small model that runs on CPU — downloads ~90 MB once then cached
    local_embedding_model: str = "all-MiniLM-L6-v2"

    # ── Semantic Scholar ──────────────────────────────────────────────────────
    semantic_scholar_api_key: str = ""
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"

    # ── arXiv ─────────────────────────────────────────────────────────────────
    arxiv_max_results: int = 5

    # ── App ───────────────────────────────────────────────────────────────────
    max_papers: int = 10
    chroma_persist_dir: str = "./data/chroma_db"
    papers_dir: str = "./data/papers"
    log_level: str = "INFO"

    # ── FastAPI ───────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Frontend ──────────────────────────────────────────────────────────────
    api_base_url: str = "http://localhost:8000"

    class Config:
        env_file = _ENV_FILE
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton settings object."""
    return Settings()
