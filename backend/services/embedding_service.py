"""
embedding_service.py — Provides embeddings via local SentenceTransformers (FREE)
                        or OpenAI (paid), controlled by EMBEDDING_PROVIDER in .env.

Default: LOCAL (all-MiniLM-L6-v2 runs on CPU, ~90 MB download, cached after first run)
"""

from functools import lru_cache
from loguru import logger
from config import get_settings

settings = get_settings()


@lru_cache()
def get_embeddings():
    """
    Return a cached embeddings object.

    LOCAL  → HuggingFaceEmbeddings (sentence-transformers, FREE, CPU-based)
    OPENAI → OpenAIEmbeddings (requires billing credits)
    """
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            logger.warning("[Embeddings] OPENAI_API_KEY not set — falling back to local embeddings")
            return _local_embeddings()
        logger.info(f"[Embeddings] Using OpenAI embeddings: {settings.openai_embedding_model}")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    else:
        return _local_embeddings()


def _local_embeddings():
    """Load local SentenceTransformers model — downloads once, then cached."""
    logger.info(f"[Embeddings] Loading local model: {settings.local_embedding_model}")
    logger.info("[Embeddings] First run may download ~90 MB — please wait...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.local_embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.success("[Embeddings] Local embedding model ready ✓")
    return embeddings
