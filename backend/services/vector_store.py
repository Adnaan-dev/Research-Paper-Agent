"""
vector_store.py — Persistent ChromaDB vector store for paper chunks.

ChromaDB stores embeddings on disk so that data survives across
application restarts. Papers are deduplicated by chunk ID.
"""

import os
import hashlib
from typing import List, Dict, Any, Optional
from loguru import logger

import chromadb
from langchain_chroma import Chroma
from langchain.schema import Document

from services.embedding_service import get_embeddings
from config import get_settings

settings = get_settings()

# Chroma collection name — one collection per application
COLLECTION_NAME = "research_papers"


def _get_chroma_client() -> chromadb.PersistentClient:
    """Return (or create) the persistent Chroma client."""
    persist_dir = os.path.abspath(settings.chroma_persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def get_vector_store() -> Chroma:
    """Return the LangChain Chroma wrapper connected to our collection."""
    # For Vercel deployment, use in-memory store since file system is ephemeral
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


def add_chunks_to_store(chunks: List[Dict[str, Any]]) -> int:
    """
    Embed and upsert text chunks into ChromaDB.

    Chunks are deduplicated using a hash of (paper_id + chunk_index)
    so re-indexing the same paper is idempotent.

    Args:
        chunks: List of chunk dicts from text_chunker.chunk_papers()

    Returns:
        Number of NEW chunks added (skips duplicates)
    """
    if not chunks:
        logger.warning("[VectorStore] No chunks to add.")
        return 0

    store = get_vector_store()

    # Fetch existing IDs to avoid re-embedding duplicates
    existing_ids: set = set()
    try:
        client = _get_chroma_client()
        # get_collection raises if collection doesn't exist — use get_or_create instead
        col = client.get_or_create_collection(COLLECTION_NAME)
        result = col.get(include=[])
        existing_ids = set(result.get("ids", []))
    except Exception as e:
        logger.debug(f"[VectorStore] Could not fetch existing IDs (normal on first run): {e}")

    documents = []
    ids = []

    for chunk in chunks:
        chunk_id = _make_chunk_id(chunk)
        if chunk_id in existing_ids:
            continue  # skip duplicates

        doc = Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"],
        )
        documents.append(doc)
        ids.append(chunk_id)

    if not documents:
        logger.info("[VectorStore] All chunks already indexed — nothing new to add.")
        return 0

    logger.info(f"[VectorStore] Adding {len(documents)} new chunks to ChromaDB...")
    store.add_documents(documents=documents, ids=ids)
    logger.success(f"[VectorStore] ✓ Stored {len(documents)} chunks")
    return len(documents)


def similarity_search(
    query: str,
    k: int = 5,
    filter_source: Optional[str] = None,
) -> List[Document]:
    """
    Retrieve the k most relevant chunks for a query.

    Args:
        query:         User question or search phrase
        k:             Number of chunks to retrieve
        filter_source: Optional filter by source ("arXiv" / "Semantic Scholar")

    Returns:
        List of LangChain Document objects with page_content + metadata
    """
    store = get_vector_store()
    where = {"source": filter_source} if filter_source else None

    try:
        docs = store.similarity_search(query, k=k, filter=where)
        logger.debug(f"[VectorStore] Retrieved {len(docs)} chunks for: '{query[:50]}'")
        return docs
    except Exception as exc:
        logger.error(f"[VectorStore] Search error: {exc}")
        return []


def get_collection_stats() -> Dict[str, Any]:
    """Return basic stats about the current collection."""
    try:
        col = _get_chroma_client().get_or_create_collection(COLLECTION_NAME)
        count = col.count()
        return {"total_chunks": count, "collection": COLLECTION_NAME}
    except Exception as e:
        logger.debug(f"[VectorStore] Stats error: {e}")
        return {"total_chunks": 0, "collection": COLLECTION_NAME}


def reset_collection() -> None:
    """Drop and recreate the collection (useful for testing)."""
    client = _get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.warning("[VectorStore] Collection deleted.")
    except Exception:
        pass


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_chunk_id(chunk: Dict[str, Any]) -> str:
    """Stable, unique ID for a chunk based on paper_id + chunk_index."""
    raw = f"{chunk['metadata']['paper_id']}_{chunk['metadata']['chunk_index']}"
    return hashlib.md5(raw.encode()).hexdigest()
