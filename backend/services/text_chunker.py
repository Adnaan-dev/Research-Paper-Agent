"""
text_chunker.py — Splits paper text into overlapping chunks for embedding.

Uses LangChain's RecursiveCharacterTextSplitter which intelligently splits
on paragraph → sentence → word boundaries so chunks stay semantically coherent.
"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


# Default chunking parameters — tuned for abstracts + short bodies
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 150     # overlap to preserve cross-chunk context


def chunk_paper(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split a paper dict into text chunks, each carrying metadata.

    Args:
        paper: Paper dict with at least 'title', 'abstract', 'id', 'url', 'source'

    Returns:
        List of chunk dicts:
            {text, metadata: {paper_id, title, source, url, chunk_index}}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Compose the full text we want to index
    full_text = _build_text(paper)

    raw_chunks = splitter.split_text(full_text)
    logger.debug(f"[Chunker] '{paper['title'][:40]}' → {len(raw_chunks)} chunks")

    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "paper_id": paper.get("id", ""),
                "title": paper.get("title", ""),
                "source": paper.get("source", ""),
                "url": paper.get("url", ""),
                "authors": ", ".join(paper.get("authors", [])[:3]),  # top-3 authors
                "published": paper.get("published", ""),
                "chunk_index": idx,
            },
        })

    return chunks


def chunk_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chunk multiple papers and return a flat list of all chunks."""
    all_chunks = []
    for paper in papers:
        all_chunks.extend(chunk_paper(paper))
    logger.info(f"[Chunker] Total chunks across {len(papers)} papers: {len(all_chunks)}")
    return all_chunks


def _build_text(paper: Dict[str, Any]) -> str:
    """Combine paper fields into a single searchable text blob."""
    parts = [
        f"Title: {paper.get('title', '')}",
        f"Authors: {', '.join(paper.get('authors', [])[:5])}",
        f"Published: {paper.get('published', 'N/A')}",
        f"Source: {paper.get('source', '')}",
        "",
        "Abstract:",
        paper.get("abstract", "No abstract available."),
    ]
    return "\n".join(parts)
