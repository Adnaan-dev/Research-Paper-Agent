"""
arxiv_service.py — Fetches research papers from the arXiv API.

Uses the official `arxiv` Python client library which wraps the arXiv API.
Returns a normalized list of paper dicts so the rest of the system is
API-agnostic.
"""

import arxiv
from loguru import logger
from typing import List, Dict, Any

from config import get_settings

settings = get_settings()


def search_arxiv(query: str, max_results: int = None) -> List[Dict[str, Any]]:
    """
    Search arXiv for papers matching the query.

    Args:
        query:       Free-text search query (e.g. "transformer attention mechanism")
        max_results: Maximum number of papers to return (defaults to settings value)

    Returns:
        List of paper dicts with keys:
            id, title, abstract, authors, published, url, source
    """
    if max_results is None:
        max_results = settings.arxiv_max_results

    logger.info(f"[arXiv] Searching for: '{query}' (max {max_results} results)")

    try:
        # Build and execute the search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers = []
        client = arxiv.Client()

        for result in client.results(search):
            paper = {
                "id": result.entry_id,                         # arXiv URL like https://arxiv.org/abs/xxxx.xxxxx
                "title": result.title.strip(),
                "abstract": result.summary.strip(),
                "authors": [str(a) for a in result.authors],
                "published": result.published.strftime("%Y-%m-%d") if result.published else "N/A",
                "url": result.entry_id,                        # direct link to the paper
                "pdf_url": result.pdf_url,
                "source": "arXiv",
                "categories": result.categories,
            }
            papers.append(paper)
            logger.debug(f"[arXiv] Found: {paper['title'][:60]}...")

        logger.success(f"[arXiv] Retrieved {len(papers)} papers")
        return papers

    except Exception as exc:
        logger.error(f"[arXiv] Search failed: {exc}")
        return []
