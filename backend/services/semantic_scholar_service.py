"""
semantic_scholar_service.py — Fetches papers from the Semantic Scholar Graph API.

Semantic Scholar provides citation counts, field-of-study tags, and
open-access PDF links on top of the standard paper metadata.
"""

import httpx
from loguru import logger
from typing import List, Dict, Any

from config import get_settings

settings = get_settings()

BASE_URL = settings.semantic_scholar_base_url

# Fields we request from the API (keeps payload small)
FIELDS = (
    "paperId,title,abstract,authors,year,externalIds,"
    "openAccessPdf,url,citationCount,fieldsOfStudy"
)


def search_semantic_scholar(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search Semantic Scholar for papers matching the query.

    Args:
        query:       Free-text query string
        max_results: Number of papers to return (max 100 per API rules)

    Returns:
        Normalized list of paper dicts (same schema as arXiv service)
    """
    logger.info(f"[SemanticScholar] Searching: '{query}' (max {max_results})")

    headers = {"Content-Type": "application/json"}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": FIELDS,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{BASE_URL}/paper/search",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data = response.json()

        papers = []
        for item in data.get("data", []):
            # Extract abstract (may be None)
            abstract = item.get("abstract") or "Abstract not available."

            # Build author list
            authors = [a.get("name", "") for a in item.get("authors", [])]

            # Prefer open-access PDF link
            pdf_info = item.get("openAccessPdf") or {}
            pdf_url = pdf_info.get("url", "")

            # Canonical URL
            url = item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}"

            paper = {
                "id": item.get("paperId", ""),
                "title": item.get("title", "Untitled").strip(),
                "abstract": abstract.strip(),
                "authors": authors,
                "published": str(item.get("year", "N/A")),
                "url": url,
                "pdf_url": pdf_url,
                "source": "Semantic Scholar",
                "citation_count": item.get("citationCount", 0),
                "fields_of_study": item.get("fieldsOfStudy") or [],
            }
            papers.append(paper)
            logger.debug(f"[SemanticScholar] Found: {paper['title'][:60]}...")

        logger.success(f"[SemanticScholar] Retrieved {len(papers)} papers")
        return papers

    except httpx.HTTPStatusError as exc:
        logger.error(f"[SemanticScholar] HTTP error {exc.response.status_code}: {exc}")
        return []
    except Exception as exc:
        logger.error(f"[SemanticScholar] Unexpected error: {exc}")
        return []
