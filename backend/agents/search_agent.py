"""
search_agent.py — Orchestrates paper fetching from arXiv + Semantic Scholar.

Responsibilities:
  * Query both APIs in sequence
  * Deduplicate results by title similarity
  * Return a unified list of paper dicts
"""

from typing import List, Dict, Any
from loguru import logger

from services.arxiv_service import search_arxiv
from services.semantic_scholar_service import search_semantic_scholar
from config import get_settings

settings = get_settings()


class SearchAgent:
    """
    Fetches and merges papers from multiple academic APIs.
    Each source contributes up to half of max_papers.
    """

    def __init__(self, max_papers: int = None):
        self.max_papers = max_papers or settings.max_papers
        self.per_source = max(1, self.max_papers // 2)

    # ── public API ────────────────────────────────────────────────────────

    def run(self, topic: str) -> List[Dict[str, Any]]:
        """
        Search both APIs and return deduplicated, ranked papers.

        Args:
            topic: Research topic string

        Returns:
            List of paper dicts (combined from all sources)
        """
        logger.info(f"[SearchAgent] Starting search for: '{topic}'")

        arxiv_papers = self._fetch_arxiv(topic)
        scholar_papers = self._fetch_semantic_scholar(topic)

        # Merge and deduplicate
        all_papers = self._deduplicate(arxiv_papers + scholar_papers)

        # Limit to max_papers
        all_papers = all_papers[: self.max_papers]

        logger.success(
            f"[SearchAgent] Found {len(all_papers)} unique papers "
            f"({len(arxiv_papers)} arXiv + {len(scholar_papers)} Scholar)"
        )
        return all_papers

    # ── private helpers ───────────────────────────────────────────────────

    def _fetch_arxiv(self, topic: str) -> List[Dict[str, Any]]:
        try:
            return search_arxiv(topic, max_results=self.per_source)
        except Exception as exc:
            logger.error(f"[SearchAgent] arXiv error: {exc}")
            return []

    def _fetch_semantic_scholar(self, topic: str) -> List[Dict[str, Any]]:
        try:
            return search_semantic_scholar(topic, max_results=self.per_source)
        except Exception as exc:
            logger.error(f"[SearchAgent] Semantic Scholar error: {exc}")
            return []

    def _deduplicate(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate papers using normalised title matching.
        Keeps the first occurrence (preserving source order).
        """
        seen_titles: set = set()
        unique = []
        for paper in papers:
            key = _normalise_title(paper.get("title", ""))
            if key not in seen_titles:
                seen_titles.add(key)
                unique.append(paper)
        removed = len(papers) - len(unique)
        if removed:
            logger.debug(f"[SearchAgent] Removed {removed} duplicates")
        return unique


# ── module-level helper ───────────────────────────────────────────────────────

def _normalise_title(title: str) -> str:
    """Lower-case, strip punctuation for fuzzy dedup."""
    import re
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()
