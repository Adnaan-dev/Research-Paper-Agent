"""
summarizer_agent.py — Generates concise, structured summaries for each paper.
Uses get_llm() which is Groq (free) or OpenAI depending on your .env config.
"""

import time
from typing import List, Dict, Any
from loguru import logger
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from services.llm_service import get_llm
from config import get_settings

settings = get_settings()

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert academic research assistant. "
        "Produce clear, accurate, concise summaries of research papers for a literature review. "
        "Write in formal, third-person academic style."
    ),
    (
        "human",
        """Summarize this research paper in 3-5 sentences.
Focus on: what problem it solves, how it solves it, and the main result.

Title: {title}
Authors: {authors}
Published: {published}
Source: {source}

Abstract:
{abstract}

Write ONLY the summary paragraph. No headings, no bullet points."""
    ),
])


class SummarizerAgent:
    """Generates plain-language summaries for a batch of papers."""

    def __init__(self):
        self.llm = get_llm(temperature=0.3, max_tokens=512)
        self.chain = SUMMARY_PROMPT | self.llm

    def run(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add a 'summary' field to each paper dict."""
        logger.info(f"[SummarizerAgent] Summarising {len(papers)} papers...")
        enriched = []
        for i, paper in enumerate(papers, 1):
            logger.debug(f"[SummarizerAgent] {i}/{len(papers)}: {paper['title'][:50]}...")
            summary = self._summarise(paper)
            paper = {**paper, "summary": summary}
            enriched.append(paper)
            if i < len(papers):
                time.sleep(0.5)

        logger.success(f"[SummarizerAgent] Done -- {len(enriched)} summaries generated")
        return enriched

    def _summarise(self, paper: Dict[str, Any]) -> str:
        """Generate a summary with retry on transient errors."""
        abstract = paper.get("abstract", "")
        if not abstract or abstract == "Abstract not available.":
            return (
                f"This paper titled '{paper.get('title','')}' was retrieved from "
                f"{paper.get('source','')}. Full abstract not available."
            )

        for attempt in range(3):
            try:
                response = self.chain.invoke({
                    "title": paper.get("title", ""),
                    "authors": ", ".join(paper.get("authors", [])[:5]),
                    "published": paper.get("published", "N/A"),
                    "source": paper.get("source", ""),
                    "abstract": abstract[:2000],
                })
                return response.content.strip()

            except Exception as exc:
                err_str = str(exc).lower()
                if "rate_limit" in err_str or "429" in err_str or "quota" in err_str:
                    wait = 15 * (attempt + 1)
                    logger.warning(f"[SummarizerAgent] Rate limit -- waiting {wait}s (attempt {attempt+1}/3)...")
                    time.sleep(wait)
                elif attempt < 2:
                    logger.warning(f"[SummarizerAgent] Attempt {attempt+1} failed: {exc} -- retrying...")
                    time.sleep(3)
                else:
                    logger.error(f"[SummarizerAgent] All retries failed: {exc}")

        return abstract[:500] + ("..." if len(abstract) > 500 else "")
