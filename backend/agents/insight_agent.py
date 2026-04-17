"""
insight_agent.py — Extracts structured insights from each paper via LLM.
Uses get_llm() -- works with Groq (free) or OpenAI.
"""

import json
import re
import time
from typing import List, Dict, Any
from loguru import logger
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from services.llm_service import get_llm
from config import get_settings

settings = get_settings()

INSIGHT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research analyst extracting structured information from academic papers. "
        "Always respond with VALID JSON only -- no markdown fences, no extra text."
    ),
    (
        "human",
        """Extract the following information from the research paper below.
Return ONLY a JSON object with these exact keys:
  "problem_statement": the core problem or research gap being addressed
  "methodology":       the approach, technique, or framework proposed
  "datasets":          list of strings -- datasets used for experiments
  "models_used":       list of strings -- ML models, algorithms, or architectures used
  "key_results":       main findings, metrics, or conclusions

Title: {title}
Authors: {authors}
Year: {published}

Abstract:
{abstract}

Summary:
{summary}

Return ONLY valid JSON. If a field is unknown, use empty string or empty list."""
    ),
])

EMPTY_INSIGHTS: Dict[str, Any] = {
    "problem_statement": "Not available",
    "methodology": "Not available",
    "datasets": [],
    "models_used": [],
    "key_results": "Not available",
}


class InsightAgent:
    """Extracts structured key insights from research papers via LLM."""

    def __init__(self):
        self.llm = get_llm(temperature=0.0, max_tokens=600)
        self.chain = INSIGHT_PROMPT | self.llm

    def run(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add an 'insights' dict to each paper."""
        logger.info(f"[InsightAgent] Extracting insights from {len(papers)} papers...")
        enriched = []
        for i, paper in enumerate(papers, 1):
            logger.debug(f"[InsightAgent] {i}/{len(papers)}: {paper['title'][:50]}...")
            insights = self._extract(paper)
            paper = {**paper, "insights": insights}
            enriched.append(paper)
            if i < len(papers):
                time.sleep(0.5)

        logger.success(f"[InsightAgent] Done -- insights extracted for {len(enriched)} papers")
        return enriched

    def _extract(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Run LLM extraction with retry."""
        for attempt in range(3):
            try:
                response = self.chain.invoke({
                    "title": paper.get("title", ""),
                    "authors": ", ".join(paper.get("authors", [])[:5]),
                    "published": paper.get("published", "N/A"),
                    "abstract": paper.get("abstract", "")[:1500],
                    "summary": paper.get("summary", paper.get("abstract", ""))[:500],
                })
                return _parse_json(response.content)

            except Exception as exc:
                err_str = str(exc).lower()
                if "rate_limit" in err_str or "429" in err_str or "quota" in err_str:
                    wait = 15 * (attempt + 1)
                    logger.warning(f"[InsightAgent] Rate limit -- waiting {wait}s...")
                    time.sleep(wait)
                elif attempt < 2:
                    logger.warning(f"[InsightAgent] Attempt {attempt+1} failed: {exc}")
                    time.sleep(3)
                else:
                    logger.error(f"[InsightAgent] All retries failed for '{paper.get('title','?')[:40]}': {exc}")

        return EMPTY_INSIGHTS.copy()


def _parse_json(raw: str) -> Dict[str, Any]:
    """Robustly parse JSON from LLM output."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(cleaned)
        return {
            "problem_statement": str(data.get("problem_statement", "Not available")),
            "methodology": str(data.get("methodology", "Not available")),
            "datasets": _ensure_list(data.get("datasets", [])),
            "models_used": _ensure_list(data.get("models_used", [])),
            "key_results": str(data.get("key_results", "Not available")),
        }
    except json.JSONDecodeError:
        pass

    # Try extracting first {...} block
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return {
                "problem_statement": str(data.get("problem_statement", "Not available")),
                "methodology": str(data.get("methodology", "Not available")),
                "datasets": _ensure_list(data.get("datasets", [])),
                "models_used": _ensure_list(data.get("models_used", [])),
                "key_results": str(data.get("key_results", "Not available")),
            }
        except json.JSONDecodeError:
            pass

    logger.warning(f"[InsightAgent] JSON parse failed. Raw: {raw[:200]}")
    return EMPTY_INSIGHTS.copy()


def _ensure_list(val) -> list:
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str) and val:
        return [val]
    return []
