"""
comparison_agent.py -- Cross-paper comparative analysis.
Uses get_llm() -- works with Groq (free) or OpenAI.
"""

import time
from typing import List, Dict, Any
from loguru import logger
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from services.llm_service import get_llm
from config import get_settings

settings = get_settings()

COMPARISON_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior academic researcher writing a comparative analysis "
        "section for a literature review. Write in formal academic prose using markdown."
    ),
    (
        "human",
        """Compare and contrast the following {n} research papers on the topic of "{topic}".

{paper_summaries}

Write a comparative analysis covering:
1. **Methodological Approaches** -- how each paper's method differs
2. **Datasets & Experimental Setup** -- what data was used
3. **Results & Performance** -- key metrics and findings
4. **Strengths & Limitations** -- notable pros/cons of each approach
5. **Research Gaps** -- open problems or future directions

Use markdown headings (##). Cite papers by title in parentheses."""
    ),
])


class ComparisonAgent:
    """Generates a structured cross-paper comparison."""

    def __init__(self):
        self.llm = get_llm(temperature=0.4, max_tokens=2048)
        self.chain = COMPARISON_PROMPT | self.llm

    def run(self, papers: List[Dict[str, Any]], topic: str) -> str:
        if not papers:
            return "No papers available for comparison."
        if len(papers) == 1:
            return "Only one paper retrieved -- comparison requires multiple papers."

        logger.info(f"[ComparisonAgent] Comparing {len(papers)} papers on '{topic}'")
        paper_summaries = _format_papers_for_prompt(papers)

        for attempt in range(2):
            try:
                response = self.chain.invoke({
                    "n": len(papers),
                    "topic": topic,
                    "paper_summaries": paper_summaries,
                })
                logger.success("[ComparisonAgent] Comparison generated")
                return response.content.strip()

            except Exception as exc:
                err_str = str(exc).lower()
                if "rate_limit" in err_str or "429" in err_str or "quota" in err_str:
                    logger.warning("[ComparisonAgent] Rate limit -- waiting 20s...")
                    time.sleep(20)
                else:
                    logger.error(f"[ComparisonAgent] Attempt {attempt+1} error: {exc}")
                    if attempt == 0:
                        time.sleep(3)

        return f"Comparison could not be generated after retries. Check your API key and quota."


def _format_papers_for_prompt(papers: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, p in enumerate(papers, 1):
        insights = p.get("insights", {})
        block = (
            f"**Paper {i}: {p.get('title', 'Untitled')}**\n"
            f"- Authors: {', '.join(p.get('authors', [])[:3])}\n"
            f"- Year: {p.get('published', 'N/A')} | Source: {p.get('source', 'N/A')}\n"
            f"- Problem: {str(insights.get('problem_statement', 'N/A'))[:200]}\n"
            f"- Methodology: {str(insights.get('methodology', 'N/A'))[:200]}\n"
            f"- Datasets: {', '.join(insights.get('datasets', [])) or 'N/A'}\n"
            f"- Models: {', '.join(insights.get('models_used', [])) or 'N/A'}\n"
            f"- Results: {str(insights.get('key_results', 'N/A'))[:200]}\n"
        )
        blocks.append(block)
    return "\n---\n".join(blocks)
