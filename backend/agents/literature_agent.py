"""
literature_agent.py — Writer node: generates full structured literature review.

Produces ALL required output elements:
  - 6-section academic review with [Author, Year] citations
  - Structured markdown comparison table
  - Taxonomy / classification of approaches
  - Identified research gaps
  - References with URLs
"""

import time
from typing import List, Dict, Any
from loguru import logger # type: ignore
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from services.llm_service import get_llm
from config import get_settings

settings = get_settings()

REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a distinguished academic researcher and technical writer. "
        "Generate a comprehensive, well-cited literature review in formal academic English using markdown. "
        "You MUST include a comparison table and taxonomy section — these are mandatory."
    ),
    (
        "human",
        """Write a complete structured literature review on: "{topic}"

Based on {n} research papers:
{paper_data}

COMPARATIVE ANALYSIS (incorporate):
{comparison}

MANDATORY STRUCTURE — include ALL of these exactly:

# Literature Review: {topic}

## 1. Introduction
(2-3 paragraphs: topic importance, scope, why this matters)

## 2. Background & Context
(foundational concepts, evolution of the field)

## 3. Taxonomy of Approaches
(classify the papers into 2-4 categories/approaches with brief description of each category)

## 4. Review of Key Papers
(each paper discussed with [Author, Year] inline citations)

## 5. Comparison of Methods

### 5.1 Comparison Table
| Paper | Year | Method | Dataset | Key Result | Limitation |
|-------|------|--------|---------|------------|------------|
(fill one row per paper — this table is MANDATORY)

### 5.2 Analysis
(synthesise the comparison — 2 paragraphs)

## 6. Research Gaps & Open Challenges
(explicitly list 4-5 gaps as bullet points)

## 7. Conclusion
(summary + future directions)

## References
[1] Authors. "Title". Source. Year. URL
(one entry per paper)

Rules:
- Use [Author et al., Year] for all inline citations
- Fill every cell in the comparison table
- Be specific with metrics and numbers where available"""
    ),
])

TABLE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research analyst. Generate only a clean markdown comparison table. "
        "No prose, no headings — just the table."
    ),
    (
        "human",
        """Create a markdown comparison table for these {n} papers on "{topic}":

{paper_data}

Format EXACTLY like this (fill every cell — use N/A if unknown):
| Paper | Year | Method/Approach | Dataset | Key Metric | Result | Limitation |
|-------|------|-----------------|---------|------------|--------|------------|
| Short title | year | approach | dataset name | metric | value | limitation |

One row per paper. Short titles (max 5 words). Be specific."""
    ),
])


class LiteratureReviewAgent:
    """Writer node — generates the full structured literature review draft."""

    def __init__(self):
        self.llm = get_llm(temperature=0.5, max_tokens=4096)
        self.table_llm = get_llm(temperature=0.1, max_tokens=1000)
        self.chain = REVIEW_PROMPT | self.llm
        self.table_chain = TABLE_PROMPT | self.table_llm

    def run(self, state: dict) -> dict:
        """Graph node function — generates draft and stores in state."""
        topic      = state["topic"]
        papers     = state.get("papers", [])
        comparison = state.get("comparison_prose", "")
        logger.info(f"[LiteratureAgent] Writing review for '{topic}' ({len(papers)} papers)...")

        paper_data = _format_papers(papers)

        # Generate comparison table separately for reliability
        comparison_table = self._generate_table(papers, topic, paper_data)
        state["comparison_table"] = comparison_table

        for attempt in range(2):
            try:
                response = self.chain.invoke({
                    "topic": topic,
                    "n": len(papers),
                    "paper_data": paper_data,
                    "comparison": comparison or "See comparison table above.",
                })
                draft = response.content.strip()

                # Inject the standalone table if not already present
                if comparison_table and "| Paper |" not in draft:
                    inject = f"\n\n## 5. Comparison of Methods\n\n### 5.1 Comparison Table\n\n{comparison_table}\n"
                    draft = draft + inject

                state["literature_draft"] = draft
                logger.success(f"[LiteratureAgent] Draft generated ({len(draft)} chars)")
                return state

            except Exception as exc:
                err_str = str(exc).lower()
                if "rate_limit" in err_str or "429" in err_str or "quota" in err_str:
                    logger.warning("[LiteratureAgent] Rate limit — waiting 20s...")
                    time.sleep(20)
                else:
                    logger.error(f"[LiteratureAgent] Attempt {attempt+1}: {exc}")
                    if attempt == 0:
                        time.sleep(3)

        state["literature_draft"] = _fallback_review(papers, topic, comparison_table)
        return state

    def run_legacy(self, papers, topic, comparison="") -> str:
        """Legacy interface for backward compatibility with old coordinator."""
        state = {"topic": topic, "papers": papers, "comparison_prose": comparison,
                 "comparison_table": "", "literature_draft": ""}
        state = self.run(state)
        return state.get("literature_draft", "")

    def _generate_table(self, papers, topic, paper_data) -> str:
        """Generate the structured comparison table separately."""
        if not papers:
            return ""
        try:
            response = self.table_chain.invoke({
                "n": len(papers),
                "topic": topic,
                "paper_data": paper_data[:3000],
            })
            table = response.content.strip()
            if "|" in table:
                logger.success("[LiteratureAgent] Comparison table generated")
                return table
        except Exception as exc:
            logger.warning(f"[LiteratureAgent] Table generation failed: {exc}")
        return _fallback_table(papers)


def _format_papers(papers: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, p in enumerate(papers, 1):
        insights = p.get("insights", {})
        authors  = p.get("authors", ["Unknown"])
        authors_short = authors[0].split()[-1] if authors else "Unknown"
        summary  = (p.get("summary") or p.get("abstract") or "")[:350]
        block = (
            f"[{i}] {authors_short} et al., {p.get('published', 'N/A')}\n"
            f"Title: {p.get('title', 'Untitled')}\n"
            f"Authors: {', '.join(authors[:3])}\n"
            f"Year: {p.get('published','N/A')} | Source: {p.get('source','')}\n"
            f"URL: {p.get('url','')}\n"
            f"Summary: {summary}\n"
            f"Problem: {str(insights.get('problem_statement',''))[:180]}\n"
            f"Method: {str(insights.get('methodology',''))[:180]}\n"
            f"Models: {', '.join(insights.get('models_used',[]))[:120]}\n"
            f"Datasets: {', '.join(insights.get('datasets',[]))[:120]}\n"
            f"Results: {str(insights.get('key_results',''))[:180]}\n"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _fallback_table(papers: List[Dict[str, Any]]) -> str:
    rows = ["| Paper | Year | Method | Dataset | Result | Limitation |",
            "|-------|------|--------|---------|--------|------------|"]
    for p in papers:
        insights = p.get("insights", {})
        title_short = p.get("title","Untitled")[:40]
        year   = p.get("published", "N/A")
        method = str(insights.get("methodology","N/A"))[:50]
        data   = ", ".join(insights.get("datasets",[]) or ["N/A"])[:40]
        result = str(insights.get("key_results","N/A"))[:50]
        rows.append(f"| {title_short} | {year} | {method} | {data} | {result} | Not specified |")
    return "\n".join(rows)


def _fallback_review(papers, topic, table="") -> str:
    lines = [f"# Literature Review: {topic}\n",
             "*Auto-generated fallback — LLM unavailable.*\n"]
    if table:
        lines.append("## Comparison Table\n")
        lines.append(table + "\n")
    for i, p in enumerate(papers, 1):
        lines.append(f"## [{i}] {p.get('title','Untitled')}")
        lines.append(f"**Source:** {p.get('source')} | **Year:** {p.get('published')}")
        lines.append(f"**URL:** {p.get('url','')}\n")
        lines.append(p.get("summary", p.get("abstract","No summary available.")))
        lines.append("")
    return "\n".join(lines)
