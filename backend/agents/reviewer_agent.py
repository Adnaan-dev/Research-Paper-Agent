"""
reviewer_agent.py — Validates the draft literature review against quality criteria.

This is the Reviewer node in the LangGraph self-correction loop:

    Writer → Reviewer → (pass?) → Publisher
                              ↘ (fail?) → Reviser → Writer (retry)

The Reviewer checks for:
  - Completeness (all sections present)
  - Citation quality (proper [Author, Year] format)
  - Comparison table present
  - Taxonomy section present
  - Research gaps identified
  - No hallucinated claims
"""

import time
from loguru import logger
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from services.llm_service import get_llm

REVIEWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict academic peer reviewer evaluating a literature review draft. "
        "Be critical but constructive. Identify specific missing elements and quality issues."
    ),
    (
        "human",
        """Review this literature review draft and check ALL of these criteria:

DRAFT:
{draft}

TOPIC: {topic}
PAPER COUNT: {paper_count}

Check for each criterion (be specific):
1. Are all 6 sections present? (Introduction, Background, Paper Reviews, Comparison, Trends/Gaps, Conclusion)
2. Are inline citations in [Author, Year] format used consistently?
3. Is there a structured comparison table (markdown table with | columns |)?
4. Is there a taxonomy/classification of approaches?
5. Are research gaps explicitly identified?
6. Does the References section list all papers with URLs?

Respond in this format:
PASSED: yes/no
SCORE: X/10
MISSING: (list specific missing elements, or "none")
FEEDBACK: (2-3 sentences of specific actionable feedback for improvement)

Be strict — only mark PASSED: yes if score >= 7 and no critical sections are missing."""
    ),
])


class ReviewerAgent:
    """
    Graph node that validates the literature review draft.
    Returns updated state with review_feedback and review_passed.
    """

    def __init__(self):
        self.llm = get_llm(temperature=0.1, max_tokens=500)
        self.chain = REVIEWER_PROMPT | self.llm

    def run(self, state: dict) -> dict:
        """
        Graph node function.
        Evaluates the draft and sets state['review_passed'].
        """
        draft = state.get("literature_draft", "")
        topic = state["topic"]
        papers = state.get("papers", [])

        logger.info(f"[ReviewerAgent] Reviewing draft ({len(draft)} chars)...")

        # Quick structural check before calling LLM
        structural_issues = _structural_check(draft)

        if not draft or len(draft) < 500:
            state["review_feedback"] = "Draft is too short or empty. Regenerate with all 6 sections."
            state["review_passed"] = False
            return state

        for attempt in range(2):
            try:
                response = self.chain.invoke({
                    "draft": draft[:6000],   # truncate for context window
                    "topic": topic,
                    "paper_count": len(papers),
                })
                feedback_text = response.content.strip()
                passed = _parse_passed(feedback_text)
                score  = _parse_score(feedback_text)

                # Override: if structural issues found, fail
                if structural_issues:
                    passed = False
                    feedback_text += f"\n\nSTRUCTURAL ISSUES: {'; '.join(structural_issues)}"

                state["review_feedback"] = feedback_text
                state["review_passed"] = passed

                logger.info(f"[ReviewerAgent] Score: {score}/10 | Passed: {passed}")
                if structural_issues:
                    logger.warning(f"[ReviewerAgent] Structural issues: {structural_issues}")
                return state

            except Exception as exc:
                logger.error(f"[ReviewerAgent] Attempt {attempt+1} failed: {exc}")
                time.sleep(3)

        # If LLM fails — do structural check only
        state["review_passed"] = len(structural_issues) == 0
        state["review_feedback"] = (
            f"LLM review failed. Structural check: "
            + (f"Issues found: {'; '.join(structural_issues)}" if structural_issues else "OK")
        )
        return state


def _structural_check(draft: str) -> list:
    """Fast regex-free check for required sections."""
    issues = []
    required = ["introduction", "background", "conclusion", "references"]
    draft_lower = draft.lower()
    for section in required:
        if section not in draft_lower:
            issues.append(f"Missing '{section}' section")
    if "|" not in draft:
        issues.append("No markdown comparison table found")
    return issues


def _parse_passed(text: str) -> bool:
    text_lower = text.lower()
    for line in text_lower.split("\n"):
        if "passed:" in line:
            return "yes" in line
    return False


def _parse_score(text: str) -> int:
    import re
    match = re.search(r"score:\s*(\d+)", text.lower())
    if match:
        return int(match.group(1))
    return 5
