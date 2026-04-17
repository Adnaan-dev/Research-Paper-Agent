"""
reviser_agent.py — Incorporates reviewer feedback to improve the draft.

This is the Reviser node in the self-correction loop:

    Reviewer → Reviser → Writer (retry, max 2 times)

The Reviser takes the draft + reviewer feedback and produces
an improved version, specifically addressing the identified gaps.
"""

import time
from loguru import logger
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from services.llm_service import get_llm

REVISER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert academic editor. Improve a literature review draft "
        "based on peer reviewer feedback. Fix EVERY issue identified. "
        "Preserve all correct content — only add and fix, do not remove good sections."
    ),
    (
        "human",
        """Revise this literature review draft based on the reviewer feedback below.

TOPIC: {topic}

REVIEWER FEEDBACK:
{feedback}

CURRENT DRAFT:
{draft}

INSTRUCTIONS:
1. Fix every issue mentioned in the feedback
2. Add a markdown comparison table if missing (| Paper | Method | Dataset | Result |)
3. Add a Taxonomy section if missing (classify approaches into 2-3 categories)
4. Ensure all 6 sections are present and well-developed
5. Ensure [Author, Year] inline citations are used throughout
6. Keep the References section with all paper URLs

Return the complete improved literature review (all sections, no truncation)."""
    ),
])


class ReviserAgent:
    """
    Graph node that revises the draft based on reviewer feedback.
    Updates state['literature_draft'] with the improved version.
    """

    def __init__(self):
        self.llm = get_llm(temperature=0.4, max_tokens=4096)
        self.chain = REVISER_PROMPT | self.llm

    def run(self, state: dict) -> dict:
        """Graph node function — revises and returns updated state."""
        draft    = state.get("literature_draft", "")
        feedback = state.get("review_feedback", "")
        topic    = state["topic"]

        revision_count = state.get("revision_count", 0) + 1
        state["revision_count"] = revision_count
        logger.info(f"[ReviserAgent] Revision #{revision_count} based on feedback...")

        for attempt in range(2):
            try:
                response = self.chain.invoke({
                    "topic": topic,
                    "feedback": feedback[:1000],
                    "draft": draft[:5000],   # truncate to fit context
                })
                revised = response.content.strip()
                if len(revised) > 300:
                    state["literature_draft"] = revised
                    logger.success(f"[ReviserAgent] Revision #{revision_count} complete ({len(revised)} chars)")
                    return state

            except Exception as exc:
                err_str = str(exc).lower()
                if "rate_limit" in err_str or "429" in err_str:
                    logger.warning("[ReviserAgent] Rate limit — waiting 20s...")
                    time.sleep(20)
                else:
                    logger.error(f"[ReviserAgent] Attempt {attempt+1} failed: {exc}")
                    time.sleep(3)

        logger.warning("[ReviserAgent] Could not revise — keeping current draft")
        return state
