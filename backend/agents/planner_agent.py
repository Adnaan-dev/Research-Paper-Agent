"""
planner_agent.py — Decomposes the research topic into sub-questions.

In LangGraph this is the first node in the directed graph.
The Planner breaks the user's topic into 3-5 targeted sub-questions
that guide the Searcher and Reader agents downstream.
"""

import time
from typing import List
from loguru import logger
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

from services.llm_service import get_llm

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research planning expert. Decompose research topics into "
        "precise, searchable sub-questions that will guide a literature review."
    ),
    (
        "human",
        """Decompose this research topic into 4-5 specific sub-questions for a literature review:

Topic: "{topic}"

Return ONLY a numbered list of sub-questions, one per line. No explanations.
Each question should target a different aspect: methods, datasets, results, applications, limitations."""
    ),
])


class PlannerAgent:
    """
    Node 1 in the LangGraph pipeline.
    Decomposes the topic into sub-questions that structure the search.
    """

    def __init__(self):
        self.llm = get_llm(temperature=0.3, max_tokens=400)
        self.chain = PLANNER_PROMPT | self.llm

    def run(self, state: dict) -> dict:
        """Graph node function — takes state, returns updated state."""
        topic = state["topic"]
        logger.info(f"[PlannerAgent] Decomposing topic: '{topic}'")

        try:
            response = self.chain.invoke({"topic": topic})
            lines = response.content.strip().split("\n")
            questions = []
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    # Strip leading numbers like "1." or "1)"
                    import re
                    cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
                    if cleaned:
                        questions.append(cleaned)

            state["sub_questions"] = questions[:5]
            logger.success(f"[PlannerAgent] Generated {len(questions)} sub-questions")

        except Exception as exc:
            logger.error(f"[PlannerAgent] Failed: {exc} — using default questions")
            state["sub_questions"] = [
                f"What are the main methodologies used in {topic}?",
                f"What datasets are commonly used to evaluate {topic}?",
                f"What are the state-of-the-art results in {topic}?",
                f"What are the open challenges in {topic}?",
                f"How has {topic} evolved over the past 5 years?",
            ]

        return state
