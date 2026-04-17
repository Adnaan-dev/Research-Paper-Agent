"""
qa_agent.py -- RAG-based QA agent.
Uses get_llm() -- works with Groq (free) or OpenAI.
"""

import time
from typing import List, Dict, Any, Tuple
from loguru import logger
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
# from langchain.schema import Document
from langchain_core.documents import Document

from services.vector_store import similarity_search
from services.llm_service import get_llm
from config import get_settings

settings = get_settings()

QA_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a knowledgeable research assistant with deep expertise in academic literature. "
        "Answer the user's question using ONLY the provided context from research papers. "
        "If the context does not contain enough information, say so honestly. "
        "Always cite the specific paper(s) you used using the title and URL provided in the context."
    ),
    (
        "human",
        """Use the following research paper excerpts to answer the question.

CONTEXT FROM PAPERS:
{context}

---
QUESTION: {question}

Instructions:
- Base your answer strictly on the provided context
- Cite papers using: (Source: "Paper Title", URL)
- If answer cannot be found, state: "The provided papers do not contain information about this topic."
- Use markdown formatting for clarity"""
    ),
])


class QAAgent:
    """Retrieval-Augmented Generation QA agent."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.llm = get_llm(temperature=0.2, max_tokens=1500)
        self.chain = QA_PROMPT | self.llm

    def run(self, question: str) -> Dict[str, Any]:
        logger.info(f"[QAAgent] Question: '{question[:80]}'")

        docs = similarity_search(question, k=self.top_k)

        if not docs:
            return {
                "answer": (
                    "No relevant papers are currently indexed. "
                    "Please search and analyse some papers first."
                ),
                "sources": [],
                "chunks": 0,
            }

        context, sources = self._build_context(docs)

        for attempt in range(2):
            try:
                response = self.chain.invoke({"context": context, "question": question})
                logger.success("[QAAgent] Answer generated")
                return {"answer": response.content.strip(), "sources": sources, "chunks": len(docs)}
            except Exception as exc:
                err_str = str(exc).lower()
                if "rate_limit" in err_str or "429" in err_str or "quota" in err_str:
                    logger.warning("[QAAgent] Rate limit -- waiting 15s...")
                    time.sleep(15)
                else:
                    logger.error(f"[QAAgent] LLM call failed: {exc}")
                    return {"answer": f"Error generating answer: {exc}", "sources": sources, "chunks": len(docs)}

        return {"answer": "Could not generate answer after retries.", "sources": sources, "chunks": len(docs)}

    def _build_context(self, docs: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
        context_parts = []
        sources = []
        seen_papers: set = set()

        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            paper_title = meta.get("title", "Unknown")
            paper_url = meta.get("url", "")
            context_parts.append(
                f"[Excerpt {i}]\nTitle: {paper_title}\n"
                f"Authors: {meta.get('authors','')}\nYear: {meta.get('published','')} | Source: {meta.get('source','')}\n"
                f"URL: {paper_url}\nContent: {doc.page_content}\n"
            )
            if paper_title not in seen_papers:
                seen_papers.add(paper_title)
                sources.append({
                    "title": paper_title, "url": paper_url,
                    "source": meta.get("source", ""), "authors": meta.get("authors", ""),
                    "published": meta.get("published", ""),
                })

        return "\n---\n".join(context_parts), sources
