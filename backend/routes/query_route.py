"""
query_route.py — FastAPI router for the RAG-based Q&A endpoint.

POST /query  — answer a user question from indexed paper chunks
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from agents.coordinator_agent import CoordinatorAgent

router = APIRouter(prefix="/query", tags=["Q&A"])

coordinator = CoordinatorAgent()


# ── Request / Response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Natural language question about the indexed papers",
        example="What datasets are commonly used in transformer-based NLP research?",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Number of context chunks to retrieve from the vector store",
    )


class SourceOut(BaseModel):
    title: str
    url: str
    source: str
    authors: str = ""
    published: str = ""


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    chunks_retrieved: int
    status: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/", response_model=QueryResponse, summary="Ask a question about indexed papers")
async def query_papers(request: QueryRequest):
    """
    RAG-based Q&A endpoint.

    Retrieves relevant paper chunks from ChromaDB using vector similarity,
    then asks GPT-4o-mini to answer based on those chunks.

    Requires at least one prior call to POST /search to index papers.
    """
    logger.info(f"[Route /query] Question: '{request.question[:60]}'")

    try:
        # Set the QA agent's top_k for this request
        coordinator._qa_agent = None
        from agents.qa_agent import QAAgent
        coordinator._qa_agent = QAAgent(top_k=request.top_k)

        result = coordinator.run_qa(request.question)

        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"],
            chunks_retrieved=result["chunks"],
            status=result["status"],
        )

    except Exception as exc:
        logger.error(f"[Route /query] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
