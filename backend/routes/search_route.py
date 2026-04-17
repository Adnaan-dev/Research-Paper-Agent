"""
search_route.py — FastAPI routes for the LangGraph research pipeline.

POST /search/      — compile + invoke the full LangGraph pipeline
GET  /search/status — vector store stats
GET  /search/data   — view all indexed chunks
DELETE /search/reset — clear vector store
GET  /search/export/pdf — export literature review as PDF
"""

import traceback
import os
from asyncio import get_event_loop
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from loguru import logger

from agents.coordinator_agent import CoordinatorAgent
from agents.search_agent import SearchAgent
from services.vector_store import get_collection_stats, reset_collection

router = APIRouter(prefix="/search", tags=["Research Pipeline"])

_executor = ThreadPoolExecutor(max_workers=2)
coordinator = CoordinatorAgent()


# ── Models ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500,
        example="transformer models for natural language processing")
    max_papers: int = Field(default=8, ge=2, le=20)


class SearchResponse(BaseModel):
    topic: str
    status: str
    total_papers: int
    sub_questions: list
    papers: list
    comparison_table: str
    comparison: str
    literature_review: str
    review_feedback: str
    revision_count: int
    review_passed: bool
    chunks_indexed: int
    node_log: list
    vector_store_stats: dict


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """Compile and invoke the LangGraph research pipeline."""
    logger.info(f"[Route /search] Topic: '{request.topic}'")

    def _run():
        return coordinator.run_full_pipeline(request.topic, request.max_papers)

    try:
        loop = get_event_loop()
        result = await loop.run_in_executor(_executor, _run)

        return SearchResponse(
            topic=result["topic"],
            status=result["status"],
            total_papers=len(result["papers"]),
            sub_questions=result.get("sub_questions", []),
            papers=result["papers"],
            comparison_table=result.get("comparison_table", ""),
            comparison=result.get("comparison", ""),
            literature_review=result["literature_review"],
            review_feedback=result.get("review_feedback", ""),
            revision_count=result.get("revision_count", 0),
            review_passed=result.get("review_passed", False),
            chunks_indexed=result.get("chunks_indexed", 0),
            node_log=result.get("node_log", []),
            vector_store_stats=result.get("vector_store_stats", {}),
        )

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[Route /search] Error:\n{tb}")
        raise HTTPException(status_code=500, detail=f"{exc}\n\n{tb}")


@router.get("/status")
async def get_status():
    return get_collection_stats()


@router.delete("/reset")
async def reset_store():
    reset_collection()
    return {"message": "Vector store reset."}


@router.get("/export/pdf")
async def export_pdf(review: str = ""):
    """
    Convert a markdown literature review to PDF.
    Pass review text as a query param, or POST with body.
    """
    if not review:
        raise HTTPException(status_code=400, detail="No review text provided")
    try:
        pdf_bytes = _markdown_to_pdf(review)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=literature_review.pdf"}
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}")


@router.post("/export/pdf")
async def export_pdf_post(body: dict):
    """POST version of PDF export — accepts {review: str} JSON body."""
    review = body.get("review", "")
    if not review:
        raise HTTPException(status_code=400, detail="No review text provided")
    try:
        pdf_bytes = _markdown_to_pdf(review)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=literature_review.pdf"}
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {exc}")


@router.get("/data")
async def view_stored_data():
    """Return all papers and chunks stored in ChromaDB."""
    from services.vector_store import _get_chroma_client, COLLECTION_NAME
    try:
        client = _get_chroma_client()
        col    = client.get_or_create_collection(COLLECTION_NAME)
        result = col.get(include=["documents", "metadatas"])
        ids, documents, metadatas = (
            result.get("ids", []),
            result.get("documents", []),
            result.get("metadatas", []),
        )
        if not ids:
            return {"total_chunks": 0, "total_papers": 0, "papers": [],
                    "message": "Vector store is empty."}
        papers = {}
        for cid, doc, meta in zip(ids, documents, metadatas):
            t = meta.get("title", "Unknown")
            if t not in papers:
                papers[t] = {**{k: meta.get(k,"") for k in ["title","source","url","authors","published"]},
                             "chunk_count": 0, "chunks": []}
            papers[t]["chunk_count"] += 1
            papers[t]["chunks"].append({
                "chunk_id": cid,
                "chunk_index": meta.get("chunk_index", 0),
                "text_preview": doc[:200] + "..." if len(doc) > 200 else doc,
            })
        for p in papers.values():
            p["chunks"].sort(key=lambda c: c["chunk_index"])
        papers_list = sorted(papers.values(), key=lambda p: p["title"])
        return {"total_chunks": len(ids), "total_papers": len(papers_list), "papers": papers_list}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── PDF helper ────────────────────────────────────────────────────────────────

def _markdown_to_pdf(markdown_text: str) -> bytes:
    """
    Convert markdown to PDF bytes.
    Tries weasyprint first, then reportlab, then plain text fallback.
    """
    # Strategy 1: weasyprint + markdown2
    try:
        import markdown2
        from weasyprint import HTML
        html_body = markdown2.markdown(
            markdown_text,
            extras=["tables", "fenced-code-blocks", "header-ids"]
        )
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  body {{ font-family: Georgia, serif; max-width: 800px; margin: 40px auto;
          font-size: 13px; line-height: 1.7; color: #222; }}
  h1 {{ font-size: 22px; color: #0d47a1; border-bottom: 2px solid #0d47a1; padding-bottom: 8px; }}
  h2 {{ font-size: 16px; color: #1565c0; margin-top: 24px; }}
  h3 {{ font-size: 14px; color: #1976d2; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 11px; }}
  th {{ background: #1565c0; color: white; padding: 6px 10px; text-align: left; }}
  td {{ border: 1px solid #ccc; padding: 5px 10px; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; font-size: 11px; }}
  blockquote {{ border-left: 4px solid #1565c0; margin: 0; padding-left: 16px; color: #555; }}
  a {{ color: #1565c0; }}
</style></head><body>{html_body}</body></html>"""
        return HTML(string=html).write_pdf()
    except ImportError:
        pass

    # Strategy 2: reportlab
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch
        import io
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=inch, rightMargin=inch,
                                topMargin=inch, bottomMargin=inch)
        styles = getSampleStyleSheet()
        story  = []
        for line in markdown_text.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 8))
            elif line.startswith("# "):
                story.append(Paragraph(line[2:], styles["Title"]))
            elif line.startswith("## "):
                story.append(Paragraph(line[3:], styles["Heading2"]))
            elif line.startswith("### "):
                story.append(Paragraph(line[4:], styles["Heading3"]))
            else:
                safe = line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                story.append(Paragraph(safe, styles["Normal"]))
        doc.build(story)
        return buf.getvalue()
    except ImportError:
        pass

    # Strategy 3: plain text PDF (always works)
    return _plain_text_pdf(markdown_text)


def _plain_text_pdf(text: str) -> bytes:
    """Minimal PDF that embeds the markdown as plain text — no dependencies."""
    lines = text.replace("\r\n", "\n").split("\n")
    pdf_lines = []
    for line in lines[:200]:   # cap pages
        safe = line.encode("latin-1", errors="replace").decode("latin-1")
        pdf_lines.append(safe[:120])

    content = "\n".join(pdf_lines)
    pdf = (
        "%PDF-1.4\n"
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842]\n"
        "   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        f"4 0 obj\n<< /Length {len(content) + 50} >>\nstream\n"
        "BT /F1 9 Tf 50 800 Td 12 TL\n"
        + "\n".join(f"({l.replace('(','').replace(')','')}) Tj T*" for l in pdf_lines[:60])
        + "\nET\nendstream\nendobj\n"
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        "xref\n0 6\n0000000000 65535 f\n"
        "0000000009 00000 n\n0000000068 00000 n\n0000000125 00000 n\n"
        "0000000280 00000 n\n0000000400 00000 n\n"
        "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n460\n%%EOF"
    )
    return pdf.encode("latin-1", errors="replace")
