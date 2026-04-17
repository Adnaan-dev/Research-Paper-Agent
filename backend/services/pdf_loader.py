"""
pdf_loader.py — Downloads and extracts text from open-access PDFs.

When a paper has a PDF link we try to download and parse it so the
vector store gets more content than just the abstract.  If downloading
fails we fall back gracefully to abstract-only indexing.
"""

import os
import hashlib
import httpx
import pdfplumber
from typing import Optional
from loguru import logger

from config import get_settings

settings = get_settings()


def download_and_extract_pdf(pdf_url: str, paper_id: str) -> Optional[str]:
    """
    Download a PDF and extract its full text.

    Args:
        pdf_url:  Direct URL to the PDF file
        paper_id: Unique paper identifier (used to name the local cache file)

    Returns:
        Extracted text string, or None if download/parsing failed
    """
    if not pdf_url:
        return None

    # Build a safe local filename
    safe_id = hashlib.md5(paper_id.encode()).hexdigest()[:12]
    papers_dir = os.path.abspath(settings.papers_dir)
    os.makedirs(papers_dir, exist_ok=True)
    local_path = os.path.join(papers_dir, f"{safe_id}.pdf")

    # Use cached file if it exists
    if os.path.exists(local_path):
        logger.debug(f"[PDF] Using cached: {local_path}")
        return _extract_text(local_path)

    # Download
    logger.info(f"[PDF] Downloading: {pdf_url[:80]}")
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(pdf_url, headers={"User-Agent": "ResearchBot/1.0"})
            resp.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(resp.content)
        logger.success(f"[PDF] Saved to {local_path}")

    except Exception as exc:
        logger.warning(f"[PDF] Download failed ({exc}) — skipping PDF for {paper_id}")
        return None

    return _extract_text(local_path)


def _extract_text(path: str) -> Optional[str]:
    """Extract plain text from a local PDF using pdfplumber."""
    try:
        pages_text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:20]:   # cap at 20 pages for performance
                text = page.extract_text()
                if text:
                    pages_text.append(text)

        full_text = "\n\n".join(pages_text).strip()
        logger.debug(f"[PDF] Extracted {len(full_text)} chars from {os.path.basename(path)}")
        return full_text if full_text else None

    except Exception as exc:
        logger.warning(f"[PDF] Text extraction failed: {exc}")
        return None
