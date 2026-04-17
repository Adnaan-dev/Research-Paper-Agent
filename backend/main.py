"""
main.py — FastAPI application entry point.

Registers all routers, configures CORS, sets up logging,
and provides a health-check endpoint.
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(__file__))

# ── Load .env and bust settings cache BEFORE any other local imports ──────────
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)

from config import get_settings
get_settings.cache_clear()

# ── Now safe to import routes (which trigger all agent/service imports) ───────
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from routes.search_route import router as search_router
from routes.query_route import router as query_router

settings = get_settings()

# ── Ensure data dirs exist before logging tries to write ──────────────────────
os.makedirs("data", exist_ok=True)
os.makedirs("data/papers", exist_ok=True)
os.makedirs("data/chroma_db", exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    backtrace=True,
    diagnose=True,
)
logger.add(
    "data/app.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    backtrace=True,
    diagnose=True,
)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Research Paper Analyzer API",
    description=(
        "Multi-agent system for autonomous research paper analysis, "
        "literature survey generation, and RAG-based Q&A."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit (and any local dev client) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(search_router)
app.include_router(query_router)


# ── Global exception handler — returns full traceback in detail ───────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception on {request.url}:\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "traceback": tb},
    )

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "running",
        "app": "Research Paper Analyzer",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Lightweight health probe (no DB hit)."""
    return {"status": "healthy"}


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
