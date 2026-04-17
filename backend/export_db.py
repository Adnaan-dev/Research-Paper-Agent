"""
export_db.py -- Export everything in ChromaDB to a JSON file.

HOW TO RUN:
    cd backend
    python export_db.py

OUTPUT:
    data/chroma_export.json   -- full export (all papers + chunks)
    data/chroma_summary.json  -- compact summary (papers only, no chunk text)

You can open these files in any text editor, VS Code, or browser.
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from config import get_settings

settings = get_settings()

COLLECTION_NAME = "research_papers"


def export():
    print("\n" + "="*60)
    print("  ChromaDB Export Tool")
    print("="*60)

    # Connect to the DB
    persist_dir = os.path.abspath(settings.chroma_persist_dir)
    if not os.path.exists(persist_dir):
        print(f"\nERROR: ChromaDB folder not found at: {persist_dir}")
        print("Run a search first to create and populate the database.")
        return

    client = chromadb.PersistentClient(path=persist_dir)

    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        print("\nNo 'research_papers' collection found.")
        print("Run a search first to index some papers.")
        return

    # Fetch everything
    result = col.get(include=["documents", "metadatas"])
    ids       = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    if not ids:
        print("\nVector store is empty. Run a search first.")
        return

    print(f"\nFound {len(ids)} chunks in the database.\n")

    # Group by paper
    papers = {}
    for chunk_id, doc, meta in zip(ids, documents, metadatas):
        title = meta.get("title", "Unknown Paper")
        if title not in papers:
            papers[title] = {
                "title":       title,
                "source":      meta.get("source", ""),
                "url":         meta.get("url", ""),
                "authors":     meta.get("authors", ""),
                "published":   meta.get("published", ""),
                "chunk_count": 0,
                "chunks":      [],
            }
        papers[title]["chunk_count"] += 1
        papers[title]["chunks"].append({
            "chunk_id":    chunk_id,
            "chunk_index": meta.get("chunk_index", 0),
            "text":        doc,
        })

    for p in papers.values():
        p["chunks"].sort(key=lambda c: c["chunk_index"])

    papers_list = sorted(papers.values(), key=lambda p: p["title"])

    # Print summary to terminal
    print(f"{'PAPER':<55} {'SOURCE':<18} {'YEAR':<6} {'CHUNKS'}")
    print("-" * 90)
    for p in papers_list:
        title_short = p["title"][:52] + "..." if len(p["title"]) > 52 else p["title"]
        print(f"{title_short:<55} {p['source']:<18} {p['published']:<6} {p['chunk_count']}")

    print("-" * 90)
    print(f"TOTAL: {len(papers_list)} papers, {len(ids)} chunks\n")

    # Build export objects
    export_data = {
        "exported_at":  datetime.now().isoformat(),
        "total_chunks": len(ids),
        "total_papers": len(papers_list),
        "papers":       papers_list,
    }

    summary_data = {
        "exported_at":  datetime.now().isoformat(),
        "total_chunks": len(ids),
        "total_papers": len(papers_list),
        "papers": [
            {
                "title":       p["title"],
                "source":      p["source"],
                "url":         p["url"],
                "authors":     p["authors"],
                "published":   p["published"],
                "chunk_count": p["chunk_count"],
            }
            for p in papers_list
        ],
    }

    # Save files
    os.makedirs("data", exist_ok=True)

    full_path    = os.path.abspath("data/chroma_export.json")
    summary_path = os.path.abspath("data/chroma_summary.json")

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"Full export saved to   : {full_path}")
    print(f"Summary saved to       : {summary_path}")
    print("\nOpen either file in VS Code, a browser, or any text editor.")
    print("="*60 + "\n")


if __name__ == "__main__":
    export()
