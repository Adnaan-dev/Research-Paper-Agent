# Project Document — AI Research Paper Analyzer & Literature Survey Agent

## 1. Problem Statement

Researchers and students spend **40–60 hours** compiling a literature survey for any new research topic. This bottleneck involves:
- Manually searching across arXiv, Semantic Scholar, and other databases
- Reading and extracting key findings from dozens of papers
- Writing a coherent, well-cited survey with comparison tables and identified gaps

**Our system eliminates this bottleneck** by automating the entire pipeline using a multi-agent LangGraph state machine.

---

## 2. Project Scope

| Constraint | Value |
|---|---|
| Maximum papers per run | 20 (default: 8 for reliable latency) |
| Paper sources | arXiv + Semantic Scholar |
| LLM | Groq llama-3.3-70b (free tier) or OpenAI GPT-4o-mini |
| Embeddings | all-MiniLM-L6-v2 (local, CPU-based, free) |
| Vector DB | ChromaDB (persistent on disk) |
| Self-correction loop | Max 2 Reviewer → Reviser iterations |
| Expected pipeline time | 2–4 minutes for 8 papers |

---

## 3. Features

- **Multi-source search:** arXiv API + Semantic Scholar Graph API
- **9-agent LangGraph pipeline** with directed edges and conditional transitions
- **Structured output:** comparison table, taxonomy, 6-section review, citations
- **Self-correction loop:** Reviewer validates → Reviser improves → Writer retries
- **Persistent RAG Q&A:** ChromaDB vector store survives restarts
- **Export:** Markdown download + PDF generation
- **Interactive UI:** Streamlit with live graph node progress visualization

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH PIPELINE                       │
│                                                                 │
│  PLANNER → SEARCHER → READER → INSIGHT → INDEXER               │
│       → COMPARISON → WRITER → REVIEWER                         │
│                               ├─(pass)──→ PUBLISHER → END      │
│                               └─(fail)──→ REVISER → WRITER     │
│                                          (max 2 retries)        │
└─────────────────────────────────────────────────────────────────┘
         ↑ State object (ResearchState TypedDict) flows through all nodes

┌──────────────┐    HTTP REST    ┌──────────────────────────────┐
│  Streamlit   │ ◄────────────► │  FastAPI Backend              │
│  Frontend    │                │  /search/  /query/            │
│  (port 8501) │                │  (port 8000)                  │
└──────────────┘                └──────────────────────────────┘
                                          │
                    ┌─────────────────────┼──────────────────┐
                    ▼                     ▼                  ▼
              ChromaDB              Groq LLM API         arXiv /
           (vector store)       llama-3.3-70b        Semantic Scholar
```

---

## 5. LangGraph Node Descriptions

| Node | Agent | Responsibility |
|---|---|---|
| PLANNER | `PlannerAgent` | Decomposes topic into 4–5 sub-questions |
| SEARCHER | `SearchAgent` | Fetches + deduplicates from arXiv + Semantic Scholar |
| READER | `SummarizerAgent` | Generates 3–5 sentence summary per paper |
| INSIGHT | `InsightAgent` | Extracts JSON: problem/method/datasets/models/results |
| INDEXER | *(inline)* | Chunks papers, embeds, stores in ChromaDB |
| COMPARISON | `ComparisonAgent` | 5-section cross-paper comparison |
| WRITER | `LiteratureReviewAgent` | Full 6-section review + comparison table + taxonomy |
| REVIEWER | `ReviewerAgent` | Validates draft (score/10, section check, table check) |
| REVISER | `ReviserAgent` | Incorporates feedback, rewrites draft |
| PUBLISHER | *(inline)* | Finalizes output, records stats |

**Conditional edge (Reviewer → ?):**
```python
def reviewer_decision(state):
    if state["review_passed"]:         return "pass"  → PUBLISHER
    if state["revision_count"] >= 2:   return "pass"  → PUBLISHER (cap)
    return "fail"                                      → REVISER
```

---

## 6. Team Roles

| Role | Responsibility |
|---|---|
| **Agent Orchestration** | LangGraph graph definition, ResearchState TypedDict, conditional edges, Reviewer/Reviser loop |
| **RAG Pipeline** | ChromaDB vector store, text chunking, embedding service, QAAgent retrieval |
| **Frontend** | Streamlit UI, 6-tab layout, graph node progress, PDF export, chat interface |
| **API & Backend** | FastAPI routes, async ThreadPoolExecutor, PDF generation endpoint |

---

## 7. Tech Stack

| Layer | Technology |
|---|---|
| Agent Graph | LangGraph (StateGraph pattern, ResearchState TypedDict) |
| LLM | Groq llama-3.3-70b (FREE) / OpenAI GPT-4o-mini |
| Orchestration | LangChain (prompts, chains) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (local, CPU) |
| Vector DB | ChromaDB 0.5 (persistent) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Paper APIs | arXiv official client + Semantic Scholar Graph API |

---

## 8. Expected Output

Given topic: **"Vision Transformers for Medical Imaging"**

1. **Sub-questions** (Planner): 5 targeted research questions
2. **Papers** (Searcher): 8 papers from arXiv + Semantic Scholar
3. **Summaries** (Reader): 3–5 sentence academic summary per paper
4. **Insights** (Insight): JSON with problem/method/datasets/models/results
5. **Comparison Table** (Writer):

| Paper | Year | Method | Dataset | Key Result | Limitation |
|-------|------|--------|---------|------------|------------|
| ViT-Med | 2023 | ViT + pretraining | ChestX-ray14 | AUC 0.91 | High compute |
| ... | ... | ... | ... | ... | ... |

6. **Taxonomy** (Writer): Classification into 3 approach categories
7. **Literature Review** (Writer → Reviewer → Reviser): Full 6-section markdown review with [Author, Year] citations
8. **Q&A** (QAAgent + ChromaDB): Grounded answers to follow-up questions with source URLs

---

## 9. References

- LangGraph: https://langchain-ai.github.io/langgraph/
- arXiv:2411.18241 — "LLM Multi-Agent Application with LangGraph+CrewAI"
- arXiv:2412.17481 — "Survey on LLM-based Multi-Agent Systems"
- gpt-researcher (24.8k stars): https://github.com/assafelovic/gpt-researcher
- Pinecone LangGraph Tutorial: https://www.pinecone.io/learn/langgraph/
