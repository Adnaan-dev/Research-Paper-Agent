# 🔬 Autonomous AI Research Paper Analyzer
### Multi-Agent Literature Survey System

A production-ready, multi-agent AI system that autonomously searches, analyzes, and synthesizes research papers from arXiv and Semantic Scholar — complete with a structured literature review generator and RAG-based Q&A interface.

---

## 🏗 Architecture

```
research-paper-agent/
├── backend/
│   ├── main.py                    # FastAPI entry point
│   ├── config.py                  # Pydantic settings
│   ├── agents/
│   │   ├── coordinator_agent.py   # Master orchestrator
│   │   ├── search_agent.py        # arXiv + Semantic Scholar fetcher
│   │   ├── summarizer_agent.py    # GPT-powered summarization
│   │   ├── insight_agent.py       # Structured insight extraction
│   │   ├── comparison_agent.py    # Cross-paper comparison
│   │   ├── literature_agent.py    # Literature review generator
│   │   └── qa_agent.py            # RAG-based Q&A
│   ├── services/
│   │   ├── arxiv_service.py       # arXiv API client
│   │   ├── semantic_scholar_service.py  # S2 API client
│   │   ├── embedding_service.py   # OpenAI embeddings
│   │   ├── vector_store.py        # ChromaDB persistence
│   │   ├── text_chunker.py        # Document chunking
│   │   └── pdf_loader.py          # PDF download + extraction
│   └── routes/
│       ├── search_route.py        # POST /search
│       └── query_route.py         # POST /query
├── frontend/
│   └── streamlit_app.py           # Streamlit UI
├── data/
│   ├── papers/                    # Downloaded PDFs (auto-created)
│   └── chroma_db/                 # Vector store (auto-created)
├── requirements.txt
└── README.md
```

---

## 🤖 Multi-Agent Pipeline

```
User Input (topic)
        │
        ▼
┌─────────────────┐
│ CoordinatorAgent│  ← Master orchestrator
└────────┬────────┘
         │
    ┌────┴─────────────────────────────────────────┐
    │                                              │
    ▼                                              ▼
┌──────────┐   papers    ┌─────────────┐    ┌──────────────┐
│  Search  │────────────▶│ Summarizer  │───▶│    Insight   │
│  Agent   │             │   Agent     │    │    Agent     │
└──────────┘             └─────────────┘    └──────┬───────┘
arXiv + S2                                         │
                                                   ▼
                                          ┌─────────────────┐
                                          │  ChromaDB Index  │
                                          │  (VectorStore)   │
                                          └────────┬────────┘
                                                   │
                              ┌────────────────────┴──────────┐
                              │                               │
                              ▼                               ▼
                     ┌─────────────────┐           ┌──────────────────┐
                     │   Comparison    │           │    Literature    │
                     │     Agent       │──────────▶│  Review Agent   │
                     └─────────────────┘           └──────────────────┘

Q&A Mode:
  User Question → QAAgent → ChromaDB similarity search → GPT answer + citations
```

---

## ⚙️ Setup Instructions

### 1. Clone / Download the project

```bash
git clone <repo-url>
cd research-paper-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cd backend
cp .env.example .env
```

Open `.env` and fill in:

```env
OPENAI_API_KEY=sk-your-key-here
SEMANTIC_SCHOLAR_API_KEY=your-key-here   # optional but recommended
MAX_PAPERS=10
```

**Getting API keys:**
- **OpenAI**: https://platform.openai.com/api-keys
- **Semantic Scholar** (free): https://www.semanticscholar.org/product/api

---

## 🚀 Running Locally

### Start the FastAPI backend

```bash
cd backend
python main.py
```

API will be live at: http://localhost:8000  
Swagger docs: http://localhost:8000/docs

### Start the Streamlit frontend (new terminal)

```bash
cd frontend
streamlit run streamlit_app.py
```

UI will be live at: http://localhost:8501

---

## 📖 How to Use

### Via the Streamlit UI

1. Open http://localhost:8501
2. Enter a research topic in the sidebar (e.g. "attention mechanism transformers")
3. Set Max Papers (4–20) — more papers = richer review but slower
4. Click **🚀 Analyze Papers**
5. Navigate tabs:
   - **Papers & Insights** — individual paper cards with extracted insights
   - **Comparative Analysis** — cross-paper comparison
   - **Literature Review** — full downloadable survey
   - **Ask Questions (RAG)** — chat with your papers

### Via the REST API

```bash
# Search and analyze papers
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{"topic": "BERT fine-tuning NLP", "max_papers": 6}'

# Ask a question (requires prior /search call)
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What datasets are used in these papers?", "top_k": 5}'

# Check vector store stats
curl http://localhost:8000/search/status

# Reset vector store
curl -X DELETE http://localhost:8000/search/reset
```

---

## 💡 Example Queries

**Research Topics:**
- `"transformer models for natural language processing"`
- `"diffusion models image synthesis stable diffusion"`
- `"federated learning data privacy healthcare"`
- `"graph neural networks molecular property prediction"`
- `"vision transformer ViT image classification"`

**Q&A Questions:**
- `"What are the main contributions of each paper?"`
- `"Which model achieved the best performance and on what dataset?"`
- `"What are the common limitations mentioned across these papers?"`
- `"How does the methodology in paper 1 compare to paper 3?"`
- `"What future research directions are suggested?"`

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | Your OpenAI API key |
| `SEMANTIC_SCHOLAR_API_KEY` | optional | Increases S2 rate limits |
| `MAX_PAPERS` | 10 | Papers per pipeline run |
| `OPENAI_MODEL` | gpt-4o-mini | LLM for all agents |
| `CHROMA_PERSIST_DIR` | ./data/chroma_db | Vector DB location |
| `API_PORT` | 8000 | FastAPI port |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o-mini |
| Orchestration | LangChain (chains, prompts, retrievers) |
| Vector DB | ChromaDB (persistent) |
| Embeddings | OpenAI text-embedding-3-small |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Paper APIs | arXiv, Semantic Scholar |
| PDF Parsing | pdfplumber |

---

## � Deployment

### Backend (Vercel)

1. Push your code to a GitHub repository.

2. Go to [Vercel](https://vercel.com) and sign in with GitHub.

3. Click "New Project" and import your repository.

4. Configure environment variables in Vercel dashboard:
   - `OPENAI_API_KEY`
   - `SEMANTIC_SCHOLAR_API_KEY` (optional)
   - `MAX_PAPERS` (default 10)

5. Deploy. The API will be available at `https://your-project.vercel.app`

**Note:** Vercel uses serverless functions, so the vector store is in-memory and data does not persist across requests. For production, consider using a cloud vector database like Pinecone.

### Frontend (Streamlit Cloud)

1. Go to [Streamlit Cloud](https://share.streamlit.io).

2. Connect your GitHub repository.

3. Set the main file to `frontend/streamlit_app.py`.

4. Add environment variable `API_BASE_URL` to the deployed Vercel backend URL (e.g., `https://your-project.vercel.app`).

5. Deploy.

The UI will be available on Streamlit Cloud.

---

## �📝 License

MIT License — free for academic and commercial use.
