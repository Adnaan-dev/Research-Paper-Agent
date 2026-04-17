"""
streamlit_app.py — Research Paper Analyzer (LangGraph Edition)
Full UI with: LangGraph pipeline status, comparison table, taxonomy,
PDF export, reviewer feedback, and RAG Q&A chat.
"""

import os, time, requests, io
import streamlit as st # type: ignore

st.set_page_config(
    page_title="Research Paper Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def _extract_section(text: str, keywords: list) -> str:
    """Extract a section from markdown text by heading keyword."""
    if not text:
        return ""
    lines = text.split("\n")
    capturing, result = False, []
    for line in lines:
        line_lower = line.lower()
        if any(f"## " in line and kw in line_lower for kw in keywords):
            capturing = True
            result.append(line); continue
        if capturing:
            if line.startswith("## ") and not any(kw in line.lower() for kw in keywords):
                break
            result.append(line)
    return "\n".join(result).strip()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
.main-header { background: linear-gradient(135deg,#1a2744 0%,#0d1b2a 60%,#1a1a2e 100%);
  border:1px solid #2d4a7a; border-radius:12px; padding:1.8rem 2.2rem; margin-bottom:1.2rem; }
.main-header h1 { font-family:'IBM Plex Mono',monospace; font-size:1.75rem; font-weight:600;
  color:#58a6ff; margin:0 0 0.3rem; }
.main-header p { color:#8b949e; font-size:0.88rem; margin:0; }
.section-hdr { font-family:'IBM Plex Mono',monospace; font-size:0.75rem; font-weight:600;
  color:#58a6ff; letter-spacing:.12em; text-transform:uppercase;
  border-bottom:1px solid #21262d; padding-bottom:.4rem; margin:1.2rem 0 0.8rem; }
.paper-card { background:#161b22; border:1px solid #30363d; border-radius:10px;
  padding:1.1rem 1.3rem; margin-bottom:.8rem; }
.paper-card:hover { border-color:#58a6ff; }
.paper-title { font-size:.95rem; font-weight:600; color:#e6edf3; margin-bottom:.3rem; }
.paper-meta  { font-family:'IBM Plex Mono',monospace; font-size:.72rem; color:#8b949e; margin-bottom:.7rem; }
.paper-summary { font-size:.88rem; color:#c9d1d9; line-height:1.6; }
.src-badge { display:inline-block; background:#1f3a5f; color:#58a6ff;
  font-family:'IBM Plex Mono',monospace; font-size:.68rem; padding:2px 8px; border-radius:4px; margin-right:5px; }
.src-badge.s { background:#1a3a2a; color:#3fb950; }
.chat-user { background:#1f3a5f; border:1px solid #2d4a7a; border-radius:10px 10px 2px 10px;
  padding:.75rem 1rem; margin:.4rem 0; margin-left:12%; color:#cae3ff; font-size:.9rem; }
.chat-bot  { background:#161b22; border:1px solid #30363d; border-radius:10px 10px 10px 2px;
  padding:.75rem 1rem; margin:.4rem 0; margin-right:8%; color:#c9d1d9; font-size:.9rem; line-height:1.6; }
.src-cite  { background:#0d1117; border-left:3px solid #58a6ff; padding:.35rem .7rem;
  margin-top:.5rem; font-size:.78rem; color:#8b949e; border-radius:0 4px 4px 0; }
.metric-card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:.9rem 1rem; text-align:center; }
.metric-val { font-family:'IBM Plex Mono',monospace; font-size:1.9rem; font-weight:600; color:#58a6ff; line-height:1.1; }
.metric-lbl { font-size:.72rem; color:#8b949e; margin-top:.25rem; text-transform:uppercase; letter-spacing:.06em; }
.graph-node { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:.72rem;
  padding:3px 10px; border-radius:4px; margin:2px; }
.node-done { background:#1a3a2a; color:#3fb950; border:1px solid #2d5a3d; }
.node-pend { background:#21262d; color:#8b949e; border:1px solid #30363d; }
.review-box { background:#161b22; border:1px solid #30363d; border-radius:10px;
  padding:1.5rem 1.8rem; color:#c9d1d9; font-size:.9rem; line-height:1.75; }
.feedback-box { background:#1a1a0d; border:1px solid #3a3a1a; border-radius:8px;
  padding:1rem 1.2rem; color:#d4c896; font-size:.85rem; line-height:1.6; }
.revision-badge { background:#2a1a0d; color:#f79b1c; border:1px solid #5a3a0d;
  border-radius:4px; padding:2px 8px; font-size:.72rem; font-family:'IBM Plex Mono',monospace; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top:1.2rem; padding-bottom:1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "result": None, "search_done": False, "chat": [],
    "topic": "", "active_tab": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── API helpers ───────────────────────────────────────────────────────────────
def api_search(topic, max_papers):
    r = requests.post(f"{API_BASE}/search/", json={"topic": topic, "max_papers": max_papers}, timeout=360)
    r.raise_for_status(); return r.json()

def api_query(question, top_k=5):
    r = requests.post(f"{API_BASE}/query/", json={"question": question, "top_k": top_k}, timeout=90)
    r.raise_for_status(); return r.json()

def api_health():
    try: return requests.get(f"{API_BASE}/health", timeout=3).status_code == 200
    except: return False

def api_status():
    try: return requests.get(f"{API_BASE}/search/status", timeout=5).json()
    except: return {"total_chunks": 0}

def api_export_pdf(review_text):
    r = requests.post(f"{API_BASE}/search/export/pdf", json={"review": review_text}, timeout=60)
    r.raise_for_status(); return r.content

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='padding:.4rem 0 1rem'>
    <div style='font-family:IBM Plex Mono,monospace;font-size:1rem;font-weight:600;color:#58a6ff'>🔬 ResearchBot</div>
    <div style='font-size:.75rem;color:#8b949e'>LangGraph Multi-Agent System</div></div>""",
    unsafe_allow_html=True)

    healthy = api_health()
    col = "#3fb950" if healthy else "#f85149"
    st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:.72rem;color:{col};margin-bottom:.8rem'>● {'API Online' if healthy else 'API Offline'}</div>",
    unsafe_allow_html=True)

    st.markdown("<div class='section-hdr'>Search Settings</div>", unsafe_allow_html=True)
    topic_input = st.text_area("Research Topic",
        placeholder="e.g. Vision Transformers for Medical Imaging",
        height=85, help="Tip: be specific for better results")
    max_papers = st.slider("Max Papers", 4, 20, 8, 2)
    top_k_qa   = st.slider("RAG Chunks (Q&A)", 3, 10, 5)
    search_btn = st.button("🚀  Analyze Papers", use_container_width=True, type="primary")

    st.markdown("<div class='section-hdr'>Quick Examples</div>", unsafe_allow_html=True)
    examples = [
        "Vision Transformers for Medical Imaging",
        "Large Language Model fine-tuning RLHF",
        "Graph Neural Networks drug discovery",
        "Federated learning privacy healthcare",
        "Diffusion models image synthesis",
    ]
    for ex in examples:
        if st.button(f"↗ {ex[:38]}", key=f"ex_{ex}", use_container_width=True):
            st.session_state["_prefill"] = ex; st.rerun()

    if "_prefill" in st.session_state:
        topic_input = st.session_state.pop("_prefill")

    st.markdown("<div class='section-hdr'>Vector Store</div>", unsafe_allow_html=True)
    stats = api_status()
    st.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:.82rem;color:#8b949e'>"
        f"📦 <b style='color:#58a6ff'>{stats.get('total_chunks',0)}</b> chunks indexed</div>",
        unsafe_allow_html=True)
    if st.button("🗑 Reset Vector Store", use_container_width=True):
        try:
            requests.delete(f"{API_BASE}/search/reset", timeout=10)
            st.success("Cleared."); time.sleep(1); st.rerun()
        except Exception as e:
            st.error(str(e))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""<div class='main-header'>
<h1>🔬 Research Paper Analyzer</h1>
<p>LangGraph multi-agent pipeline &nbsp;·&nbsp; Planner → Searcher → Reader → Insight → Writer → Reviewer/Reviser → Publisher &nbsp;·&nbsp; RAG Q&A</p>
</div>""", unsafe_allow_html=True)

# ── Search execution ──────────────────────────────────────────────────────────
if search_btn:
    if not topic_input or len(topic_input.strip()) < 3:
        st.error("Please enter a research topic (at least 3 characters).")
    elif not healthy:
        st.error("Backend API is offline. Run:  cd backend && python main.py")
    else:
        prog = st.empty()

        # Show LangGraph node progress animation
        graph_nodes = ["planner","searcher","reader","insight","indexer",
                       "comparison","writer","reviewer","reviser","publisher"]
        progress_bar = st.progress(0)
        node_display = st.empty()

        def show_nodes(active_idx):
            html = ""
            for i, n in enumerate(graph_nodes):
                cls = "node-done" if i <= active_idx else "node-pend"
                html += f"<span class='graph-node {cls}'>{n}</span>"
                if i < len(graph_nodes)-1:
                    html += "<span style='color:#30363d;font-size:.8rem'> → </span>"
            node_display.markdown(
                f"<div style='font-size:.8rem;margin-bottom:.5rem'><b style='color:#58a6ff;font-family:IBM Plex Mono,monospace'>LangGraph Pipeline:</b><br>{html}</div>",
                unsafe_allow_html=True)

        # Animate through nodes while waiting
        with prog.container():
            st.markdown("<div style='background:#161b22;border:1px solid #30363d;border-radius:10px;padding:1.2rem 1.5rem'>", unsafe_allow_html=True)
            show_nodes(-1)
            step_label = st.empty()

        step_label.markdown("<div style='font-family:IBM Plex Mono,monospace;font-size:.8rem;color:#8b949e'>Invoking LangGraph pipeline...</div>", unsafe_allow_html=True)

        # Simulate node animation in parallel
        for i in range(len(graph_nodes) - 2):
            show_nodes(i)
            progress_bar.progress((i+1) / (len(graph_nodes)+1))
            time.sleep(0.35)

        try:
            result = api_search(topic_input.strip(), max_papers)
            # Show actual nodes executed
            node_log = result.get("node_log", [])
            for i, node in enumerate(node_log):
                show_nodes(graph_nodes.index(node) if node in graph_nodes else i)
            progress_bar.progress(1.0)
            show_nodes(len(graph_nodes)-1)
            step_label.markdown("<div style='font-family:IBM Plex Mono,monospace;font-size:.8rem;color:#3fb950'>✓ Pipeline complete!</div>", unsafe_allow_html=True)
            st.session_state["result"] = result
            st.session_state["topic"]  = result.get("topic", topic_input)
            st.session_state["search_done"] = True
            time.sleep(0.8)
            prog.empty(); node_display.empty(); progress_bar.empty()
            st.rerun()

        except requests.exceptions.ConnectionError:
            prog.empty(); progress_bar.empty()
            st.error("Cannot connect to backend.  cd backend && python main.py")
        except requests.exceptions.HTTPError as e:
            prog.empty(); progress_bar.empty()
            try: detail = e.response.json().get("detail","")[:1500]
            except: detail = str(e)
            st.error(f"**Pipeline error:** {detail}")
            st.info("Check your terminal for full traceback.")
        except Exception as e:
            prog.empty(); progress_bar.empty()
            st.error(f"Error: {e}")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state["search_done"] and st.session_state["result"]:
    result = st.session_state["result"]
    papers = result.get("papers", [])
    topic  = st.session_state["topic"]

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    arxiv_n   = sum(1 for p in papers if p.get("source") == "arXiv")
    scholar_n = len(papers) - arxiv_n
    rev_pass  = result.get("review_passed", False)
    revisions = result.get("revision_count", 0)
    chunks    = result.get("chunks_indexed", result.get("vector_store_stats", {}).get("total_chunks", 0))

    for col, val, lbl in [
        (c1, len(papers),    "Papers"),
        (c2, arxiv_n,        "arXiv"),
        (c3, scholar_n,      "Semantic Scholar"),
        (c4, chunks,         "Indexed Chunks"),
        (c5, revisions,      "Revisions"),
    ]:
        col.markdown(f"<div class='metric-card'><div class='metric-val'>{val}</div><div class='metric-lbl'>{lbl}</div></div>",
            unsafe_allow_html=True)

    # Reviewer status badge
    if rev_pass:
        st.markdown("<div style='color:#3fb950;font-family:IBM Plex Mono,monospace;font-size:.8rem;margin:.6rem 0'>✓ Reviewer passed the literature review</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color:#f79b1c;font-family:IBM Plex Mono,monospace;font-size:.8rem;margin:.6rem 0'>⚠ Reviewer requested {revisions} revision(s)</div>", unsafe_allow_html=True)

    # Sub-questions panel
    sub_qs = result.get("sub_questions", [])
    if sub_qs:
        with st.expander("📋 Planner Agent — Research Sub-Questions", expanded=False):
            for i, q in enumerate(sub_qs, 1):
                st.markdown(f"**{i}.** {q}")

    # Node log
    node_log = result.get("node_log", [])
    if node_log:
        html = " → ".join(
            f"<span style='font-family:IBM Plex Mono,monospace;font-size:.72rem;"
            f"background:#1a3a2a;color:#3fb950;padding:2px 8px;border-radius:4px'>{n}</span>"
            for n in node_log
        )
        st.markdown(f"<div style='margin:.4rem 0 1rem'><b style='font-size:.78rem;color:#8b949e'>Graph execution:</b> {html}</div>",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 Papers & Insights",
        "📊 Comparison Table",
        "📚 Literature Review",
        "🌳 Taxonomy",
        "🔍 Reviewer Report",
        "💬 Ask Questions (RAG)",
    ])

    # TAB 1 — Papers
    with tab1:
        st.markdown(f"<div class='section-hdr'>Papers on: {topic}</div>", unsafe_allow_html=True)
        for i, paper in enumerate(papers):
            src    = paper.get("source","")
            badge  = "s" if "Scholar" in src else ""
            authors = ", ".join(paper.get("authors",[])[:3])
            if len(paper.get("authors",[])) > 3: authors += " et al."
            insights = paper.get("insights", {})
            models   = insights.get("models_used", [])
            datasets = insights.get("datasets", [])
            st.markdown(f"""<div class='paper-card'>
<div class='paper-title'>{i+1}. {paper.get('title','')}</div>
<div class='paper-meta'>
  <span class='src-badge {badge}'>{src}</span>{authors} · {paper.get('published','N/A')}
</div>
<div class='paper-summary'>{(paper.get('summary') or paper.get('abstract',''))[:380]}...</div>
</div>""", unsafe_allow_html=True)
            with st.expander(f"🔍 Full Insights — {paper.get('title','')[:55]}..."):
                cl, cr = st.columns(2)
                with cl:
                    st.markdown("**🎯 Problem Statement**")
                    st.info(insights.get("problem_statement","N/A"))
                    st.markdown("**🛠 Methodology**")
                    st.info(insights.get("methodology","N/A"))
                with cr:
                    st.markdown("**📈 Key Results**")
                    st.success(insights.get("key_results","N/A"))
                    if models:
                        st.markdown("**🤖 Models**"); st.write(" · ".join(models))
                    if datasets:
                        st.markdown("**📂 Datasets**"); st.write(" · ".join(datasets))
                st.markdown(f"**🔗 URL:** [{paper.get('url','')}]({paper.get('url','')})")

    # TAB 2 — Comparison Table
    with tab2:
        st.markdown("<div class='section-hdr'>Structured Comparison Table</div>", unsafe_allow_html=True)
        table = result.get("comparison_table", "")
        if table and "|" in table:
            st.markdown(table)
            st.download_button("⬇ Download Table (Markdown)", data=table,
                file_name="comparison_table.md", mime="text/markdown")
        else:
            st.info("No structured table generated. Run a search to populate this tab.")
        st.markdown("<div class='section-hdr'>Comparative Analysis (Prose)</div>", unsafe_allow_html=True)
        comp = result.get("comparison","")
        if comp:
            st.markdown(f"<div class='review-box'>{comp}</div>", unsafe_allow_html=True)

    # TAB 3 — Literature Review
    with tab3:
        st.markdown("<div class='section-hdr'>Generated Literature Review</div>", unsafe_allow_html=True)
        review = result.get("literature_review","")
        if review:
            if result.get("revision_count",0) > 0:
                st.markdown(
                    f"<span class='revision-badge'>Revised {result['revision_count']}x by Reviser Agent</span>",
                    unsafe_allow_html=True)

            st.markdown(review)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button("⬇ Download as Markdown", data=review,
                    file_name=f"literature_review_{topic[:30].replace(' ','_')}.md",
                    mime="text/markdown", use_container_width=True)
            with dl2:
                if st.button("⬇ Download as PDF", use_container_width=True):
                    with st.spinner("Generating PDF..."):
                        try:
                            pdf_bytes = api_export_pdf(review)
                            st.download_button("📄 Click to download PDF", data=pdf_bytes,
                                file_name="literature_review.pdf", mime="application/pdf")
                        except Exception as e:
                            st.error(f"PDF export error: {e}")
        else:
            st.info("Run a search to generate the literature review.")

    # TAB 4 — Taxonomy
    with tab4:
        st.markdown("<div class='section-hdr'>Taxonomy of Approaches</div>", unsafe_allow_html=True)
        st.markdown("""<div style='font-size:.85rem;color:#8b949e;margin-bottom:1rem'>
        The taxonomy classifies the retrieved papers into categories based on their approach.
        This section is extracted from the generated literature review.</div>""",
        unsafe_allow_html=True)
        review = result.get("literature_review", "")
        taxonomy_text = _extract_section(review, ["taxonomy", "classification", "approaches"]) # type: ignore
        if taxonomy_text:
            st.markdown(f"<div class='review-box'>{taxonomy_text}</div>", unsafe_allow_html=True)
        else:
            st.info("Taxonomy section will appear here after generating the review.")
            st.markdown("**Manual classification based on retrieved papers:**")
            if papers:
                by_src = {}
                for p in papers:
                    src = p.get("source","Other")
                    by_src.setdefault(src,[]).append(p.get("title",""))
                for src, titles in by_src.items():
                    st.markdown(f"**{src}** ({len(titles)} papers)")
                    for t in titles:
                        st.markdown(f"- {t}")

    # TAB 5 — Reviewer Report
    with tab5:
        st.markdown("<div class='section-hdr'>Reviewer Agent Quality Report</div>", unsafe_allow_html=True)
        feedback = result.get("review_feedback","")
        rev_pass = result.get("review_passed", False)
        if feedback:
            status_color = "#3fb950" if rev_pass else "#f79b1c"
            status_text  = "PASSED" if rev_pass else "REVISION REQUESTED"
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;font-size:.85rem;"
                f"color:{status_color};margin-bottom:.8rem'>{status_text}</div>",
                unsafe_allow_html=True)
            st.markdown(f"<div class='feedback-box'>{feedback}</div>", unsafe_allow_html=True)
            if result.get("revision_count",0) > 0:
                st.markdown(
                    f"<div style='margin-top:.8rem;font-size:.82rem;color:#8b949e'>"
                    f"Reviser agent applied <b style='color:#f79b1c'>{result['revision_count']}</b> revision(s) based on this feedback.</div>",
                    unsafe_allow_html=True)
        else:
            st.info("Reviewer report will appear here after running a search.")
        with st.expander("ℹ About the Reviewer/Reviser loop"):
            st.markdown("""
**How it works (LangGraph self-correction loop):**

```
Writer → Reviewer → (score ≥ 7 AND all sections present?)
                         ├─ YES → Publisher → END
                         └─ NO  → Reviser → Writer (retry, max 2x)
```

The **Reviewer Agent** checks:
- All 6 sections present (Introduction, Background, Review, Comparison, Gaps, Conclusion)
- Inline citations in [Author, Year] format
- Structured comparison table (markdown `| columns |`)
- Taxonomy of approaches section
- Research gaps explicitly listed
- References section with URLs

The **Reviser Agent** incorporates the feedback and rewrites the draft, specifically addressing missing elements.
            """)

    # TAB 6 — RAG Q&A
    with tab6:
        st.markdown("<div class='section-hdr'>Ask Questions About the Papers (RAG)</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:.82rem;color:#8b949e;margin-bottom:.8rem'>Answers are retrieved from the ChromaDB vector store and grounded in actual paper content.</div>",
            unsafe_allow_html=True)

        for msg in st.session_state["chat"]:
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-user'>👤 {msg['content']}</div>", unsafe_allow_html=True)
            else:
                srcs = "".join(
                    f"<div>• <a href='{s.get('url', '')}' target='_blank' style='color:#58a6ff'>"
                    f"{s.get('title','')[:65]}</a> ({s.get('source','')})</div>"
                    for s in msg.get("sources",[])[:3]
                )
                src_block = f"<div class='src-cite'><b>Sources:</b>{srcs}</div>" if srcs else ""
                st.markdown(f"<div class='chat-bot'>🔬 {msg['content']}{src_block}</div>", unsafe_allow_html=True)

        with st.form("qa_form", clear_on_submit=True):
            question = st.text_input("Your question", placeholder="What methods are used? Which model performed best?", label_visibility="collapsed")
            a_col, c_col = st.columns([5,1])
            ask   = a_col.form_submit_button("Ask ↵", use_container_width=True, type="primary")
            clear = c_col.form_submit_button("Clear", use_container_width=True)

        if clear:
            st.session_state["chat"] = []; st.rerun()

        if ask and question.strip():
            st.session_state["chat"].append({"role":"user","content":question.strip()})
            with st.spinner("Retrieving from ChromaDB and generating answer..."):
                try:
                    r = api_query(question.strip(), top_k_qa)
                    st.session_state["chat"].append({
                        "role":"assistant", "content":r.get("answer","No answer."),
                        "sources":r.get("sources",[])})
                except Exception as e:
                    st.session_state["chat"].append({"role":"assistant","content":f"Error: {e}","sources":[]})
            st.rerun()

        if not st.session_state["chat"]:
            st.markdown("<div style='font-size:.8rem;color:#8b949e;margin-top:.8rem'>💡 Suggested questions:</div>", unsafe_allow_html=True)
            suggestions = [
                f"What are the main contributions of papers on {topic}?",
                "Which model or method achieved the best results?",
                "What datasets are most commonly used?",
                "What are the key open research challenges?",
                "How do the methodologies differ across papers?",
            ]
            cols = st.columns(2)
            for j, sug in enumerate(suggestions):
                if cols[j%2].button(sug[:72], key=f"sug{j}", use_container_width=True):
                    st.session_state["chat"].append({"role":"user","content":sug})
                    with st.spinner("Thinking..."):
                        try:
                            r = api_query(sug, top_k_qa)
                            st.session_state["chat"].append({
                                "role":"assistant","content":r.get("answer",""),
                                "sources":r.get("sources",[])})
                        except Exception as e:
                            st.session_state["chat"].append({"role":"assistant","content":f"Error: {e}","sources":[]})
                    st.rerun()

# ── Empty state ───────────────────────────────────────────────────────────────
elif not st.session_state["search_done"]:
    st.markdown("""<div style='text-align:center;padding:3.5rem 1rem;color:#8b949e'>
<div style='font-size:2.8rem;margin-bottom:.8rem'>🔬</div>
<div style='font-family:IBM Plex Mono,monospace;font-size:1rem;color:#c9d1d9;margin-bottom:.7rem'>Enter a research topic to begin</div>
<div style='font-size:.88rem;max-width:520px;margin:0 auto;line-height:1.7'>
The <b style='color:#58a6ff'>LangGraph</b> pipeline will autonomously plan, search, read, analyze,
write, review, revise, and publish a complete literature survey.
</div>
<div style='margin-top:2rem;display:flex;justify-content:center;gap:1rem;flex-wrap:wrap'>
<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.9rem 1.2rem;font-size:.8rem'>🗺 <b>Planner</b><br><span style='color:#8b949e'>Decomposes topic</span></div>
<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.9rem 1.2rem;font-size:.8rem'>🔍 <b>Searcher</b><br><span style='color:#8b949e'>arXiv + Semantic Scholar</span></div>
<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.9rem 1.2rem;font-size:.8rem'>📖 <b>Reader</b><br><span style='color:#8b949e'>Summarizes papers</span></div>
<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.9rem 1.2rem;font-size:.8rem'>🧠 <b>Insight</b><br><span style='color:#8b949e'>Extracts JSON</span></div>
<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.9rem 1.2rem;font-size:.8rem'>✍ <b>Writer</b><br><span style='color:#8b949e'>Literature review</span></div>
<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:.9rem 1.2rem;font-size:.8rem'>🔎 <b>Reviewer/Reviser</b><br><span style='color:#8b949e'>Self-correction loop</span></div>
</div></div>""", unsafe_allow_html=True)



