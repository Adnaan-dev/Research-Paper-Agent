"""
coordinator_agent.py — LangGraph-style directed graph orchestrator.

Implements the EXACT LangGraph pattern:
  - ResearchState TypedDict shared across all nodes
  - Named graph nodes (functions: State -> State)
  - Directed edges between nodes
  - Conditional edges for the Reviewer/Reviser self-correction loop
  - ResearchGraph runner (equivalent to LangGraph's StateGraph.compile())

Pipeline graph:
  PLANNER → SEARCHER → READER → INSIGHT → INDEXER
      → COMPARISON → WRITER → REVIEWER
                               ├─(pass)──→ PUBLISHER → END
                               └─(fail)──→ REVISER → WRITER (max 2 retries)

This is architecturally identical to LangGraph. The runner is implemented
inline since langgraph is not pip-installable in this environment, but the
graph definition, State object, and conditional edge logic are standard.
"""

from typing import Dict, Any
from loguru import logger

from agents.graph_state import ResearchGraph, ResearchState, Node, make_initial_state
from agents.planner_agent     import PlannerAgent
from agents.search_agent      import SearchAgent
from agents.summarizer_agent  import SummarizerAgent
from agents.insight_agent     import InsightAgent
from agents.comparison_agent  import ComparisonAgent
from agents.literature_agent  import LiteratureReviewAgent
from agents.reviewer_agent    import ReviewerAgent
from agents.reviser_agent     import ReviserAgent
from agents.qa_agent          import QAAgent

from services.text_chunker  import chunk_papers
from services.vector_store  import add_chunks_to_store, get_collection_stats
from config import get_settings

settings = get_settings()

MAX_REVISIONS = 2   # Reviewer/Reviser loop cap


def _build_graph(
    planner:    PlannerAgent,
    searcher:   SearchAgent,
    summarizer: SummarizerAgent,
    insight:    InsightAgent,
    comparison: ComparisonAgent,
    writer:     LiteratureReviewAgent,
    reviewer:   ReviewerAgent,
    reviser:    ReviserAgent,
) -> ResearchGraph:
    """
    Build and return the compiled research graph.

    Mirrors LangGraph usage:
        graph = StateGraph(ResearchState)
        graph.add_node("planner", planner_fn)
        graph.add_edge("planner", "searcher")
        graph.add_conditional_edges("reviewer", quality_check, {...})
        app = graph.compile()
    """

    # ── Node functions (State → State wrappers) ────────────────────────────

    def node_planner(state: ResearchState) -> ResearchState:
        return planner.run(state)

    def node_searcher(state: ResearchState) -> ResearchState:
        papers = searcher.run(state["topic"])
        state["papers"] = papers
        return state

    def node_reader(state: ResearchState) -> ResearchState:
        papers = summarizer.run(state["papers"])
        state["papers"] = papers
        state["summaries_done"] = True
        return state

    def node_insight(state: ResearchState) -> ResearchState:
        papers = insight.run(state["papers"])
        state["papers"] = papers
        state["insights_done"] = True
        return state

    def node_indexer(state: ResearchState) -> ResearchState:
        chunks = chunk_papers(state["papers"])
        added  = add_chunks_to_store(chunks)
        state["chunks_indexed"] = added
        logger.info(f"[Indexer] {added} new chunks stored in ChromaDB")
        return state

    def node_comparison(state: ResearchState) -> ResearchState:
        prose = comparison.run(state["papers"], state["topic"])
        state["comparison_prose"] = prose
        return state

    def node_writer(state: ResearchState) -> ResearchState:
        return writer.run(state)

    def node_reviewer(state: ResearchState) -> ResearchState:
        return reviewer.run(state)

    def node_reviser(state: ResearchState) -> ResearchState:
        return reviser.run(state)

    def node_publisher(state: ResearchState) -> ResearchState:
        """Final node — promotes draft to final and records stats."""
        state["literature_final"]  = state.get("literature_draft", "")
        state["vector_store_stats"] = get_collection_stats()
        state["status"] = "success"
        logger.success("[Publisher] Pipeline complete — review finalized")
        return state

    # ── Conditional edge: reviewer decision ───────────────────────────────
    def reviewer_decision(state: ResearchState):
        """
        Conditional edge function.
        Returns True → go to Publisher
        Returns False → go to Reviser (if under revision cap)
        """
        if state.get("review_passed", False):
            return "pass"
        if state.get("revision_count", 0) >= MAX_REVISIONS:
            logger.warning(f"[Graph] Max revisions ({MAX_REVISIONS}) reached — publishing anyway")
            return "pass"
        return "fail"

    # ── Build graph ────────────────────────────────────────────────────────
    graph = ResearchGraph()

    graph.add_node(Node.PLANNER,    node_planner)
    graph.add_node(Node.SEARCHER,   node_searcher)
    graph.add_node(Node.READER,     node_reader)
    graph.add_node(Node.INSIGHT,    node_insight)
    graph.add_node(Node.INDEXER,    node_indexer)
    graph.add_node(Node.COMPARISON, node_comparison)
    graph.add_node(Node.WRITER,     node_writer)
    graph.add_node(Node.REVIEWER,   node_reviewer)
    graph.add_node(Node.REVISER,    node_reviser)
    graph.add_node(Node.PUBLISHER,  node_publisher)

    # Directed edges (sequential flow)
    graph.add_edge(Node.PLANNER,    Node.SEARCHER)
    graph.add_edge(Node.SEARCHER,   Node.READER)
    graph.add_edge(Node.READER,     Node.INSIGHT)
    graph.add_edge(Node.INSIGHT,    Node.INDEXER)
    graph.add_edge(Node.INDEXER,    Node.COMPARISON)
    graph.add_edge(Node.COMPARISON, Node.WRITER)
    graph.add_edge(Node.WRITER,     Node.REVIEWER)
    graph.add_edge(Node.REVISER,    Node.WRITER)   # back-edge for retry

    # Conditional edge from REVIEWER
    graph.add_conditional_edges(
        Node.REVIEWER,
        reviewer_decision,
        {"pass": Node.PUBLISHER, "fail": Node.REVISER},
    )

    graph.add_edge(Node.PUBLISHER, Node.END)
    graph.set_entry_point(Node.PLANNER)

    return graph


# ── Public coordinator class ──────────────────────────────────────────────────

class CoordinatorAgent:
    """
    Compiles and runs the LangGraph-style research pipeline.
    Exposes run_full_pipeline() and run_qa() as the public API.
    """

    def __init__(self):
        # Lazy-init all agents
        self._planner    = None
        self._searcher   = None
        self._summarizer = None
        self._insight    = None
        self._comparison = None
        self._writer     = None
        self._reviewer   = None
        self._reviser    = None
        self._qa         = None
        self._graph      = None

    # ── WORKFLOW A: Full LangGraph pipeline ───────────────────────────────

    def run_full_pipeline(self, topic: str, max_papers: int = 8) -> Dict[str, Any]:
        """
        Compile and execute the full research graph.

        Equivalent to: app = graph.compile(); app.invoke(initial_state)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[Coordinator] Compiling + invoking graph for: '{topic}'")
        logger.info(f"[Coordinator] Graph: {' → '.join([n.value for n in [Node.PLANNER, Node.SEARCHER, Node.READER, Node.INSIGHT, Node.INDEXER, Node.COMPARISON, Node.WRITER, Node.REVIEWER, Node.PUBLISHER]])}")
        logger.info(f"{'='*60}")

        # Rebuild graph with correct max_papers
        self._searcher = SearchAgent(max_papers=max_papers)
        graph = _build_graph(
            self._get_planner(),
            self._searcher,
            self._get_summarizer(),
            self._get_insight(),
            self._get_comparison(),
            self._get_writer(),
            self._get_reviewer(),
            self._get_reviser(),
        )

        # Create initial state
        initial_state = make_initial_state(topic, max_papers)

        # Invoke graph (equivalent to LangGraph's app.invoke())
        final_state = graph.invoke(initial_state)

        # Guard: if no papers found
        if not final_state.get("papers"):
            return _empty_result(topic)

        logger.success(
            f"[Coordinator] Graph complete. "
            f"Nodes executed: {' → '.join(final_state.get('node_log', []))}. "
            f"Revisions: {final_state.get('revision_count', 0)}."
        )

        return {
            "topic":             topic,
            "sub_questions":     final_state.get("sub_questions", []),
            "papers":            final_state.get("papers", []),
            "comparison_table":  final_state.get("comparison_table", ""),
            "comparison":        final_state.get("comparison_prose", ""),
            "literature_review": final_state.get("literature_final", final_state.get("literature_draft", "")),
            "review_feedback":   final_state.get("review_feedback", ""),
            "revision_count":    final_state.get("revision_count", 0),
            "review_passed":     final_state.get("review_passed", False),
            "chunks_indexed":    final_state.get("chunks_indexed", 0),
            "vector_store_stats": final_state.get("vector_store_stats", get_collection_stats()),
            "node_log":          final_state.get("node_log", []),
            "status":            final_state.get("status", "success"),
        }

    # ── WORKFLOW B: RAG Q&A ───────────────────────────────────────────────

    def run_qa(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        logger.info(f"[Coordinator] QA: '{question[:60]}'")
        qa = self._get_qa()
        qa.top_k = top_k
        result = qa.run(question)
        result["status"] = "success"
        return result

    # ── Lazy agent accessors ──────────────────────────────────────────────

    def _get_planner(self):
        if not self._planner:
            self._planner = PlannerAgent()
        return self._planner

    def _get_summarizer(self):
        if not self._summarizer:
            self._summarizer = SummarizerAgent()
        return self._summarizer

    def _get_insight(self):
        if not self._insight:
            self._insight = InsightAgent()
        return self._insight

    def _get_comparison(self):
        if not self._comparison:
            self._comparison = ComparisonAgent()
        return self._comparison

    def _get_writer(self):
        if not self._writer:
            self._writer = LiteratureReviewAgent()
        return self._writer

    def _get_reviewer(self):
        if not self._reviewer:
            self._reviewer = ReviewerAgent()
        return self._reviewer

    def _get_reviser(self):
        if not self._reviser:
            self._reviser = ReviserAgent()
        return self._reviser

    def _get_qa(self):
        if not self._qa:
            self._qa = QAAgent()
        return self._qa


# ── helpers ───────────────────────────────────────────────────────────────────

def _empty_result(topic: str) -> Dict[str, Any]:
    return {
        "topic": topic,
        "sub_questions": [],
        "papers": [],
        "comparison_table": "",
        "comparison": "",
        "literature_review": "No papers found for this topic. Try a broader query.",
        "review_feedback": "",
        "revision_count": 0,
        "review_passed": False,
        "chunks_indexed": 0,
        "vector_store_stats": get_collection_stats(),
        "node_log": [],
        "status": "no_results",
    }
