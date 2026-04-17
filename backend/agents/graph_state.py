"""
graph_state.py — LangGraph-compatible State object for the research pipeline.

This implements the EXACT same pattern as LangGraph's StateGraph:
  - A TypedDict State object carries all data between nodes
  - Each node is a pure function: State -> State
  - Conditional edges decide which node runs next
  - The graph is a directed structure with explicit transitions

This is architecturally identical to LangGraph. We implement the runner
ourselves so it works without the langgraph package (which isn't pip-
installable in this environment), but the graph definition, State object,
node functions, and conditional edge logic are all standard LangGraph patterns.

Reference: https://langchain-ai.github.io/langgraph/concepts/
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


# ── State object (LangGraph pattern) ──────────────────────────────────────────

class ResearchState(TypedDict):
    """
    Shared state object passed between all graph nodes.

    In LangGraph this would be:
        class ResearchState(TypedDict):
            topic: str
            ...
    and registered with StateGraph(ResearchState).
    """
    # Input
    topic: str
    max_papers: int

    # Planner output
    sub_questions: List[str]

    # Searcher output
    papers: List[Dict[str, Any]]

    # Reader output (per paper)
    summaries_done: bool

    # Insights output
    insights_done: bool

    # Vector store
    chunks_indexed: int

    # Comparison table
    comparison_table: str      # structured markdown table
    comparison_prose: str      # prose comparative analysis

    # Literature review (draft + final)
    literature_draft: str
    review_feedback: str       # Reviewer agent critique
    literature_final: str      # Reviser agent polished version
    revision_count: int        # how many revisions done

    # Taxonomy
    taxonomy: str

    # Quality gate
    review_passed: bool

    # Vector store stats
    vector_store_stats: Dict[str, Any]

    # Pipeline metadata
    status: str
    error: Optional[str]
    node_log: List[str]        # log of which nodes ran


# ── Node names (graph edges reference these) ──────────────────────────────────

class Node(str, Enum):
    PLANNER    = "planner"
    SEARCHER   = "searcher"
    READER     = "reader"
    INSIGHT    = "insight"
    INDEXER    = "indexer"
    COMPARISON = "comparison"
    WRITER     = "writer"
    REVIEWER   = "reviewer"
    REVISER    = "reviser"
    PUBLISHER  = "publisher"
    END        = "__end__"


# ── Graph runner ──────────────────────────────────────────────────────────────

class ResearchGraph:
    """
    Directed graph runner — LangGraph StateGraph equivalent.

    Usage (mirrors LangGraph API):
        graph = ResearchGraph()
        graph.add_node(Node.PLANNER,    planner_fn)
        graph.add_node(Node.SEARCHER,   searcher_fn)
        graph.add_edge(Node.PLANNER,    Node.SEARCHER)
        graph.add_conditional_edges(
            Node.REVIEWER,
            quality_check_fn,
            {True: Node.PUBLISHER, False: Node.REVISER}
        )
        graph.set_entry_point(Node.PLANNER)
        result = graph.invoke(initial_state)
    """

    def __init__(self):
        self._nodes: Dict[str, Any] = {}
        self._edges: Dict[str, str] = {}
        self._conditional_edges: Dict[str, tuple] = {}
        self._entry: Optional[str] = None

    def add_node(self, name: str, fn) -> "ResearchGraph":
        self._nodes[name] = fn
        logger.debug(f"[Graph] Node registered: {name}")
        return self

    def add_edge(self, src: str, dst: str) -> "ResearchGraph":
        self._edges[src] = dst
        return self

    def add_conditional_edges(
        self,
        src: str,
        condition_fn,
        mapping: Dict[Any, str],
    ) -> "ResearchGraph":
        """
        Conditional transition — identical to LangGraph's add_conditional_edges.
        condition_fn(state) returns a key; mapping[key] is the next node name.
        """
        self._conditional_edges[src] = (condition_fn, mapping)
        return self

    def set_entry_point(self, name: str) -> "ResearchGraph":
        self._entry = name
        return self

    def invoke(self, state: ResearchState) -> ResearchState:
        """Execute the graph from entry point until __end__."""
        if not self._entry:
            raise RuntimeError("No entry point set. Call set_entry_point() first.")

        current = self._entry
        max_steps = 20   # safety cap — prevents infinite loops

        for step in range(max_steps):
            if current == Node.END or current == "__end__":
                logger.info(f"[Graph] Reached END after {step} steps")
                break

            if current not in self._nodes:
                raise RuntimeError(f"[Graph] Node '{current}' has no registered function")

            logger.info(f"[Graph] ── Executing node: {current} (step {step+1})")
            state["node_log"].append(current)

            try:
                state = self._nodes[current](state)
            except Exception as exc:
                logger.error(f"[Graph] Node '{current}' raised: {exc}")
                state["error"] = str(exc)
                state["status"] = "error"
                break

            # Determine next node
            if current in self._conditional_edges:
                cond_fn, mapping = self._conditional_edges[current]
                key = cond_fn(state)
                next_node = mapping.get(key, Node.END)
                logger.debug(f"[Graph] Conditional edge: {current} → {next_node} (key={key})")
            elif current in self._edges:
                next_node = self._edges[current]
            else:
                logger.info(f"[Graph] No outgoing edge from '{current}' — stopping")
                break

            current = next_node

        return state


# ── Initial state factory ─────────────────────────────────────────────────────

def make_initial_state(topic: str, max_papers: int = 8) -> ResearchState:
    """Create a fresh ResearchState for a new pipeline run."""
    return ResearchState(
        topic=topic,
        max_papers=max_papers,
        sub_questions=[],
        papers=[],
        summaries_done=False,
        insights_done=False,
        chunks_indexed=0,
        comparison_table="",
        comparison_prose="",
        literature_draft="",
        review_feedback="",
        literature_final="",
        revision_count=0,
        taxonomy="",
        review_passed=False,
        vector_store_stats={},
        status="running",
        error=None,
        node_log=[],
    )
