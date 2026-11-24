from __future__ import annotations

from functools import partial
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    NodeDeps,
    abstain,
    generate,
    grade_documents,
    guardrail_verify,
    retrieve,
    rewrite_query,
)
from app.graph.state import CragState

RECURSION_LIMIT = 15


async def cache_check(state: CragState, deps: NodeDeps) -> dict[str, Any]:
    """Temporary cache node; caching is not wired yet."""
    del deps
    return {"cache_hit": False, "trace": [*state["trace"], "cache_check"]}


async def prepare_regeneration(
    state: CragState,
    deps: NodeDeps,
) -> dict[str, Any]:
    """Carry verifier feedback into the next generation attempt."""
    del deps
    claims = state["verified"].unsupported_claims if state["verified"] else []
    return {
        "regen_count": state["regen_count"] + 1,
        "missing_aspect": "; ".join(claims),
        "trace": [*state["trace"], "prepare_regeneration"],
    }


def _grade_route(state: CragState, deps: NodeDeps) -> Literal[
    "generate", "rewrite_query", "abstain"
]:
    grade = state["grade"]
    if grade is not None and grade.score == "relevant":
        return "generate"
    if (
        grade is not None
        and grade.score in {"ambiguous", "irrelevant"}
        and state["retry_count"] < deps.settings.max_rewrites
    ):
        return "rewrite_query"
    return "abstain"


def _verification_route(state: CragState, deps: NodeDeps) -> Literal[
    "end", "prepare_regeneration", "abstain"
]:
    if state["verified"] is not None and state["verified"].supported:
        return "end"
    if state["regen_count"] < deps.settings.max_regenerations:
        return "prepare_regeneration"
    return "abstain"


def build_graph(deps: NodeDeps) -> Any:
    """Build and compile the CRAG state graph for the supplied dependencies."""
    builder = StateGraph(CragState)
    builder.add_node("cache_check", partial(cache_check, deps=deps))
    builder.add_node("retrieve", partial(retrieve, deps=deps))
    builder.add_node("grade_documents", partial(grade_documents, deps=deps))
    builder.add_node("rewrite_query", partial(rewrite_query, deps=deps))
    builder.add_node("generate", partial(generate, deps=deps))
    builder.add_node("guardrail_verify", partial(guardrail_verify, deps=deps))
    builder.add_node("prepare_regeneration", partial(prepare_regeneration, deps=deps))
    builder.add_node("abstain", partial(abstain, deps=deps))
    builder.add_edge(START, "cache_check")
    builder.add_edge("cache_check", "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        lambda state: _grade_route(state, deps),
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "abstain": "abstain",
        },
    )
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("generate", "guardrail_verify")
    builder.add_conditional_edges(
        "guardrail_verify",
        lambda state: _verification_route(state, deps),
        {
            "end": END,
            "prepare_regeneration": "prepare_regeneration",
            "abstain": "abstain",
        },
    )
    builder.add_edge("prepare_regeneration", "generate")
    builder.add_edge("abstain", END)
    return builder.compile().with_config({"recursion_limit": RECURSION_LIMIT})


def get_graph(deps: NodeDeps) -> Any:
    """Return the drawable graph representation for Mermaid rendering."""
    return build_graph(deps).get_graph()
