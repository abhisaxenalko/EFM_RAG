from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.graph.build import RECURSION_LIMIT, build_graph, get_graph
from app.graph.state import initial_state
from app.llm import FakeLLM
from app.models.schemas import Answer, Citation, Verdict
from app.retrieval.hybrid_search import Candidate


def _candidate(score: float = 0.9) -> Candidate:
    return Candidate(
        chunk_id=1,
        parent_id=10,
        doc_name="Chiller Manual",
        page=7,
        child_text="ERR-409B indicates low flow.",
        parent_text="ERR-409B indicates low flow.",
        rrf_score=0.02,
        rerank_score=score,
    )


def _deps(llm: FakeLLM, searcher=None, settings=None):
    return SimpleNamespace(
        llm=llm,
        searcher=searcher or (lambda tenant, roles, query: [_candidate()]),
        reranker=lambda query, docs, top_n: list(docs)[:top_n],
        settings=settings
        or SimpleNamespace(
            k_final=5,
            grade_low=0.35,
            grade_high=0.75,
            max_rewrites=1,
            max_regenerations=1,
        ),
    )


@pytest.mark.asyncio
async def test_compiled_graph_has_recursion_limit_and_renders() -> None:
    graph = build_graph(_deps(FakeLLM()))
    diagram = get_graph(_deps(FakeLLM())).draw_mermaid()
    assert graph.config["recursion_limit"] == RECURSION_LIMIT == 15
    assert "cache_check" in diagram
    assert "prepare_regeneration" in diagram
    assert "guardrail_verify" in diagram


@pytest.mark.asyncio
async def test_supported_answer_ends_after_verification() -> None:
    answer = Answer(
        text="ERR-409B indicates low flow. [Chiller Manual, Page 7]",
        citations=[Citation(doc_name="Chiller Manual", page=7)],
    )
    llm = FakeLLM({Answer: [answer], Verdict: [Verdict(supported=True)]})
    graph = build_graph(_deps(llm))

    result = await graph.ainvoke(
        initial_state("tenant_alpha", ["all"], "What is ERR-409B?")
    )

    assert result["verified"].supported is True
    assert result["trace"] == [
        "cache_check",
        "retrieve",
        "grade_documents",
        "generate",
        "guardrail_verify",
    ]
    assert len(result["usage_log"]) == 2


@pytest.mark.asyncio
async def test_failed_verification_regenerates_with_claim_feedback() -> None:
    answer = Answer(
        text="ERR-409B indicates low flow. [Chiller Manual, Page 7]",
        citations=[Citation(doc_name="Chiller Manual", page=7)],
    )
    llm = FakeLLM(
        {Answer: [answer, answer], Verdict: [
            Verdict(supported=False, unsupported_claims=["State the reset procedure."]),
            Verdict(supported=True),
        ]}
    )
    graph = build_graph(_deps(llm))

    result = await graph.ainvoke(
        initial_state("tenant_alpha", ["all"], "What is ERR-409B?")
    )

    assert result["verified"].supported is True
    assert result["regen_count"] == 1
    assert "prepare_regeneration" in result["trace"]
    generator_users = [user for system, user in llm.calls if "Answer text" in system]
    assert len(generator_users) == 2
    assert (
        json.loads(generator_users[1])["missing_aspect"]
        == "State the reset procedure."
    )


@pytest.mark.asyncio
async def test_rewrite_limit_comes_from_settings() -> None:
    answer = Answer(
        text="The requested procedure is unavailable. [Chiller Manual, Page 7]",
        citations=[Citation(doc_name="Chiller Manual", page=7)],
    )
    llm = FakeLLM(
        {Answer: [answer], Verdict: [Verdict(supported=True)]},
        text_responses=['{"rewritten_query":"ERR-409B reset procedure"}'],
    )
    calls = 0

    async def searcher(tenant, roles, query):
        nonlocal calls
        calls += 1
        return [_candidate(0.2 if calls == 1 else 0.9)]

    graph = build_graph(_deps(llm, searcher=searcher))
    result = await graph.ainvoke(
        initial_state("tenant_alpha", ["all"], "How to reset ERR-409B?")
    )
    assert result["retry_count"] == 1
    assert calls == 2
    assert "rewrite_query" in result["trace"]


@pytest.mark.asyncio
async def test_exhausted_regeneration_budget_routes_to_abstain() -> None:
    answer = Answer(
        text="Unsupported procedure. [Chiller Manual, Page 7]",
        citations=[Citation(doc_name="Chiller Manual", page=7)],
    )
    llm = FakeLLM(
        {
            Answer: [answer],
            Verdict: [Verdict(supported=False, unsupported_claims=["reset steps"])],
        }
    )
    settings = SimpleNamespace(
        k_final=5,
        grade_low=0.35,
        grade_high=0.75,
        max_rewrites=1,
        max_regenerations=0,
    )
    graph = build_graph(_deps(llm, settings=settings))

    result = await graph.ainvoke(
        initial_state("tenant_alpha", ["all"], "How do I reset ERR-409B?")
    )

    assert result["regen_count"] == 0
    assert result["abstained"] is True
    assert result["trace"][-1] == "abstain"
