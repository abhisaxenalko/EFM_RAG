from __future__ import annotations

import asyncio
import html
import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from app.cache.semantic_cache import SemanticCache
from app.config import Settings
from app.graph.state import CragState
from app.llm import LLMClient
from app.models.schemas import Answer, GradeOutput, Usage, Verdict
from app.retrieval.hybrid_search import Candidate


@dataclass(frozen=True)
class NodeDeps:
    llm: LLMClient
    searcher: Callable[..., Any]
    reranker: Callable[..., Any]
    settings: Settings
    cache: SemanticCache | None = None
    embedder: Callable[[list[str]], list[list[float]]] | None = None


def _update(state: CragState, node: str, **updates: Any) -> dict[str, Any]:
    return {**updates, "trace": [*state["trace"], node]}


def _prompt(name: str) -> str:
    return (Path(__file__).parent / "prompts" / f"{name}.md").read_text(
        encoding="utf-8"
    )


async def _invoke(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    result = await asyncio.to_thread(function, *args, **kwargs)
    return await result if inspect.isawaitable(result) else result


def _context(docs: Sequence[Candidate]) -> str:
    chunks = [
        f'<chunk doc_name="{html.escape(doc.doc_name, quote=True)}" '
        f'page="{doc.page}">{html.escape(doc.child_text)}</chunk>'
        for doc in docs
    ]
    return "<retrieved_context>\n" + "\n".join(chunks) + "\n</retrieved_context>"


def _usage_update(state: CragState, usage: Usage) -> list[Usage]:
    return [*state["usage_log"], usage]


async def retrieve(state: CragState, deps: NodeDeps) -> dict[str, Any]:
    query = state["rewritten_query"] or state["query"]
    candidates = await _invoke(deps.searcher, state["tenant_id"], state["roles"], query)
    docs = await _invoke(deps.reranker, query, candidates, top_n=deps.settings.k_final)
    return _update(state, "retrieve", docs=list(docs))


async def _grade_in_band(
    state: CragState,
    deps: NodeDeps,
    score: float,
) -> tuple[GradeOutput, list[Usage]]:
    system = _prompt("grader")
    user = json.dumps(
        {
            "query": state["rewritten_query"] or state["query"],
            "reranker_score": score,
            "GRADE_LOW": deps.settings.grade_low,
            "GRADE_HIGH": deps.settings.grade_high,
            "retrieved_context": _context(state["docs"]),
        }
    )
    usage_log: list[Usage] = []
    for _ in range(2):
        usage: Usage | None = None
        try:
            response, usage = await deps.llm.complete_structured(
                system=system, user=user, schema=GradeOutput
            )
        except (ValidationError, TypeError, ValueError):
            usage_log.append(Usage())
            continue
        usage_log.append(usage)
        result = response.model_dump() if isinstance(response, BaseModel) else response
        try:
            return GradeOutput.model_validate(result), usage_log
        except (ValidationError, TypeError, ValueError):
            continue
    return GradeOutput(score="ambiguous"), usage_log


async def grade_documents(state: CragState, deps: NodeDeps) -> dict[str, Any]:
    low, high = deps.settings.grade_low, deps.settings.grade_high
    if low >= high:
        raise ValueError("GRADE_LOW must be less than GRADE_HIGH")
    scores = [doc.rerank_score for doc in state["docs"] if doc.rerank_score is not None]
    if not scores:
        grade, usage = GradeOutput(score="irrelevant"), []
    elif max(scores) >= high:
        grade, usage = GradeOutput(score="relevant"), []
    elif max(scores) <= low:
        grade, usage = GradeOutput(score="irrelevant"), []
    else:
        grade, usage = await _grade_in_band(state, deps, max(scores))
    return _update(
        state,
        "grade_documents",
        grade=grade,
        missing_aspect=grade.missing_aspect,
        usage_log=[*state["usage_log"], *usage],
    )


async def rewrite_query(state: CragState, deps: NodeDeps) -> dict[str, Any]:
    user = json.dumps(
        {
            "query": state["query"],
            "missing_aspect": state["missing_aspect"],
            "retry_count": state["retry_count"],
        }
    )
    text, usage = await deps.llm.complete_text(system=_prompt("rewriter"), user=user)
    try:
        payload = json.loads(text)
        rewritten = payload["rewritten_query"]
        if not isinstance(rewritten, str) or not rewritten.strip():
            raise ValueError("rewritten_query must be a non-empty string")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        rewritten = None
    return _update(
        state,
        "rewrite_query",
        rewritten_query=rewritten,
        retry_count=state["retry_count"] + 1,
        usage_log=_usage_update(state, usage),
    )


async def generate(state: CragState, deps: NodeDeps) -> dict[str, Any]:
    user = json.dumps(
        {
            "query": state["rewritten_query"] or state["query"],
            "missing_aspect": state["missing_aspect"],
            "retrieved_context": _context(state["docs"]),
        }
    )
    response, usage = await deps.llm.complete_structured(
        system=_prompt("generator"), user=user, schema=Answer
    )
    answer = Answer.model_validate(
        response.model_dump() if isinstance(response, BaseModel) else response
    )
    return _update(
        state,
        "generate",
        answer=answer,
        citations=answer.citations,
        usage_log=_usage_update(state, usage),
    )


async def guardrail_verify(state: CragState, deps: NodeDeps) -> dict[str, Any]:
    answer = state["answer"] or Answer()
    available = {(doc.doc_name, doc.page) for doc in state["docs"]}
    invalid = [
        citation
        for citation in answer.citations
        if (citation.doc_name, citation.page) not in available
    ]
    if invalid:
        missing = [
            f"Citation not retrieved: {c.doc_name}, Page {c.page}"
            for c in invalid
        ]
        verdict = Verdict(supported=False, unsupported_claims=missing)
        return _update(state, "guardrail_verify", verified=verdict)
    verdict, usage = await deps.llm.complete_structured(
        system=_prompt("verifier"),
        user=json.dumps(
            {
                "query": state["rewritten_query"] or state["query"],
                "answer": answer.model_dump(),
                "retrieved_context": _context(state["docs"]),
            }
        ),
        schema=Verdict,
    )
    verdict = Verdict.model_validate(
        verdict.model_dump() if isinstance(verdict, BaseModel) else verdict
    )
    return _update(
        state,
        "guardrail_verify",
        verified=verdict,
        usage_log=_usage_update(state, usage),
    )


async def abstain(state: CragState, deps: NodeDeps) -> dict[str, Any]:
    message = (
        "Insufficient verified documentation to answer safely. "
        "Please escalate to a human engineer."
    )
    names = list(dict.fromkeys(doc.doc_name for doc in state["docs"]))[:3]
    closest = ", ".join(names) if names else "none"
    answer = Answer(text=f"{message}\nClosest documents: {closest}.")
    return _update(state, "abstain", answer=answer, citations=[], abstained=True)


async def cache_store(state: CragState, deps: NodeDeps) -> dict[str, Any]:
    cache = getattr(deps, "cache", None)
    embedder = getattr(deps, "embedder", None)
    verdict, answer = state["verified"], state["answer"]
    if cache is None or embedder is None or verdict is None or not verdict.supported:
        return _update(state, "cache_store")
    if state["abstained"] or answer is None:
        return _update(state, "cache_store")
    query = state["rewritten_query"] or state["query"]
    vectors = await _invoke(embedder, [query])
    if vectors:
        await _invoke(
            cache.store,
            state["tenant_id"],
            state["roles"],
            query,
            vectors[0],
            answer,
        )
    return _update(state, "cache_store")
