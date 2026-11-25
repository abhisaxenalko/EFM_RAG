from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing
from typing import Any

import psycopg
import redis
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api.events import (
    CitationsEvent,
    ErrorEvent,
    FinalEvent,
    NodeEvent,
    SSEEvent,
    TokenEvent,
    to_sse,
)
from app.auth import Identity, get_identity
from app.cache.semantic_cache import SemanticCache
from app.config import get_settings
from app.graph.build import build_graph
from app.graph.nodes import NodeDeps
from app.graph.state import initial_state
from app.llm import OpenAICompatClient
from app.models.embeddings import embed
from app.retrieval.hybrid_search import search
from app.retrieval.reranker import rerank

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

NODE_LABELS = {
    "cache_check": "Checking cache...",
    "retrieve": "Retrieving...",
    "grade_documents": "Assessing sources...",
    "rewrite_query": "Refining the search...",
    "generate": "Generating...",
    "guardrail_verify": "Verifying sources...",
    "cache_store": "Saving answer...",
    "prepare_regeneration": "Preparing a revision...",
    "abstain": "Preparing a safe response...",
}


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=8000)


def get_chat_deps() -> Iterator[NodeDeps]:
    settings = get_settings()
    llm = OpenAICompatClient(
        "fast",
        settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
    )
    redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)

    def searcher(tenant_id: str, roles: list[str], query: str):
        query_vector = embed([query])[0]
        with psycopg.connect(settings.database_url) as connection:
            return search(connection, tenant_id, roles, query, query_vector)

    deps = NodeDeps(
        llm=llm,
        searcher=searcher,
        reranker=rerank,
        settings=settings,
        cache=SemanticCache(
            redis_client,
            cache_sim=settings.cache_sim,
            ttl_seconds=settings.cache_ttl_seconds,
        ),
        embedder=embed,
    )
    try:
        yield deps
    finally:
        redis_client.close()


def _value(output: Any, key: str, default: Any = None) -> Any:
    if isinstance(output, dict):
        return output.get(key, default)
    return getattr(output, key, default)


def _answer_text(output: Any) -> str | None:
    answer = _value(output, "answer")
    if isinstance(answer, dict):
        text = answer.get("text")
    else:
        text = getattr(answer, "text", None)
    return text if isinstance(text, str) and text else None


def _citations(output: Any) -> list[dict[str, Any]]:
    answer = _value(output, "answer")
    if isinstance(answer, dict):
        values = answer.get("citations", [])
    else:
        values = getattr(answer, "citations", [])
    result = []
    for citation in values or []:
        if isinstance(citation, dict):
            result.append(citation)
        else:
            result.append({"doc_name": citation.doc_name, "page": citation.page})
    return result


def _field(output: Any, key: str, default: Any) -> Any:
    value = _value(output, key, default)
    return default if value is None else value


def _log_request(
    *,
    request_id: str,
    tenant_id: str,
    node_latencies: dict[str, list[float]],
    state: dict[str, Any],
) -> None:
    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "tenant_id": tenant_id,
                "node_latency_ms": node_latencies,
                "retry_count": state.get("retry_count", 0),
                "abstained": state.get("abstained", False),
                "cache_hit": state.get("cache_hit", False),
            },
            separators=(",", ":"),
        )
    )


async def _event_stream(
    request: Request,
    identity: Identity,
    question: str,
    deps: NodeDeps,
    request_id: str,
) -> AsyncIterator[str]:
    node_started: dict[str, float] = {}
    node_latencies: dict[str, list[float]] = {}
    final_state: dict[str, Any] = {}
    sequence = 0
    generated_text: str | None = None

    def frame(event: SSEEvent) -> str:
        nonlocal sequence
        sequence += 1
        return to_sse(event.model_copy(update={"id": sequence}))

    try:
        graph = build_graph(deps)
        state = initial_state(identity.tenant_id, identity.roles, question)
        final_state.update(state)
        async with asyncio.timeout(deps.settings.chat_timeout_seconds):
            async with aclosing(graph.astream_events(state, version="v2")) as events:
                async for event in events:
                    if await request.is_disconnected():
                        return
                    metadata = event.get("metadata", {})
                    node = metadata.get("langgraph_node")
                    event_type = event.get("event")
                    if node not in NODE_LABELS:
                        if (
                            event_type == "on_chain_end"
                            and event.get("name") == "LangGraph"
                        ):
                            output = event.get("data", {}).get("output")
                            if isinstance(output, dict):
                                final_state.update(output)
                        continue

                    if event_type == "on_chain_start":
                        node_started[node] = time.perf_counter()
                        yield frame(NodeEvent(id=1, label=NODE_LABELS[node]))
                    elif event_type == "on_chain_end":
                        elapsed = (
                            time.perf_counter()
                            - node_started.pop(node, time.perf_counter())
                        ) * 1000
                        node_latencies.setdefault(node, []).append(round(elapsed, 2))
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict):
                            final_state.update(output)
                        if node == "generate":
                            generated_text = _answer_text(output)

        verified = _field(final_state, "verified", None)
        supported = bool(_field(verified, "supported", False))
        abstained = bool(_field(final_state, "abstained", False))
        if generated_text and supported and not abstained:
            yield frame(TokenEvent(id=1, text=generated_text))
        citations = _citations(final_state)
        yield frame(CitationsEvent(id=1, citations=citations))
        yield frame(
            FinalEvent(
                id=1,
                answer=_answer_text(final_state) or "",
                abstained=bool(_field(final_state, "abstained", False)),
                cache_hit=bool(_field(final_state, "cache_hit", False)),
            )
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        yield frame(ErrorEvent(id=1))
    finally:
        _log_request(
            request_id=request_id,
            tenant_id=identity.tenant_id,
            node_latencies=node_latencies,
            state=final_state,
        )


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    identity: Identity = Depends(get_identity),
    deps: NodeDeps = Depends(get_chat_deps),
) -> StreamingResponse:
    request_id = str(uuid.uuid4())
    return StreamingResponse(
        _event_stream(request, identity, body.question, deps, request_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
