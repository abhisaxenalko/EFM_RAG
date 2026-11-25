from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import chat
from app.api.main import app
from app.auth import Identity, get_identity
from app.graph.nodes import NodeDeps
from app.llm import FakeLLM
from app.models.schemas import Answer, Verdict
from app.retrieval.hybrid_search import Candidate


def _deps() -> NodeDeps:
    candidate = Candidate(
        chunk_id=1,
        parent_id=1,
        doc_name="manual.pdf",
        page=1,
        child_text="ERR-409B indicates low water flow.",
        parent_text="ERR-409B indicates low water flow.",
        rrf_score=0.9,
        rerank_score=0.95,
    )
    llm = FakeLLM(
        structured_responses={
            Answer: [
                {
                    "text": "ERR-409B indicates low water flow. [manual.pdf, Page 1]",
                    "citations": [{"doc_name": "manual.pdf", "page": 1}],
                }
            ],
            Verdict: [{"supported": True, "unsupported_claims": []}],
        }
    )
    return NodeDeps(
        llm=llm,
        searcher=lambda *_args: [candidate],
        reranker=lambda _query, candidates, **_kwargs: list(candidates),
        settings=SimpleNamespace(
            grade_low=0.35,
            grade_high=0.75,
            k_final=5,
            max_rewrites=1,
            max_regenerations=1,
            chat_timeout_seconds=5,
            llm_api_key=None,
        ),
    )


@pytest.fixture(autouse=True)
def override_identity():
    app.dependency_overrides[get_identity] = lambda: Identity(
        tenant_id="tenant_alpha", roles=["all"]
    )
    yield
    app.dependency_overrides.clear()


def _events(response_text: str) -> list[tuple[str, dict]]:
    results = []
    for frame in response_text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.splitlines())
        results.append((fields["event"], json.loads(fields["data"])))
    return results


@pytest.mark.asyncio
async def test_stream_orders_node_token_and_ends_with_final():
    app.dependency_overrides[chat.get_chat_deps] = _deps
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/chat/stream", json={"question": "What is ERR-409B?"}
        )
    events = _events(response.text)
    types = [event_type for event_type, _ in events]
    assert response.status_code == 200
    assert types.index("node") < types.index("token")
    assert types[-2:] == ["citations", "final"]
    assert events[-1][1]["answer"].startswith("ERR-409B")
    assert all("grader_score" not in data for _, data in events)


@pytest.mark.asyncio
async def test_client_disconnect_cancels_graph_run(monkeypatch):
    deps = _deps()
    cancelled = asyncio.Event()

    class BlockingGraph:
        async def astream_events(self, _state, *, version):
            assert version == "v2"
            try:
                yield {
                    "event": "on_chain_start",
                    "metadata": {"langgraph_node": "retrieve"},
                    "data": {},
                }
                await asyncio.Future()
            finally:
                cancelled.set()

    monkeypatch.setattr(chat, "build_graph", lambda _deps: BlockingGraph())

    async def receive():
        return {"type": "http.disconnect"}

    request = chat.Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [],
        },
        receive,
    )
    stream = chat._event_stream(
        request,
        Identity(tenant_id="tenant_alpha", roles=["all"]),
        "query",
        deps,
        "request-id",
    )
    assert [event async for event in stream] == []
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_internal_error_streams_only_generic_error():
    class BrokenLLM:
        def __init__(self):
            self.calls = 0

        async def complete_structured(self, *, system, user, schema):
            self.calls += 1
            if schema is Answer:
                raise RuntimeError("private details and prompt text")
            raise AssertionError("verifier should not run")

        async def complete_text(self, *, system, user):
            raise AssertionError("rewriter should not run")

    deps = _deps()
    deps = NodeDeps(
        llm=BrokenLLM(),
        searcher=deps.searcher,
        reranker=deps.reranker,
        settings=deps.settings,
    )
    app.dependency_overrides[chat.get_chat_deps] = lambda: deps
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/chat/stream", json={"question": "What is ERR-409B?"}
        )
    events = _events(response.text)
    assert events[-1] == (
        "error",
        {
            "id": events[-1][1]["id"],
            "type": "error",
            "message": "The request could not be completed. Please try again.",
        },
    )
    assert "private details" not in response.text
