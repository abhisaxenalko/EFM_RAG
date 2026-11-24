from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.llm import EfmLLMError, OpenAICompatClient
from app.models.schemas import GradeOutput

CATALOG = Path(__file__).parent / "fixtures" / "llm_models.yaml"


def _response(content, *, prompt=10, completion=4):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        },
    )


def _client(handler):
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    client = OpenAICompatClient(
        "fast",
        "secret-from-env",
        catalog_path=CATALOG,
        http_client=http_client,
    )
    return client, http_client


def test_model_alias_client_uses_catalog_model_and_reuses_client():
    client = OpenAICompatClient("fast", "test-key", catalog_path=CATALOG)

    fast = client.for_alias("fast")
    strong = client.for_alias("strong")

    assert fast is client
    assert strong.model_name == "strong"
    assert strong.models[strong.model_name].model == "test-strong"
    assert client.for_alias("strong") is strong


@pytest.mark.asyncio
async def test_openai_compat_success_validates_and_records_usage():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return _response('{"score":"relevant","missing_aspect":""}')

    client, http_client = _client(handler)
    try:
        result, usage = await client.complete_structured(
            system="Grade evidence.", user="Question", schema=GradeOutput
        )
        assert result == GradeOutput(score="relevant")
        assert usage.model == "test-fast"
        assert usage.input_tokens == 10
        assert usage.output_tokens == 4
        assert usage.cost_usd == pytest.approx(0.000036)
        assert calls[0]["response_format"] == {"type": "json_object"}
        assert "JSON Schema" in calls[0]["messages"][0]["content"]
        assert client.consume_usage_log() == [usage]
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_invalid_json_retries_once_with_validation_repair():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        content = "not json" if len(calls) == 1 else '{"score":"ambiguous"}'
        return _response(content)

    client, http_client = _client(handler)
    try:
        result, _ = await client.complete_structured(
            system="Grade.", user="Question", schema=GradeOutput
        )
        assert result.score == "ambiguous"
        assert len(calls) == 2
        assert "Validation error:" in calls[1]["messages"][-1]["content"]
        assert len(client.consume_usage_log()) == 2
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_provider_error_retries_then_uses_fallback():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        if "fast.example" in str(request.url):
            return httpx.Response(503, text="provider unavailable")
        return _response('{"score":"relevant"}')

    client, http_client = _client(handler)
    try:
        result, usage = await client.complete_structured(
            system="Grade.", user="Question", schema=GradeOutput
        )
        entries = client.consume_usage_log()
        assert result.score == "relevant"
        assert usage.model == "test-strong"
        assert sum("fast.example" in url for url in urls) == 2
        assert sum("strong.example" in url for url in urls) == 1
        assert any(entry.event == "model_fallback" for entry in entries)
        assert entries[-1].model == "test-strong"
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_primary_and_fallback_fail_with_named_error():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(503, text="provider unavailable")

    client, http_client = _client(handler)
    try:
        with pytest.raises(EfmLLMError, match="test-fast -> test-strong"):
            await client.complete_structured(
                system="Grade.", user="Question", schema=GradeOutput
            )
        entries = client.consume_usage_log()
        assert len(urls) == 4
        assert sum(entry.event == "model_fallback" for entry in entries) == 1
        assert sum(entry.event == "call" for entry in entries) == 4
        assert all(entry.latency_ms >= 0 for entry in entries)
    finally:
        await http_client.aclose()
