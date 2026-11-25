from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import ingest as ingest_api
from app.auth import Identity, get_identity


class MemoryRedis:
    def __init__(self):
        self.values: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def set(self, key, value, **_kwargs):
        self.values[key] = value

    def get(self, key):
        return self.values.get(key)


@pytest.fixture
def test_app(monkeypatch):
    from app.api.main import app

    memory_redis = MemoryRedis()
    monkeypatch.setattr(ingest_api, "_redis_client", lambda: memory_redis)
    monkeypatch.setattr(
        ingest_api,
        "get_settings",
        lambda: SimpleNamespace(max_upload_mb=1, cache_sim=0.96, cache_ttl_seconds=60),
    )
    monkeypatch.setattr(
        ingest_api,
        "ingest",
        lambda *args, **kwargs: SimpleNamespace(skipped=True),
    )
    app.dependency_overrides[get_identity] = lambda: Identity(
        tenant_id="tenant_alpha", roles=["all", "engineer"]
    )
    yield app, memory_redis
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upload_returns_202_and_job_status_is_tenant_scoped(test_app):
    app, _memory_redis = test_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/ingest",
            files={"file": ("manual.pdf", b"%PDF-1.4\ncontent", "application/pdf")},
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        own_status = await client.get(f"/api/v1/ingest/{job_id}")
        assert own_status.status_code == 200
        assert own_status.json() == {"job_id": job_id, "status": "done"}

        app.dependency_overrides[get_identity] = lambda: Identity(
            tenant_id="tenant_beta", roles=["all"]
        )
        other_status = await client.get(f"/api/v1/ingest/{job_id}")
        assert other_status.status_code == 404


@pytest.mark.asyncio
async def test_upload_rejects_oversized_pdf(test_app):
    app, _memory_redis = test_app
    payload = b"%PDF-" + b"x" * (1024 * 1024)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/ingest",
            files={"file": ("large.pdf", payload, "application/pdf")},
        )
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_upload_rejects_non_pdf(test_app):
    app, _memory_redis = test_app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/ingest",
            files={"file": ("notes.txt", b"not a PDF", "text/plain")},
        )
    assert response.status_code == 415
