from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from starlette.requests import Request

from app import auth
from app.config import Settings


@pytest.fixture
def signing_keys() -> tuple[str, str]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("ascii")
    public_pem = (
        private_key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )
    return private_pem, public_pem


def _settings(env: str, public_key: str) -> Settings:
    return Settings(
        child_tokens=800,
        child_overlap=75,
        parent_tokens=1200,
        k_each=20,
        k_fused=20,
        k_final=5,
        rrf_k=60,
        cache_sim=0.96,
        max_rewrites=1,
        max_regenerations=1,
        health_check_timeout=3,
        database_url="postgresql://unused",
        redis_url="redis://unused",
        llm_base_url="http://unused",
        llm_model="unused",
        llm_api_key="unused",
        env=env,
        jwt_issuer="efm-test-issuer",
        jwt_audience="efm-test-api",
        jwt_public_key=public_key,
    )


def _request(headers: dict[str, str] | None = None) -> Request:
    encoded = [
        (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
    ]
    return Request({"type": "http", "headers": encoded, "method": "GET", "path": "/"})


def _token(
    private_key: str, *, audience: str = "efm-test-api", expires: int = 60
) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "iss": "efm-test-issuer",
            "aud": audience,
            "iat": now,
            "exp": now + timedelta(seconds=expires),
            "tenant_id": "tenant_alpha",
            "roles": ["all", "engineer"],
        },
        private_key,
        algorithm="RS256",
    )


@pytest.mark.asyncio
async def test_no_token_returns_401(monkeypatch, signing_keys):
    monkeypatch.setattr(
        auth, "get_settings", lambda: _settings("prod", signing_keys[1])
    )
    with pytest.raises(HTTPException) as exc:
        await auth.get_identity(_request())
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_expired_token_returns_401(monkeypatch, signing_keys):
    monkeypatch.setattr(
        auth, "get_settings", lambda: _settings("prod", signing_keys[1])
    )
    token = _token(signing_keys[0], expires=-1)
    with pytest.raises(HTTPException) as exc:
        await auth.get_identity(_request({"Authorization": f"Bearer {token}"}))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_wrong_audience_returns_401(monkeypatch, signing_keys):
    monkeypatch.setattr(
        auth, "get_settings", lambda: _settings("prod", signing_keys[1])
    )
    token = _token(signing_keys[0], audience="another-api")
    with pytest.raises(HTTPException) as exc:
        await auth.get_identity(_request({"Authorization": f"Bearer {token}"}))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_valid_rs256_token_returns_identity(monkeypatch, signing_keys):
    monkeypatch.setattr(
        auth, "get_settings", lambda: _settings("prod", signing_keys[1])
    )
    token = _token(signing_keys[0])
    identity = await auth.get_identity(_request({"Authorization": f"Bearer {token}"}))
    assert identity.tenant_id == "tenant_alpha"
    assert identity.roles == ["all", "engineer"]


@pytest.mark.asyncio
async def test_prod_rejects_dev_headers(monkeypatch, signing_keys):
    monkeypatch.setattr(
        auth, "get_settings", lambda: _settings("prod", signing_keys[1])
    )
    with pytest.raises(HTTPException) as exc:
        await auth.get_identity(
            _request({"X-Tenant-ID": "tenant_alpha", "X-Roles": "all,engineer"})
        )
    assert exc.value.status_code == 401
