from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import main


@pytest.mark.asyncio
async def test_health_returns_service_status(monkeypatch: pytest.MonkeyPatch) -> None:
    database_check = AsyncMock(return_value=True)
    redis_check = AsyncMock(return_value=True)
    monkeypatch.setattr(main, "_check_database", database_check)
    monkeypatch.setattr(main, "_check_redis", redis_check)

    async with AsyncClient(
        transport=ASGITransport(app=main.app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": True, "redis": True}
    database_check.assert_awaited_once()
    redis_check.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("database_available", "redis_available"),
    [(False, True), (True, False)],
)
async def test_health_returns_503_when_service_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    database_available: bool,
    redis_available: bool,
) -> None:
    monkeypatch.setattr(
        main,
        "_check_database",
        AsyncMock(return_value=database_available),
    )
    monkeypatch.setattr(
        main,
        "_check_redis",
        AsyncMock(return_value=redis_available),
    )

    async with AsyncClient(
        transport=ASGITransport(app=main.app),
        base_url="http://test",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "db": database_available,
        "redis": redis_available,
    }
