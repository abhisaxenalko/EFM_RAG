import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import psycopg
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.api.chat import router as chat_router
from app.api.ingest import router as ingest_router
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    get_settings()
    yield


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    db: bool
    redis: bool


app = FastAPI(lifespan=lifespan)
app.include_router(ingest_router)
app.include_router(chat_router)

console_dist = Path(__file__).resolve().parents[2] / "web" / "console" / "dist"
if console_dist.is_dir():
    app.mount(
        "/console",
        StaticFiles(directory=console_dist, html=True),
        name="console",
    )


async def _check_database() -> bool:
    settings = get_settings()
    return await asyncio.to_thread(_probe_database, settings)


def _probe_database(settings: Settings) -> bool:
    try:
        with psycopg.connect(
            settings.database_url,
            connect_timeout=settings.health_check_timeout,
        ):
            pass
        return True
    except (OSError, psycopg.Error) as error:
        logger.warning("Database health check failed (%s)", type(error).__name__)
        return False


async def _check_redis() -> bool:
    settings = get_settings()
    try:
        async with Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=settings.health_check_timeout,
            socket_timeout=settings.health_check_timeout,
        ) as client:
            return bool(await client.ping())
    except (OSError, RedisError) as error:
        logger.warning("Redis health check failed (%s)", type(error).__name__)
        return False


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse | JSONResponse:
    db_available, redis_available = await asyncio.gather(
        _check_database(),
        _check_redis(),
    )
    health_response = HealthResponse(
        status="ok" if db_available and redis_available else "unavailable",
        db=db_available,
        redis=redis_available,
    )
    if not db_available or not redis_available:
        return JSONResponse(status_code=503, content=health_response.model_dump())
    return health_response
