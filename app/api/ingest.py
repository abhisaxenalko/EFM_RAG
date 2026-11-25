from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from pathlib import Path
from typing import Literal

import redis
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict

from app.auth import Identity, get_identity
from app.cache.semantic_cache import SemanticCache
from app.config import get_settings
from app.ingestion.pipeline import ingest

router = APIRouter(prefix="/api/v1", tags=["ingestion"])
JOB_TTL_SECONDS = 24 * 60 * 60
READ_CHUNK_BYTES = 1024 * 1024


class IngestAccepted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str


class IngestJobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal["queued", "running", "done", "failed"]


def _tenant_key(tenant_id: str, job_id: str) -> str:
    tenant_hash = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
    return f"tenant:{tenant_hash}:ingest_job:{job_id}"


def _redis_client() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def _write_job(tenant_id: str, job_id: str, status: str) -> None:
    state = IngestJobStatus(job_id=job_id, status=status)
    with _redis_client() as client:
        client.set(
            _tenant_key(tenant_id, job_id),
            state.model_dump_json(),
            ex=JOB_TTL_SECONDS,
        )


def _process_job(
    tenant_id: str,
    roles: list[str],
    job_id: str,
    file_path: str,
    doc_name: str,
) -> None:
    path = Path(file_path)
    try:
        _write_job(tenant_id, job_id, "running")
        access_role = "all" if "all" in roles else roles[0]
        result = ingest(
            tenant_id,
            access_role,
            path,
            doc_name,
            bump_cache_version=False,
        )
        if not result.skipped:
            settings = get_settings()
            with _redis_client() as client:
                SemanticCache(
                    client,
                    cache_sim=settings.cache_sim,
                    ttl_seconds=settings.cache_ttl_seconds,
                ).bump_corpus_version(tenant_id)
        _write_job(tenant_id, job_id, "done")
    except Exception:
        _write_job(tenant_id, job_id, "failed")
    finally:
        path.unlink(missing_ok=True)


@router.post("/ingest", response_model=IngestAccepted, status_code=202)
async def submit_ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    identity: Identity = Depends(get_identity),
) -> IngestAccepted:
    filename = Path(file.filename or "").name
    if not filename or Path(filename).suffix.casefold() != ".pdf":
        raise HTTPException(status_code=415, detail="Only PDF uploads are accepted")
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    if max_bytes < 1:
        raise HTTPException(status_code=500, detail="Upload limit is not configured")

    temporary_path: str | None = None
    size = 0
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temporary:
            temporary_path = temporary.name
            first = await file.read(5)
            if first != b"%PDF-":
                raise HTTPException(
                    status_code=415, detail="Only PDF uploads are accepted"
                )
            temporary.write(first)
            size += len(first)
            if size > max_bytes:
                raise HTTPException(
                    status_code=413, detail="Upload exceeds MAX_UPLOAD_MB"
                )
            while block := await file.read(READ_CHUNK_BYTES):
                size += len(block)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Upload exceeds MAX_UPLOAD_MB",
                    )
                temporary.write(block)
        job_id = str(uuid.uuid4())
        try:
            await run_in_threadpool(_write_job, identity.tenant_id, job_id, "queued")
        except redis.RedisError as exc:
            raise HTTPException(
                status_code=503, detail="Job service unavailable"
            ) from exc
        background_tasks.add_task(
            _process_job,
            identity.tenant_id,
            identity.roles,
            job_id,
            temporary_path,
            filename,
        )
        temporary_path = None
        return IngestAccepted(job_id=job_id)
    finally:
        await file.close()
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


@router.get("/ingest/{job_id}", response_model=IngestJobStatus)
async def ingest_status(
    job_id: str,
    identity: Identity = Depends(get_identity),
) -> IngestJobStatus:
    try:
        raw = await run_in_threadpool(_read_job, identity.tenant_id, job_id)
    except redis.RedisError as exc:
        raise HTTPException(status_code=503, detail="Job service unavailable") from exc
    if raw is None:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        return IngestJobStatus.model_validate_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


def _read_job(tenant_id: str, job_id: str) -> str | None:
    with _redis_client() as client:
        value = client.get(_tenant_key(tenant_id, job_id))
        return value if isinstance(value, str) else None
