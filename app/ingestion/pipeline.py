from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import psycopg
import redis

from app.cache.semantic_cache import SemanticCache
from app.config import get_settings
from app.ingestion.chunker import build_children, build_parents
from app.ingestion.parser import parse_document
from app.models.embeddings import embed


@dataclass(frozen=True)
class IngestResult:
    doc_id: int
    version: int
    parents: int
    children: int
    skipped: bool


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector_literal(values: list[float]) -> str:
    if len(values) != 1024:
        raise ValueError("Every embedding must have exactly 1024 dimensions")
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Embeddings must contain only finite values")
    return "[" + ",".join(repr(value) for value in vector) + "]"


def ingest(
    tenant_id: str,
    access_role: str,
    path: str | Path,
    doc_name: str,
) -> IngestResult:
    """Parse, chunk, embed, and atomically activate a tenant document version.

    The configured database login must be the tenant-specific login provisioned
    in ``efm_private.tenant_principals`` and granted membership in ``efm_app``.
    """
    if not tenant_id:
        raise ValueError("tenant_id must not be empty")
    if not access_role or "," in access_role:
        raise ValueError("access_role must be a non-empty role name")
    if not doc_name:
        raise ValueError("doc_name must not be empty")

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    content_hash = _file_sha256(source)

    sections = parse_document(source)
    parents = build_parents(sections)
    children_by_parent = [
        build_children(parent, tenant_id, doc_name) for parent in parents
    ]
    flat_children = [child for group in children_by_parent for child in group]
    vectors = embed(flat_children)
    if len(vectors) != len(flat_children):
        raise ValueError("The embedder returned a different number of vectors")
    vector_literals = [_vector_literal(vector) for vector in vectors]

    settings = get_settings()
    with psycopg.connect(settings.database_url) as conn:
        with conn.transaction():
            # The URL must authenticate a tenant login; SET ROLE applies the
            # table grants while session_user remains the RLS identity.
            conn.execute("SET LOCAL ROLE efm_app")
            conn.execute(
                "SELECT set_config('app.current_tenant_id', %s, true)",
                (tenant_id,),
            )
            conn.execute(
                "SELECT set_config('app.allowed_roles', %s, true)",
                (access_role,),
            )

            # Serialize same-content retries and same-name version increments.
            lock_names = sorted(
                {
                    json.dumps([tenant_id, "hash", content_hash]),
                    json.dumps([tenant_id, "name", doc_name]),
                }
            )
            for lock_name in lock_names:
                conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (lock_name,),
                )

            existing = conn.execute(
                """
                SELECT registry.id, registry.version,
                       (SELECT count(*) FROM parent_documents AS parent
                        WHERE parent.registry_id = registry.id
                          AND parent.status = 'active'),
                       (SELECT count(*)
                        FROM document_chunks AS chunk
                        JOIN parent_documents AS parent
                          ON parent.id = chunk.parent_document_id
                        WHERE parent.registry_id = registry.id
                          AND parent.status = 'active'
                          AND chunk.status = 'active')
                FROM documents_registry AS registry
                WHERE registry.sha256 = %s AND registry.status = 'active'
                LIMIT 1
                """,
                (content_hash,),
            ).fetchone()
            if existing is not None:
                return IngestResult(
                    doc_id=existing[0],
                    version=existing[1],
                    parents=existing[2],
                    children=existing[3],
                    skipped=True,
                )

            version = conn.execute(
                """
                SELECT COALESCE(max(version), 0) + 1
                FROM documents_registry
                WHERE source_name = %s
                """,
                (doc_name,),
            ).fetchone()[0]

            registry_id = conn.execute(
                """
                INSERT INTO documents_registry
                    (tenant_id, access_role, status, source_name, sha256, version)
                VALUES (%s, %s, 'staging', %s, %s, %s)
                RETURNING id
                """,
                (tenant_id, access_role, doc_name, content_hash, version),
            ).fetchone()[0]

            child_offset = 0
            for parent, child_texts in zip(parents, children_by_parent, strict=True):
                parent_id = conn.execute(
                    """
                    INSERT INTO parent_documents
                        (tenant_id, access_role, status, registry_id, heading_path,
                         page_start, is_table, content)
                    VALUES (%s, %s, 'staging', %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        tenant_id,
                        access_role,
                        registry_id,
                        parent.heading_path,
                        parent.page_start,
                        parent.is_table,
                        parent.text_md,
                    ),
                ).fetchone()[0]
                for child_index, child_text in enumerate(child_texts):
                    conn.execute(
                        """
                        INSERT INTO document_chunks
                            (tenant_id, access_role, status, parent_document_id,
                             chunk_index, content, embedding)
                        VALUES (%s, %s, 'staging', %s, %s, %s, %s::vector)
                        """,
                        (
                            tenant_id,
                            access_role,
                            parent_id,
                            child_index,
                            child_text,
                            vector_literals[child_offset],
                        ),
                    )
                    child_offset += 1

            # Archive the old active version of this name before activating the
            # staging version. RLS scopes each update to this login and context.
            conn.execute(
                """
                UPDATE document_chunks AS chunk SET status = 'archived'
                WHERE chunk.status = 'active'
                  AND chunk.parent_document_id IN (
                      SELECT parent.id
                      FROM parent_documents AS parent
                      JOIN documents_registry AS registry
                        ON registry.id = parent.registry_id
                      WHERE registry.source_name = %s
                        AND registry.status = 'active'
                  )
                """,
                (doc_name,),
            )
            conn.execute(
                """
                UPDATE parent_documents AS parent SET status = 'archived'
                WHERE parent.status = 'active'
                  AND parent.registry_id IN (
                      SELECT id FROM documents_registry
                      WHERE source_name = %s AND status = 'active'
                  )
                """,
                (doc_name,),
            )
            conn.execute(
                "UPDATE documents_registry SET status = 'archived' "
                "WHERE source_name = %s AND status = 'active'",
                (doc_name,),
            )

            conn.execute(
                "UPDATE documents_registry SET status = 'active' WHERE id = %s",
                (registry_id,),
            )
            conn.execute(
                "UPDATE parent_documents SET status = 'active' "
                "WHERE registry_id = %s",
                (registry_id,),
            )
            conn.execute(
                """
                UPDATE document_chunks SET status = 'active'
                WHERE parent_document_id IN (
                    SELECT id FROM parent_documents WHERE registry_id = %s
                )
                """,
                (registry_id,),
            )

    with redis.Redis.from_url(settings.redis_url) as redis_client:
        SemanticCache(
            redis_client,
            cache_sim=settings.cache_sim,
            ttl_seconds=settings.cache_ttl_seconds,
        ).bump_corpus_version(tenant_id)

    return IngestResult(
        doc_id=registry_id,
        version=version,
        parents=len(parents),
        children=len(flat_children),
        skipped=False,
    )
