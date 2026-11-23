from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

# RLS is the tenant boundary. Do not add tenant_id predicates here: those can
# drift between the dense and lexical branches and hide policy mistakes.
HYBRID_SEARCH_SQL = """
WITH lexical_query AS MATERIALIZED (
    SELECT websearch_to_tsquery(
        'simple', regexp_replace(%s, '[-_/]+', ' ', 'g')
    ) AS query
), dense_candidates AS MATERIALIZED (
    SELECT nearest.id,
           row_number() OVER (ORDER BY nearest.distance, nearest.id) AS rank
    FROM (
        SELECT chunks.id, chunks.embedding <=> %s::vector AS distance
        FROM document_chunks AS chunks
        WHERE chunks.status = 'active'
        ORDER BY chunks.embedding <=> %s::vector
        LIMIT %s
    ) AS nearest
), lexical_candidates AS MATERIALIZED (
    SELECT matches.id,
           row_number() OVER (
               ORDER BY matches.lexical_score DESC, matches.id
           ) AS rank
    FROM (
        SELECT chunks.id,
               ts_rank_cd(chunks.tsv, lexical_query.query) AS lexical_score
        FROM document_chunks AS chunks
        CROSS JOIN lexical_query
        WHERE chunks.status = 'active'
          AND chunks.tsv @@ lexical_query.query
        ORDER BY lexical_score DESC, chunks.id
        LIMIT %s
    ) AS matches
), ranked_hits AS (
    SELECT id, rank FROM dense_candidates
    UNION ALL
    SELECT id, rank FROM lexical_candidates
), fused AS (
    SELECT id,
           sum(1.0::double precision / (%s::double precision + rank)) AS rrf_score
    FROM ranked_hits
    GROUP BY id
)
SELECT chunks.id,
       parent.id,
       registry.source_name,
       parent.page_start,
       chunks.content,
       parent.content,
       fused.rrf_score
FROM fused
JOIN document_chunks AS chunks ON chunks.id = fused.id
JOIN parent_documents AS parent ON parent.id = chunks.parent_document_id
JOIN documents_registry AS registry ON registry.id = parent.registry_id
WHERE chunks.status = 'active'
  AND parent.status = 'active'
  AND registry.status = 'active'
ORDER BY fused.rrf_score DESC, chunks.id
LIMIT %s
"""

_CODE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9]*-[A-Za-z0-9]+)(?![A-Za-z0-9])"
)


@dataclass(frozen=True)
class Candidate:
    chunk_id: int
    parent_id: int
    doc_name: str
    page: int
    child_text: str
    parent_text: str
    rrf_score: float


def _vector_literal(embedding: Sequence[float]) -> str:
    if len(embedding) != 1024:
        raise ValueError("The query embedding must contain exactly 1024 values")
    values = [float(value) for value in embedding]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("The query embedding must contain only finite values")
    return "[" + ",".join(repr(value) for value in values) + "]"


def _lexical_text(query_text: str) -> str:
    codes = _CODE_TOKEN.findall(query_text)
    if not codes:
        return query_text
    # websearch_to_tsquery supports quoted phrases; the SQL normalizes the
    # hyphen to a space so the simple parser can match adjacent code lexemes.
    return query_text + " " + " ".join(f'"{code}"' for code in dict.fromkeys(codes))


def search(
    conn: Any,
    tenant_id: str,
    roles: Sequence[str],
    query_text: str,
    query_embedding: Sequence[float],
) -> list[Candidate]:
    """Run tenant-scoped dense and lexical search, fused with reciprocal rank."""
    settings = get_settings()
    if not tenant_id:
        raise ValueError("tenant_id must not be empty")
    if isinstance(roles, str) or not roles or any(not role for role in roles):
        raise ValueError("roles must be a non-empty sequence of role names")
    if settings.k_each <= 0 or settings.k_fused <= 0 or settings.rrf_k < 0:
        raise ValueError("K_EACH and K_FUSED must be positive and RRF_K non-negative")
    vector = _vector_literal(query_embedding)
    params = (
        _lexical_text(query_text), vector, vector, settings.k_each,
        settings.k_each, settings.rrf_k, settings.k_fused,
    )
    started = perf_counter()
    count = 0
    try:
        with conn.transaction():
            conn.execute(
                "SELECT set_config('app.current_tenant_id', %s, true)",
                (tenant_id,),
            )
            conn.execute(
                "SELECT set_config('app.allowed_roles', %s, true)",
                (",".join(roles),),
            )
            conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", ("100",))
            iterative = conn.execute(
                "SELECT current_setting('hnsw.iterative_scan', true)"
            ).fetchone()[0]
            if iterative is not None:
                conn.execute(
                    "SELECT set_config('hnsw.iterative_scan', %s, true)",
                    ("relaxed_order",),
                )
            rows = conn.execute(HYBRID_SEARCH_SQL, params).fetchall()
            candidates = [Candidate(*row) for row in rows]
            count = len(candidates)
        return candidates
    finally:
        logger.info(
            "hybrid search completed rows=%d elapsed_ms=%.2f",
            count,
            (perf_counter() - started) * 1000,
        )
