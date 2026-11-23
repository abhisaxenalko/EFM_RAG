from __future__ import annotations

import hashlib
import math
import secrets
import string
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    postgres_user: str
    postgres_password: SecretStr
    postgres_db: str


@dataclass(frozen=True)
class RetrievalRows:
    admin_conninfo: str = field(repr=False)
    app_conninfo: str = field(repr=False)
    marker: str
    roles: dict[str, str]
    connection_for: Callable[[str], str] = field(repr=False)
    sla_parent_id: int
    shared_chunk_id: int
    dense_only_chunk_id: int
    query_embedding: list[float]
    rrf_query: str


def _vector(first: float, second: float = 0.0) -> str:
    values = [first, second] + [0.0] * 1022
    return "[" + ",".join(repr(value) for value in values) + "]"


def _unit_vector(seed: str) -> list[float]:
    values = [
        int.from_bytes(
            hashlib.sha256(f"{seed}:{i}".encode()).digest()[:4], "big"
        )
        / 2**31
        - 1.0
        for i in range(1024)
    ]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(repr(value) for value in values) + "]"


@pytest.fixture(scope="module")
def retrieval_rows() -> Iterator[RetrievalRows]:
    settings = RetrievalSettings()
    app_params = conninfo_to_dict(settings.database_url)
    admin_params = {
        **app_params,
        "user": settings.postgres_user,
        "password": settings.postgres_password.get_secret_value(),
        "dbname": settings.postgres_db,
    }
    app_conninfo = make_conninfo(**app_params, connect_timeout=10)
    admin_conninfo = make_conninfo(**admin_params, connect_timeout=10)
    marker = f"retrieval-test-{secrets.token_hex(6)}"
    unique_code = f"retrievaltoken{secrets.token_hex(4)}"
    query_embedding = _unit_vector(marker)
    query_vector = _vector_literal(query_embedding)
    roles = {
        "alpha": f"efm_retrieval_alpha_{secrets.token_hex(5)}",
        "beta": f"efm_retrieval_beta_{secrets.token_hex(5)}",
        "alpha_all": f"efm_retrieval_alpha_all_{secrets.token_hex(5)}",
    }
    passwords: dict[str, str] = {}
    ids: dict[str, int] = {}

    # All test data is inserted through PostgreSQL; table RLS is forced on.
    with psycopg.connect(admin_conninfo, autocommit=True) as admin:
        for principal in roles.values():
            password = "".join(
                secrets.choice(string.ascii_letters + string.digits)
                for _ in range(32)
            )
            passwords[principal] = password
            admin.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD {}"
                ).format(sql.Identifier(principal), sql.Literal(password))
            )
            admin.execute(
                sql.SQL("GRANT efm_app TO {}").format(sql.Identifier(principal))
            )

        with admin.transaction():
            admin.execute("SET ROLE efm_owner")
            for key, tenant_id, allowed_roles in (
                ("alpha", "tenant_alpha", ["all", "engineer"]),
                ("beta", "tenant_beta", ["all", "engineer"]),
                ("alpha_all", "tenant_alpha", ["all"]),
            ):
                admin.execute(
                    """
                    INSERT INTO efm_private.tenant_principals
                        (principal_name, tenant_id, allowed_roles)
                    VALUES (%s, %s, %s)
                    """,
                    (roles[key], tenant_id, allowed_roles),
                )

            records = (
                (
                    "alpha-sla",
                    "tenant_alpha",
                    "all",
                    "What is the emergency repair SLA? Emergency repairs must be "
                    "completed within four hours. ERR-409B indicates low flow. "
                    f"{unique_code}",
                    query_vector,
                ),
                (
                    "alpha-dense-only",
                    "tenant_alpha",
                    "all",
                    "Routine lubrication schedule for motor bearings and vibration.",
                    query_vector,
                ),
                (
                    "alpha-engineer",
                    "tenant_alpha",
                    "engineer",
                    "engineer-only confidential procedure ERR-409B service notes",
                    _vector(0.0, 1.0),
                ),
                (
                    "beta-code",
                    "tenant_beta",
                    "all",
                    "ERR-409B on the beta chiller indicates a restricted flow fault.",
                    _vector(1.0),
                ),
            )
            for label, tenant_id, access_role, content, vector in records:
                doc_name = f"{marker}:{tenant_id}:{label}"
                digest = hashlib.sha256(doc_name.encode()).hexdigest()
                registry_id = admin.execute(
                    """
                    INSERT INTO documents_registry
                        (tenant_id, access_role, source_name, sha256, version)
                    VALUES (%s, %s, %s, %s, 1)
                    RETURNING id
                    """,
                    (tenant_id, access_role, doc_name, digest),
                ).fetchone()[0]
                parent_id = admin.execute(
                    """
                    INSERT INTO parent_documents
                        (tenant_id, access_role, registry_id, content)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (tenant_id, access_role, registry_id, content),
                ).fetchone()[0]
                chunk_id = admin.execute(
                    """
                    INSERT INTO document_chunks
                        (tenant_id, access_role, parent_document_id,
                         chunk_index, content, embedding)
                    VALUES (%s, %s, %s, 0, %s, %s::vector)
                    RETURNING id
                    """,
                    (tenant_id, access_role, parent_id, content, vector),
                ).fetchone()[0]
                ids[label] = chunk_id
                if label == "alpha-sla":
                    ids["sla_parent"] = parent_id
            admin.execute("RESET ROLE")

    def connection_for(principal: str) -> str:
        params = conninfo_to_dict(app_conninfo)
        params["user"] = principal
        params["password"] = passwords[principal]
        return make_conninfo(**params)

    try:
        yield RetrievalRows(
            admin_conninfo=admin_conninfo,
            app_conninfo=app_conninfo,
            marker=marker,
            roles=roles,
            connection_for=connection_for,
            sla_parent_id=ids["sla_parent"],
            shared_chunk_id=ids["alpha-sla"],
            dense_only_chunk_id=ids["alpha-dense-only"],
            query_embedding=query_embedding,
            rrf_query=unique_code,
        )
    finally:
        with psycopg.connect(admin_conninfo, autocommit=True) as admin:
            with admin.transaction():
                admin.execute("SET ROLE efm_owner")
                admin.execute(
                    "DELETE FROM documents_registry WHERE source_name LIKE %s",
                    (f"{marker}:%",),
                )
                admin.execute(
                    "DELETE FROM efm_private.tenant_principals "
                    "WHERE principal_name = ANY(%s)",
                    (list(roles.values()),),
                )
                admin.execute("RESET ROLE")
            for principal in roles.values():
                admin.execute(
                    sql.SQL("REVOKE efm_app FROM {}").format(sql.Identifier(principal))
                )
                admin.execute(
                    sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(principal))
                )


def _connect_as_app(conninfo: str) -> psycopg.Connection:
    conn = psycopg.connect(conninfo)
    conn.execute("SET ROLE efm_app")
    return conn


def _query_vector() -> list[float]:
    return [1.0] + [0.0] * 1023


@pytest.mark.isolation
def test_err_code_is_found_when_dense_branch_is_empty(
    retrieval_rows: RetrievalRows,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.retrieval import hybrid_search

    dense_marker = (
        "FROM document_chunks AS chunks\n"
        "        WHERE chunks.status = 'active'\n"
        "        ORDER BY chunks.embedding <=> %s::vector"
    )
    assert dense_marker in hybrid_search.HYBRID_SEARCH_SQL
    monkeypatch.setattr(
        hybrid_search,
        "HYBRID_SEARCH_SQL",
        hybrid_search.HYBRID_SEARCH_SQL.replace(
            dense_marker,
            dense_marker.replace("chunks.status = 'active'", "false"),
            1,
        ),
    )
    alpha_conninfo = retrieval_rows.connection_for(retrieval_rows.roles["alpha"])
    with _connect_as_app(alpha_conninfo) as conn:
        results = hybrid_search.search(
            conn,
            "tenant_alpha",
            ["all"],
            "ERR-409B",
            _query_vector(),
        )
    assert any("ERR-409B" in result.child_text for result in results)
    assert any(result.chunk_id == retrieval_rows.shared_chunk_id for result in results)


@pytest.mark.isolation
def test_search_never_returns_the_other_tenant(
    retrieval_rows: RetrievalRows,
) -> None:
    from app.retrieval.hybrid_search import search

    for tenant, principal, other_tenant in (
        ("tenant_alpha", "alpha", "tenant_beta"),
        ("tenant_beta", "beta", "tenant_alpha"),
    ):
        conninfo = retrieval_rows.connection_for(retrieval_rows.roles[principal])
        with _connect_as_app(conninfo) as conn:
            results = search(
                conn, tenant, ["all", "engineer"], "ERR-409B", _query_vector()
            )
        assert results
        assert all(tenant in candidate.doc_name for candidate in results)
        assert all(other_tenant not in candidate.doc_name for candidate in results)


@pytest.mark.isolation
def test_no_tenant_setting_returns_zero_rows(
    retrieval_rows: RetrievalRows,
) -> None:
    with _connect_as_app(retrieval_rows.app_conninfo) as conn:
        with conn.transaction():
            counts = [
                conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in (
                    "documents_registry",
                    "parent_documents",
                    "document_chunks",
                )
            ]
    assert counts == [0, 0, 0]


@pytest.mark.isolation
def test_all_role_caller_never_gets_engineer_section(
    retrieval_rows: RetrievalRows,
) -> None:
    from app.retrieval.hybrid_search import search

    with _connect_as_app(
        retrieval_rows.connection_for(retrieval_rows.roles["alpha_all"])
    ) as conn:
        results = search(
            conn,
            "tenant_alpha",
            ["all"],
            "engineer-only confidential procedure",
            _query_vector(),
        )
    assert all("alpha-engineer" not in candidate.doc_name for candidate in results)
    assert all("engineer-only" not in candidate.child_text for candidate in results)


@pytest.mark.isolation
def test_rrf_adds_contributions_when_chunk_is_in_both_lists(
    retrieval_rows: RetrievalRows,
) -> None:
    from app.config import get_settings
    from app.retrieval.hybrid_search import search

    alpha_conninfo = retrieval_rows.connection_for(retrieval_rows.roles["alpha"])
    with _connect_as_app(alpha_conninfo) as conn:
        results = search(
            conn,
            "tenant_alpha",
            ["all"],
            retrieval_rows.rrf_query,
            retrieval_rows.query_embedding,
        )

    by_chunk = {candidate.chunk_id: candidate for candidate in results}
    shared = by_chunk[retrieval_rows.shared_chunk_id]
    dense_only = by_chunk[retrieval_rows.dense_only_chunk_id]
    k = get_settings().rrf_k
    assert math.isclose(shared.rrf_score, 2 / (k + 1), rel_tol=1e-6)
    assert math.isclose(dense_only.rrf_score, 1 / (k + 2), rel_tol=1e-6)
    assert shared.rrf_score > dense_only.rrf_score


@pytest.mark.isolation
def test_sla_question_top_five_contains_the_expected_parent(
    retrieval_rows: RetrievalRows,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.retrieval.hybrid_search import search
    from app.retrieval.reranker import rerank

    monkeypatch.setenv("RERANKER", "fake")
    query = "What is the emergency repair SLA?"
    alpha_conninfo = retrieval_rows.connection_for(retrieval_rows.roles["alpha"])
    with _connect_as_app(alpha_conninfo) as conn:
        candidates = search(
            conn,
            "tenant_alpha",
            ["all"],
            query,
            _query_vector(),
        )
    ranked = rerank(query, candidates, top_n=5)
    assert any(
        candidate.parent_id == retrieval_rows.sla_parent_id
        for candidate in ranked
    )
