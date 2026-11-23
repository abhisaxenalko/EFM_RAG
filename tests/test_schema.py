from __future__ import annotations

import hashlib
import secrets
import string
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.errors import InsufficientPrivilege
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class SchemaRows:
    admin_conninfo: str = field(repr=False)
    app_conninfo: str = field(repr=False)
    marker: str
    roles: dict[str, str]
    connection_for: Callable[[str], str] = field(repr=False)


class SchemaTestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    postgres_user: str
    postgres_password: SecretStr
    postgres_db: str


@pytest.fixture(scope="session")
def database_connections() -> dict[str, str]:
    settings = SchemaTestSettings()
    app_params = conninfo_to_dict(settings.database_url)
    admin_params = {
        **app_params,
        "user": settings.postgres_user,
        "password": settings.postgres_password.get_secret_value(),
        "dbname": settings.postgres_db,
    }
    return {
        "app": make_conninfo(**app_params, connect_timeout=10),
        "admin": make_conninfo(**admin_params, connect_timeout=10),
    }


@pytest.fixture
def schema_rows(database_connections: dict[str, str]) -> Iterator[SchemaRows]:
    admin_conninfo = database_connections["admin"]
    suffix = secrets.token_hex(6)
    marker = f"schema-test-{suffix}"
    roles = {
        "alpha": f"efm_test_alpha_{suffix}",
        "beta": f"efm_test_beta_{suffix}",
        "alpha_all": f"efm_test_alpha_all_{suffix}",
    }
    credentials: dict[str, str] = {}

    with psycopg.connect(admin_conninfo, autocommit=True) as admin:
        for principal in roles.values():
            password = "".join(
                secrets.choice(string.ascii_letters + string.digits) for _ in range(32)
            )
            credentials[principal] = password
            admin.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD {}"
                ).format(sql.Identifier(principal), sql.Literal(password))
            )
            admin.execute(
                sql.SQL("GRANT efm_app TO {}").format(sql.Identifier(principal))
            )

        # Seed and authorize rows as the NOLOGIN table owner. The owner policies
        # keep FORCE ROW LEVEL SECURITY enabled during setup and cleanup.
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

            for tenant_id in ("tenant_alpha", "tenant_beta"):
                for access_role in ("all", "engineer"):
                    source_name = f"{marker}:{tenant_id}:{access_role}"
                    registry = admin.execute(
                        """
                        INSERT INTO documents_registry
                            (tenant_id, access_role, source_name, sha256, version)
                        VALUES (%s, %s, %s, %s, 1)
                        RETURNING id
                        """,
                        (
                            tenant_id,
                            access_role,
                            source_name,
                            hashlib.sha256(source_name.encode()).hexdigest(),
                        ),
                    ).fetchone()
                    parent = admin.execute(
                        """
                        INSERT INTO parent_documents
                            (tenant_id, access_role, registry_id, content)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            tenant_id,
                            access_role,
                            registry[0],
                            f"{source_name} ERR-409B indicates low flow",
                        ),
                    ).fetchone()
                    vector = "[" + ",".join("0" for _ in range(1024)) + "]"
                    admin.execute(
                        """
                        INSERT INTO document_chunks
                            (tenant_id, access_role, parent_document_id,
                             chunk_index, content, embedding)
                        VALUES (%s, %s, %s, 0, %s, %s::vector)
                        """,
                        (
                            tenant_id,
                            access_role,
                            parent[0],
                            f"{source_name} ERR-409B indicates low flow",
                            vector,
                        ),
                    )
            admin.execute("RESET ROLE")

    def connection_for(principal: str) -> str:
        app_params = conninfo_to_dict(database_connections["app"])
        app_params["user"] = principal
        app_params["password"] = credentials[principal]
        return make_conninfo(**app_params)

    try:
        yield SchemaRows(
            admin_conninfo=admin_conninfo,
            app_conninfo=database_connections["app"],
            marker=marker,
            roles=roles,
            connection_for=connection_for,
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
                    """
                    DELETE FROM efm_private.tenant_principals
                    WHERE principal_name = ANY(%s)
                    """,
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


def connect_as_efm_app(conninfo: str) -> psycopg.Connection:
    conn = psycopg.connect(conninfo)
    conn.execute("SET ROLE efm_app")
    return conn


def set_context(
    conn: psycopg.Connection,
    tenant_id: str,
    allowed_roles: str,
) -> None:
    conn.execute("SELECT set_config('app.current_tenant_id', %s, true)", (tenant_id,))
    conn.execute("SELECT set_config('app.allowed_roles', %s, true)", (allowed_roles,))


@pytest.mark.isolation
def test_unmapped_efm_app_sees_no_rows_without_settings(
    schema_rows: SchemaRows,
) -> None:
    with connect_as_efm_app(schema_rows.app_conninfo) as conn:
        with conn.transaction():
            counts = {
                table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in (
                    "documents_registry",
                    "parent_documents",
                    "document_chunks",
                )
            }

    assert counts == {
        "documents_registry": 0,
        "parent_documents": 0,
        "document_chunks": 0,
    }


@pytest.mark.isolation
def test_tenant_alpha_sees_only_its_rows(schema_rows: SchemaRows) -> None:
    marker = schema_rows.marker
    role = schema_rows.roles["alpha"]
    with connect_as_efm_app(schema_rows.connection_for(role)) as conn:
        with conn.transaction():
            set_context(conn, "tenant_alpha", "all,engineer")
            visible = {
                "documents_registry": conn.execute(
                    """SELECT DISTINCT tenant_id FROM documents_registry
                       WHERE source_name LIKE %s""",
                    (f"{marker}:%",),
                ).fetchall(),
                "parent_documents": conn.execute(
                    """SELECT DISTINCT tenant_id FROM parent_documents
                       WHERE content LIKE %s""",
                    (f"{marker}:%",),
                ).fetchall(),
                "document_chunks": conn.execute(
                    """SELECT DISTINCT tenant_id FROM document_chunks
                       WHERE content LIKE %s""",
                    (f"{marker}:%",),
                ).fetchall(),
            }

    assert visible == {
        "documents_registry": [("tenant_alpha",)],
        "parent_documents": [("tenant_alpha",)],
        "document_chunks": [("tenant_alpha",)],
    }


@pytest.mark.isolation
def test_insert_for_another_tenant_is_rejected(schema_rows: SchemaRows) -> None:
    role = schema_rows.roles["alpha"]
    with connect_as_efm_app(schema_rows.connection_for(role)) as conn:
        with pytest.raises(InsufficientPrivilege, match="row-level security"):
            with conn.transaction():
                set_context(conn, "tenant_alpha", "all,engineer")
                conn.execute(
                    """
                    INSERT INTO documents_registry
                        (tenant_id, source_name, sha256, version)
                    VALUES ('tenant_beta', %s, %s, 1)
                    """,
                    (
                        f"{schema_rows.marker}:rejected",
                        "0" * 64,
                    ),
                )


@pytest.mark.isolation
def test_engineer_rows_hidden_when_only_all_role_is_allowed(
    schema_rows: SchemaRows,
) -> None:
    marker = schema_rows.marker
    role = schema_rows.roles["alpha_all"]
    with connect_as_efm_app(schema_rows.connection_for(role)) as conn:
        with conn.transaction():
            set_context(conn, "tenant_alpha", "all")
            roles = {
                "documents_registry": conn.execute(
                    """SELECT DISTINCT access_role FROM documents_registry
                       WHERE source_name LIKE %s""",
                    (f"{marker}:tenant_alpha:%",),
                ).fetchall(),
                "parent_documents": conn.execute(
                    """SELECT DISTINCT access_role FROM parent_documents
                       WHERE content LIKE %s""",
                    (f"{marker}:tenant_alpha:%",),
                ).fetchall(),
                "document_chunks": conn.execute(
                    """SELECT DISTINCT access_role FROM document_chunks
                       WHERE content LIKE %s""",
                    (f"{marker}:tenant_alpha:%",),
                ).fetchall(),
            }

    assert roles == {
        "documents_registry": [("all",)],
        "parent_documents": [("all",)],
        "document_chunks": [("all",)],
    }


@pytest.mark.isolation
def test_efm_app_cannot_alter_tables_or_disable_rls(
    schema_rows: SchemaRows,
) -> None:
    role = schema_rows.roles["alpha"]
    with connect_as_efm_app(schema_rows.connection_for(role)) as conn:
        with pytest.raises(InsufficientPrivilege):
            with conn.transaction():
                conn.execute("ALTER TABLE documents_registry ADD COLUMN forbidden text")

        with pytest.raises(InsufficientPrivilege):
            with conn.transaction():
                conn.execute(
                    "ALTER TABLE documents_registry DISABLE ROW LEVEL SECURITY"
                )


@pytest.mark.isolation
def test_hybrid_search_code_is_scoped_to_each_tenant(schema_rows: SchemaRows) -> None:
    from app.retrieval.hybrid_search import search

    vector = [1.0] + [0.0] * 1023
    results_by_tenant = {}
    for tenant, principal in (
        ("tenant_alpha", schema_rows.roles["alpha"]),
        ("tenant_beta", schema_rows.roles["beta"]),
    ):
        with connect_as_efm_app(schema_rows.connection_for(principal)) as conn:
            results_by_tenant[tenant] = search(
                conn,
                tenant,
                ["all", "engineer"],
                "ERR-409B",
                vector,
            )

    for tenant, results in results_by_tenant.items():
        assert results
        assert all(tenant in result.doc_name for result in results)
        assert all("ERR-409B" in result.child_text for result in results)
