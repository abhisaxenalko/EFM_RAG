from __future__ import annotations

import re
import secrets
from dataclasses import dataclass

import psycopg
from psycopg import sql
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import get_settings

SMALL_TENANT_ROWS = 200
LARGE_TENANT_ROWS = 20_000


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    postgres_user: str
    postgres_password: SecretStr
    postgres_db: str


@dataclass(frozen=True)
class Measurement:
    ef_search: int
    iterative_scan: bool
    returned: int
    recall: float
    uses_hnsw: bool


def _admin_conninfo() -> str:
    settings = DatabaseSettings()
    params = psycopg.conninfo.conninfo_to_dict(settings.database_url)
    params.update(
        user=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
        dbname=settings.postgres_db,
        connect_timeout=10,
    )
    return psycopg.conninfo.make_conninfo(**params)


def _query_vector() -> str:
    return "[" + ",".join(["1"] + ["0"] * 1023) + "]"


def _check_vector_extension(conn: psycopg.Connection) -> tuple[int, ...]:
    row = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    if row is None:
        raise RuntimeError("The pgvector extension is not installed in this database")
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", row[0])
    if match is None:
        raise RuntimeError(f"Cannot parse pgvector version: {row[0]}")
    return tuple(int(part or 0) for part in match.groups())


def _create_benchmark_data(
    conn: psycopg.Connection,
    schema_name: str,
) -> tuple[str, str, str]:
    table_name = "chunks"
    schema = sql.Identifier(schema_name)
    table = sql.Identifier(schema_name, table_name)
    small_tenant = f"bench_small_{secrets.token_hex(6)}"
    large_tenant = f"bench_large_{secrets.token_hex(6)}"

    with conn.transaction():
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        conn.execute(
            sql.SQL(
                "CREATE TABLE {} ("
                "id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                "tenant_id varchar(64) NOT NULL, "
                "embedding vector(1024) NOT NULL)"
            ).format(table)
        )
        conn.execute(
            sql.SQL("ALTER TABLE {} ENABLE ROW LEVEL SECURITY").format(table)
        )
        conn.execute(
            sql.SQL("ALTER TABLE {} FORCE ROW LEVEL SECURITY").format(table)
        )
        conn.execute(
            sql.SQL(
                "CREATE POLICY bench_tenant_filter ON {} FOR SELECT TO efm_app "
                "USING (tenant_id = current_setting('app.current_tenant_id', true))"
            ).format(table)
        )
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO efm_app").format(schema))
        conn.execute(sql.SQL("GRANT SELECT ON {} TO efm_app").format(table))

        # The large tenant's vectors cluster around the query, while the small
        # tenant's vectors are farther away. This makes post-scan RLS filtering
        # visible without relying on any real document or embedding data.
        insert_rows = sql.SQL(
            "INSERT INTO {} (tenant_id, embedding) "
            "SELECT %s, (ARRAY[x::real, sqrt(1.0 - x*x)::real] "
            "|| array_fill(0::real, ARRAY[1022]))::vector "
            "FROM (SELECT 0.10 + 0.60 * ((n - 1)::double precision / %s) AS x "
            "FROM generate_series(1, %s) AS n) AS small_vectors"
        ).format(table)
        conn.execute(
            insert_rows,
            (small_tenant, SMALL_TENANT_ROWS - 1, SMALL_TENANT_ROWS),
        )

        insert_large = sql.SQL(
            "INSERT INTO {} (tenant_id, embedding) "
            "SELECT %s, (ARRAY[x::real, sqrt(1.0 - x*x)::real] "
            "|| array_fill(0::real, ARRAY[1022]))::vector "
            "FROM (SELECT 0.95 + 0.049 * ((n - 1)::double precision / %s) AS x "
            "FROM generate_series(1, %s) AS n) AS large_vectors"
        ).format(table)
        conn.execute(
            insert_large,
            (large_tenant, LARGE_TENANT_ROWS - 1, LARGE_TENANT_ROWS),
        )
        conn.execute(
            sql.SQL(
                "CREATE INDEX {} ON {} USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            ).format(sql.Identifier(f"{schema_name}_chunks_hnsw"), table)
        )

    return table_name, small_tenant, large_tenant


def _measure(
    conn: psycopg.Connection,
    table: sql.Composable,
    tenant_id: str,
    query_vector: str,
    top_k: int,
    ef_search: int,
    iterative_scan: bool,
    iterative_supported: bool,
) -> tuple[Measurement, set[int]]:
    if iterative_scan and not iterative_supported:
        raise ValueError("Iterative HNSW scans are not supported by this pgvector")

    query = sql.SQL(
        "SELECT id FROM {} ORDER BY embedding <=> %s::vector LIMIT %s"
    ).format(table)
    with conn.transaction():
        conn.execute(
            "SELECT set_config('app.current_tenant_id', %s, true)", (tenant_id,)
        )
        # The benchmark is specifically comparing HNSW behavior. Keep the exact
        # baseline separate and force the approximate branch onto its HNSW index.
        conn.execute("SELECT set_config('enable_seqscan', %s, true)", ("off",))
        conn.execute(
            "SELECT set_config('hnsw.ef_search', %s, true), "
            "set_config('hnsw.iterative_scan', %s, true)",
            (str(ef_search), "relaxed_order" if iterative_scan else "off"),
        )
        plan_rows = conn.execute(
            sql.SQL("EXPLAIN (FORMAT TEXT) ") + query,
            (query_vector, top_k),
        ).fetchall()
        plan = "\n".join(row[0] for row in plan_rows)
        uses_hnsw = "hnsw" in plan.lower()
        if not uses_hnsw:
            raise RuntimeError(f"Expected an HNSW index scan, got plan: {plan}")
        rows = conn.execute(query, (query_vector, top_k)).fetchall()
    return Measurement(
        ef_search=ef_search,
        iterative_scan=iterative_scan,
        returned=len(rows),
        recall=0.0,
        uses_hnsw=uses_hnsw,
    ), {row[0] for row in rows}


def run() -> list[Measurement]:
    database = DatabaseSettings()
    admin_conninfo = _admin_conninfo()
    schema_name = f"hnsw_recall_{secrets.token_hex(6)}"
    measurements: list[Measurement] = []
    try:
        with psycopg.connect(admin_conninfo) as admin:
            extension_version = _check_vector_extension(admin)
            table_name, small_tenant, large_tenant = _create_benchmark_data(
                admin,
                schema_name,
            )

        table = sql.Identifier(schema_name, table_name)
        query_vector = _query_vector()
        top_k = get_settings().k_each
        iterative_supported = extension_version >= (0, 8, 0)
        with psycopg.connect(database.database_url) as app:
            role_flags = app.execute(
                "SELECT rolsuper, rolbypassrls FROM pg_roles "
                "WHERE rolname = current_user"
            ).fetchone()
            if role_flags is None or role_flags[0] or role_flags[1]:
                raise RuntimeError(
                    "DATABASE_URL must connect as a non-superuser with NOBYPASSRLS"
                )
            with app.transaction():
                app.execute(
                    "SELECT set_config('app.current_tenant_id', %s, true)",
                    (small_tenant,),
                )
                app.execute(
                    "SELECT set_config('enable_indexscan', %s, true)", ("off",)
                )
                exact_rows = app.execute(
                    sql.SQL(
                        "SELECT id FROM {} "
                        "ORDER BY embedding <=> %s::vector LIMIT %s"
                    ).format(table),
                    (query_vector, top_k),
                ).fetchall()
            exact_ids = {row[0] for row in exact_rows}
            if not exact_ids:
                raise RuntimeError("Exact scan returned no small-tenant results")

            for ef_search in (40, 100, 200):
                for iterative_scan in (False, True):
                    if iterative_scan and not iterative_supported:
                        continue
                    result, result_ids = _measure(
                        app,
                        table,
                        small_tenant,
                        query_vector,
                        top_k,
                        ef_search,
                        iterative_scan,
                        iterative_supported,
                    )
                    measurements.append(
                        Measurement(
                            ef_search=result.ef_search,
                            iterative_scan=result.iterative_scan,
                            returned=result.returned,
                            recall=len(result_ids & exact_ids) / len(exact_ids),
                            uses_hnsw=result.uses_hnsw,
                        )
                    )
    finally:
        # The benchmark table, synthetic rows, policy, and temporary HNSW index
        # all live in this disposable schema and are removed after measurement.
        with psycopg.connect(admin_conninfo) as admin:
            admin.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema_name)
                )
            )

    print(
        f"Synthetic tenant sizes: small={SMALL_TENANT_ROWS}, "
        f"large={LARGE_TENANT_ROWS}"
    )
    scan_support = "supported" if iterative_supported else "unsupported"
    print(f"Top K: {top_k}; pgvector iterative scan: {scan_support}")
    print("ef_search | iterative_scan | dense_results | exact_recall | HNSW plan")
    for result in measurements:
        print(
            f"{result.ef_search:9d} | "
            f"{'relaxed_order' if result.iterative_scan else 'off':14s} | "
            f"{result.returned:13d} | "
            f"{result.recall:12.3f} | "
            f"{'yes' if result.uses_hnsw else 'no'}"
        )

    complete = [
        row
        for row in measurements
        if row.returned >= top_k and row.recall >= 1.0 and row.uses_hnsw
    ]
    if complete:
        chosen = min(
            complete,
            key=lambda row: (row.ef_search, row.iterative_scan),
        )
        mode = (
            "relaxed_order iterative scan"
            if chosen.iterative_scan
            else "no iterative scan"
        )
        print(
            f"Chosen: ef_search={chosen.ef_search} with {mode}, because it returns "
            "K results with full exact recall using the lowest search setting tested."
        )
    elif measurements:
        chosen = max(
            measurements,
            key=lambda row: (row.recall, row.returned, -row.ef_search),
        )
        mode = (
            "relaxed_order iterative scan"
            if chosen.iterative_scan
            else "no iterative scan"
        )
        print(
            f"Chosen: ef_search={chosen.ef_search} with {mode}, because it had the "
            "best measured exact recall and result count."
        )
    return measurements


if __name__ == "__main__":
    run()
