from __future__ import annotations

import asyncio
import importlib.util
import json
import secrets
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import redis
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.cache.semantic_cache import SemanticCache
from app.config import Settings, get_settings
from app.graph.build import build_graph
from app.graph.nodes import NodeDeps
from app.graph.state import initial_state
from app.llm import OpenAICompatClient, load_models
from app.models.embeddings import embed
from app.retrieval.hybrid_search import search
from app.retrieval.reranker import rerank

MAX_TOTAL_COST_USD = 0.50
TENANTS = ("tenant_alpha", "tenant_beta")
ROLES = ["all", "engineer"]
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "tests/fixtures/smoke_output.json"

QUESTIONS = (
    ("sla_emergency", "What is the emergency repair SLA?", True),
    ("sla_priority", "What response time does the SLA set for urgent faults?", True),
    ("fault_code", "Which fault code indicates low flow?", True),
    ("err_409b", "What does ERR-409B indicate?", True),
    ("err_409b_action", "What troubleshooting action is listed for ERR-409B?", True),
    (
        "table_err_409b",
        "In the fault-code table, what description and action are listed for ERR-409B?",
        True,
    ),
    (
        "table_low_flow",
        "According to the fault-code table, which entry corresponds to low flow?",
        True,
    ),
    (
        "table_lookup",
        "Look up ERR-409B in the equipment table and report its row details.",
        True,
    ),
    (
        "unanswerable",
        "What is the private home address of the technician assigned to this site?",
        False,
    ),
    (
        "injection_topic",
        "What does the suspicious instruction-like passage ask a system to do?",
        True,
    ),
)


class SmokeSetupError(RuntimeError):
    """A preflight problem that must stop before model or database work."""


class AdminSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    postgres_user: str
    postgres_password: SecretStr
    postgres_db: str


@dataclass
class TenantConnection:
    tenant_id: str
    role_name: str
    conn: psycopg.Connection


def _admin_conninfo(settings: Settings) -> str:
    admin = AdminSettings()
    params = conninfo_to_dict(settings.database_url)
    params.update(
        user=admin.postgres_user,
        password=admin.postgres_password.get_secret_value(),
        dbname=admin.postgres_db,
        connect_timeout=10,
    )
    return make_conninfo(**params)


@contextmanager
def _tenant_connections(settings: Settings) -> Iterator[dict[str, TenantConnection]]:
    """Create temporary tenant logins and always remove them after the run."""
    admin_conninfo = _admin_conninfo(settings)
    admin = psycopg.connect(admin_conninfo, autocommit=True)
    created: list[tuple[str, str]] = []
    tenant_connections: dict[str, TenantConnection] = {}
    try:
        with admin.transaction():
            for tenant_id in TENANTS:
                tenant_slug = tenant_id.removeprefix("tenant_")
                role_name = f"efm_smoke_{tenant_slug}_{secrets.token_hex(5)}"
                password = secrets.token_urlsafe(32)
                admin.execute(
                    sql.SQL(
                        "CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD {}"
                    ).format(sql.Identifier(role_name), sql.Literal(password))
                )
                created.append((role_name, password))
                admin.execute(
                    sql.SQL("GRANT efm_app TO {}").format(sql.Identifier(role_name))
                )
                admin.execute("SET ROLE efm_owner")
                admin.execute(
                    """
                    INSERT INTO efm_private.tenant_principals
                        (principal_name, tenant_id, allowed_roles)
                    VALUES (%s, %s, %s)
                    """,
                    (role_name, tenant_id, ROLES),
                )
                admin.execute("RESET ROLE")

        connection_params = conninfo_to_dict(settings.database_url)
        for tenant_id, (role_name, password) in zip(TENANTS, created, strict=True):
            params = {
                **connection_params,
                "user": role_name,
                "password": password,
                "connect_timeout": 10,
            }
            conn = psycopg.connect(make_conninfo(**params))
            conn.execute("SET ROLE efm_app")
            conn.commit()
            with conn.transaction():
                conn.execute(
                    "SELECT set_config('app.current_tenant_id', %s, true)",
                    (tenant_id,),
                )
                conn.execute(
                    "SELECT set_config('app.allowed_roles', %s, true)",
                    (",".join(ROLES),),
                )
                active_docs = conn.execute(
                    "SELECT count(*) FROM documents_registry WHERE status = 'active'"
                ).fetchone()[0]
            if active_docs == 0:
                conn.close()
                raise RuntimeError(f"No active fixture documents for {tenant_id}")
            tenant_connections[tenant_id] = TenantConnection(
                tenant_id=tenant_id,
                role_name=role_name,
                conn=conn,
            )
        yield tenant_connections
    finally:
        for tenant in tenant_connections.values():
            tenant.conn.close()
        with admin.transaction():
            admin.execute("SET ROLE efm_owner")
            admin.execute(
                "DELETE FROM efm_private.tenant_principals "
                "WHERE principal_name = ANY(%s)",
                ([role_name for role_name, _ in created],),
            )
            admin.execute("RESET ROLE")
        for role_name, _ in created:
            admin.execute(
                sql.SQL("REVOKE efm_app FROM {}").format(sql.Identifier(role_name))
            )
            admin.execute(
                sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role_name))
            )
        admin.close()


def _check_llm_configuration(settings: Settings) -> None:
    models = load_models()
    api_key = settings.llm_api_key.get_secret_value()
    urls = [
        settings.llm_base_url or models[name].base_url for name in ("fast", "strong")
    ]
    uses_groq = any("api.groq.com" in url for url in urls)
    if uses_groq and (not api_key or api_key.startswith("replace-with")):
        raise RuntimeError("Set a valid LLM_API_KEY in .env before the smoke run")


def _build_deps(
    settings: Settings,
    llm: OpenAICompatClient,
    cache: SemanticCache,
    connection: psycopg.Connection,
) -> NodeDeps:
    def searcher(tenant_id: str, roles: list[str], query: str):
        query_embedding = embed([query])[0]
        return search(connection, tenant_id, roles, query, query_embedding)

    return NodeDeps(
        llm=llm,
        searcher=searcher,
        reranker=rerank,
        settings=settings,
        cache=cache,
        embedder=embed,
    )


def _state_cost(state: dict[str, Any]) -> float:
    return sum(entry.cost_usd for entry in state.get("usage_log", []))


async def _invoke_timed(
    graph: Any,
    state: dict[str, Any],
    *,
    cost_before_run: float,
) -> tuple[dict[str, Any], dict[str, float], float, bool]:
    current = dict(state)
    node_latencies: dict[str, float] = {}
    if cost_before_run >= MAX_TOTAL_COST_USD:
        return current, node_latencies, 0.0, True
    last_event = time.perf_counter()
    budget_exceeded = False
    async for event in graph.astream(state, stream_mode="updates"):
        now = time.perf_counter()
        if isinstance(event, dict):
            for node_name, update in event.items():
                node_latencies[node_name] = round((now - last_event) * 1000, 2)
                if isinstance(update, dict):
                    current.update(update)
            last_event = now
        if cost_before_run + _state_cost(current) > MAX_TOTAL_COST_USD:
            budget_exceeded = True
            break
    return current, node_latencies, _state_cost(current), budget_exceeded


def _row(
    tenant_id: str,
    question_id: str,
    question: str,
    run_number: int,
    state: dict[str, Any],
    node_latencies: dict[str, float],
    cost_usd: float,
    interrupted: bool,
) -> dict[str, Any]:
    verified = state.get("verified")
    final_answer = (
        state.get("answer")
        if not interrupted
        and (state.get("abstained") or verified and verified.supported)
        else None
    )
    return {
        "tenant": tenant_id,
        "question_id": question_id,
        "question": question,
        "run": run_number,
        "answer": final_answer.text if final_answer else "",
        "citations": [
            citation.model_dump(mode="json")
            for citation in (final_answer.citations if final_answer else [])
        ],
        "abstained": bool(state.get("abstained", False)),
        "retry_count": int(state.get("retry_count", 0)),
        "cache_hit": bool(state.get("cache_hit", False)),
        "node_latencies_ms": node_latencies,
        "cost_usd": round(cost_usd, 8),
        "interrupted_by_cost_cap": interrupted,
    }


def _save_output(payload: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUTPUT_PATH)


async def run_smoke() -> dict[str, Any]:
    settings = get_settings()
    _check_llm_configuration(settings)
    if importlib.util.find_spec("sentence_transformers") is None:
        raise SmokeSetupError(
            "sentence-transformers is required for the configured embedder and reranker"
        )
    total_cost = 0.0
    rows: list[dict[str, Any]] = []
    stopped_reason: str | None = None
    llm = OpenAICompatClient(
        "fast",
        settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
    )
    with _tenant_connections(settings) as tenant_connections:
        with redis.Redis.from_url(settings.redis_url) as redis_client:
            if not redis_client.ping():
                raise RuntimeError("Redis health check failed")
            cache = SemanticCache(
                redis_client,
                cache_sim=settings.cache_sim,
                ttl_seconds=settings.cache_ttl_seconds,
            )
            for tenant_id in TENANTS:
                tenant = tenant_connections[tenant_id]
                deps = _build_deps(settings, llm, cache, tenant.conn)
                graph = build_graph(deps)
                for question_id, question, _ in QUESTIONS:
                    for run_number in (1, 2):
                        (
                            state,
                            node_latencies,
                            run_cost,
                            interrupted,
                        ) = await _invoke_timed(
                            graph,
                            initial_state(tenant_id, ROLES, question),
                            cost_before_run=total_cost,
                        )
                        total_cost += run_cost
                        row = _row(
                            tenant_id,
                            question_id,
                            question,
                            run_number,
                            state,
                            node_latencies,
                            run_cost,
                            interrupted,
                        )
                        rows.append(row)
                        payload = {
                            "status": "running",
                            "generated_at": datetime.now(UTC).isoformat(),
                            "cost_limit_usd": MAX_TOTAL_COST_USD,
                            "total_cost_usd": round(total_cost, 8),
                            "runs": rows,
                        }
                        _save_output(payload)
                        # Windows PowerShell may use a legacy code page; keep
                        # console output ASCII-safe while the artifact stays UTF-8.
                        print(json.dumps(row, ensure_ascii=True), flush=True)
                        if interrupted or total_cost > MAX_TOTAL_COST_USD:
                            stopped_reason = (
                                "Cost limit reached; no further questions were run."
                            )
                            break
                    if stopped_reason:
                        break
                if stopped_reason:
                    break

    first_runs = {
        (row["tenant"], row["question_id"]): row for row in rows if row["run"] == 1
    }
    second_runs = {
        (row["tenant"], row["question_id"]): row for row in rows if row["run"] == 2
    }
    sla_alpha_row = first_runs.get(("tenant_alpha", "sla_emergency"), {})
    sla_beta_row = first_runs.get(("tenant_beta", "sla_emergency"), {})
    unanswerable_ok = all(
        first_runs.get((tenant, "unanswerable"), {}).get("abstained", False)
        for tenant in TENANTS
    )
    second_hits_ok = all(
        second_runs.get((tenant, question_id), {}).get("cache_hit", False)
        for tenant in TENANTS
        for question_id, _, is_answerable in QUESTIONS
        if is_answerable
        and not first_runs.get((tenant, question_id), {}).get("abstained", True)
    )
    unanswerable_not_cached = all(
        not second_runs.get((tenant, "unanswerable"), {}).get("cache_hit", True)
        for tenant in TENANTS
    )
    checks = {
        "ten_questions_per_tenant": all(
            sum(1 for row in rows if row["tenant"] == tenant and row["run"] == 1)
            == len(QUESTIONS)
            for tenant in TENANTS
        ),
        "alpha_beta_sla_answers_differ": bool(
            sla_alpha_row.get("answer")
            and sla_beta_row.get("answer")
            and not sla_alpha_row.get("abstained")
            and not sla_beta_row.get("abstained")
            and sla_alpha_row["answer"] != sla_beta_row["answer"]
        ),
        "unanswerable_abstains": unanswerable_ok,
        "second_run_cache_hits_for_answerable_questions": second_hits_ok,
        "unanswerable_is_not_cached": unanswerable_not_cached,
        "within_cost_limit": total_cost <= MAX_TOTAL_COST_USD,
    }
    passed = all(checks.values()) and stopped_reason is None
    result = {
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(UTC).isoformat(),
        "cost_limit_usd": MAX_TOTAL_COST_USD,
        "total_cost_usd": round(total_cost, 8),
        "checks": checks,
        "stopped_reason": stopped_reason,
        "runs": rows,
    }
    _save_output(result)
    print(json.dumps({"status": result["status"], "checks": checks}, indent=2))
    return result


def main() -> int:
    try:
        result = asyncio.run(run_smoke())
    except SmokeSetupError as exc:
        error = str(exc)
    except Exception as exc:
        error = (
            f"Missing required module: {exc.name}"
            if isinstance(exc, ModuleNotFoundError)
            else f"Smoke run could not complete ({type(exc).__name__})."
        )
    else:
        return 0 if result["status"] == "passed" else 1

    if error:
        result = {
            "status": "failed",
            "generated_at": datetime.now(UTC).isoformat(),
            "cost_limit_usd": MAX_TOTAL_COST_USD,
            "total_cost_usd": 0.0,
            "checks": {},
            "error": error,
            "runs": [],
        }
        _save_output(result)
        print(result["error"], file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
