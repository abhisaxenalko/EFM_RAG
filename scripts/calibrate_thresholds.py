from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import get_settings
from app.models.embeddings import embed
from app.retrieval.hybrid_search import search
from app.retrieval.reranker import rerank

DEFAULT_QUESTIONS = (
    Path(__file__).resolve().parents[1] / "eval/retrieval_questions.jsonl"
)
DEFAULT_HIGH = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85)
DEFAULT_LOW = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45)


class AdminSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str
    postgres_password: SecretStr
    postgres_db: str


def _thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "thresholds must be comma-separated numbers"
        ) from exc
    if not thresholds or any(not 0 <= item <= 1 for item in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be between 0 and 1")
    return thresholds


def _questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read evaluation questions at {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object on {path}:{line_number}")
        query = row.get("question", row.get("query", row.get("text")))
        answerable = row.get("answerable", row.get("is_answerable"))
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Missing question/query on {path}:{line_number}")
        if not isinstance(answerable, bool):
            raise ValueError(
                f"Missing boolean answerable label on {path}:{line_number}"
            )
        row["_query"] = query.strip()
        row["_answerable"] = answerable
        rows.append(row)
    if not rows:
        raise ValueError(f"No evaluation questions found in {path}")
    return rows


def _roles(row: dict[str, Any]) -> list[str]:
    roles = row.get("roles", ["all", "engineer"])
    if isinstance(roles, str):
        roles = [roles]
    if (
        not isinstance(roles, list)
        or not roles
        or any(not isinstance(role, str) for role in roles)
    ):
        raise ValueError("Each question's roles must be a non-empty string list")
    return roles


@contextmanager
def _tenant_connections(
    questions: Sequence[dict[str, Any]],
) -> Any:
    settings = get_settings()
    admin_settings = AdminSettings()
    params = conninfo_to_dict(settings.database_url)
    params.update(
        user=admin_settings.postgres_user,
        password=admin_settings.postgres_password.get_secret_value(),
        dbname=admin_settings.postgres_db,
        connect_timeout=10,
    )
    admin = psycopg.connect(make_conninfo(**params), autocommit=True)
    tenants: dict[str, set[str]] = defaultdict(set)
    for row in questions:
        tenant = row.get(
            "tenant_id", os.getenv("CALIBRATION_TENANT_ID", "tenant_alpha")
        )
        if not isinstance(tenant, str) or not tenant:
            raise ValueError("tenant_id must be a non-empty string")
        tenants[tenant].update(_roles(row))
    logins: dict[str, tuple[str, str]] = {}
    connections: dict[str, psycopg.Connection] = {}
    try:
        with admin.transaction():
            for tenant_id, roles in tenants.items():
                suffix = secrets.token_hex(5)
                role_name = f"efm_calibration_{suffix}"
                password = secrets.token_urlsafe(32)
                logins[tenant_id] = (role_name, password)
                admin.execute(
                    sql.SQL(
                        "CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD {}"
                    ).format(sql.Identifier(role_name), sql.Literal(password))
                )
                admin.execute(
                    sql.SQL("GRANT efm_app TO {}").format(sql.Identifier(role_name))
                )
                admin.execute("SET ROLE efm_owner")
                admin.execute(
                    "INSERT INTO efm_private.tenant_principals "
                    "(principal_name, tenant_id, allowed_roles) VALUES (%s, %s, %s)",
                    (role_name, tenant_id, sorted(roles)),
                )
                admin.execute("RESET ROLE")
        base = conninfo_to_dict(settings.database_url)
        for tenant_id, (role_name, password) in logins.items():
            conn_params = {
                **base,
                "user": role_name,
                "password": password,
                "connect_timeout": 10,
            }
            conn = psycopg.connect(make_conninfo(**conn_params))
            conn.execute("SET ROLE efm_app")
            conn.commit()
            connections[tenant_id] = conn
        yield connections
    finally:
        for conn in connections.values():
            conn.close()
        with admin.transaction():
            admin.execute("SET ROLE efm_owner")
            admin.execute(
                "DELETE FROM efm_private.tenant_principals "
                "WHERE principal_name = ANY(%s)",
                ([role_name for role_name, _ in logins.values()],),
            )
            admin.execute("RESET ROLE")
        for role_name, _ in logins.values():
            admin.execute(
                sql.SQL("REVOKE efm_app FROM {}").format(sql.Identifier(role_name))
            )
            admin.execute(
                sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role_name))
            )
        admin.close()


def _question_scores(path: Path) -> list[tuple[bool, float | None]]:
    if os.getenv("RERANKER", "cross-encoder").strip().lower() == "fake":
        raise ValueError("Calibration requires the real reranker; unset RERANKER=fake")
    if os.getenv("EMBEDDER", "sentence-transformers").strip().lower() == "fake":
        raise ValueError(
            "Calibration requires real query embeddings; unset EMBEDDER=fake"
        )

    questions = _questions(path)
    result: list[tuple[bool, float | None]] = []
    with _tenant_connections(questions) as connections:
        for row in questions:
            tenant_id = row.get(
                "tenant_id", os.getenv("CALIBRATION_TENANT_ID", "tenant_alpha")
            )
            if not isinstance(tenant_id, str) or not tenant_id:
                raise ValueError("tenant_id must be a non-empty string")
            query = row["_query"]
            candidates = search(
                connections[tenant_id], tenant_id, _roles(row), query, embed([query])[0]
            )
            top = rerank(query, candidates, top_n=1)
            score = top[0].rerank_score if top else None
            result.append((row["_answerable"], score))
    return result


def _distribution(label: str, scores: Sequence[float | None]) -> None:
    available = [score for score in scores if score is not None]
    if not available:
        print(f"{label}: n=0 (no retrieved candidates)")
        return
    print(
        f"{label}: n={len(available)} min={min(available):.4f} "
        f"median={statistics.median(available):.4f} max={max(available):.4f}"
    )


def _print_table(
    scores: Sequence[tuple[bool, float | None]],
    high_values: Sequence[float],
    low_values: Sequence[float],
) -> None:
    positives = sum(answerable for answerable, _ in scores)
    if not any(score is not None for _, score in scores):
        print("\nNo reranked candidates; precision and recall are unavailable.")
        return
    print(
        "\nRelevant means top rerank score >= GRADE_HIGH; "
        "middle-band cases are not relevant."
    )
    print("GRADE_HIGH GRADE_LOW  predicted_relevant ambiguous precision recall")
    for high in high_values:
        for low in low_values:
            if low >= high:
                continue
            predicted = [
                score for _, score in scores if score is not None and score >= high
            ]
            true_positive = sum(
                answerable and score is not None and score >= high
                for answerable, score in scores
            )
            ambiguous = sum(
                score is not None and low < score < high for _, score in scores
            )
            precision = true_positive / len(predicted) if predicted else 0.0
            recall = true_positive / positives if positives else 0.0
            print(
                f"{high:10.2f} {low:9.2f} {len(predicted):19d} {ambiguous:9d} "
                f"{precision:9.3f} {recall:6.3f}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate reranker relevance thresholds without LLM calls."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--high", type=_thresholds, default=DEFAULT_HIGH)
    parser.add_argument("--low", type=_thresholds, default=DEFAULT_LOW)
    args = parser.parse_args(argv)
    try:
        scores = _question_scores(args.questions)
    except (OSError, ValueError, psycopg.Error) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1

    print(f"Questions: {len(scores)}")
    _distribution("Answerable", [score for answerable, score in scores if answerable])
    _distribution(
        "Unanswerable", [score for answerable, score in scores if not answerable]
    )
    _print_table(scores, args.high, args.low)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
