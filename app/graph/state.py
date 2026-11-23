from __future__ import annotations

from typing import TypedDict

from app.models.schemas import Answer, Citation, GradeOutput, Usage, Verdict
from app.retrieval.hybrid_search import Candidate


class CragState(TypedDict):
    tenant_id: str
    roles: list[str]
    query: str
    rewritten_query: str | None
    docs: list[Candidate]
    grade: GradeOutput | None
    missing_aspect: str
    retry_count: int
    regen_count: int
    answer: Answer | None
    citations: list[Citation]
    verified: Verdict | None
    cache_hit: bool
    abstained: bool
    trace: list[str]
    usage_log: list[Usage]


def initial_state(tenant_id: str, roles: list[str], query: str) -> CragState:
    """Create a fully populated state with fresh mutable defaults."""
    return {
        "tenant_id": tenant_id,
        "roles": list(roles),
        "query": query,
        "rewritten_query": None,
        "docs": [],
        "grade": None,
        "missing_aspect": "",
        "retry_count": 0,
        "regen_count": 0,
        "answer": None,
        "citations": [],
        "verified": None,
        "cache_hit": False,
        "abstained": False,
        "trace": [],
        "usage_log": [],
    }
