from __future__ import annotations

import pytest

from app.models.schemas import Answer, Usage
from scripts.smoke_run import QUESTIONS, _invoke_timed, _row


class _OverBudgetGraph:
    async def astream(self, state, *, stream_mode):
        assert stream_mode == "updates"
        yield {
            "generate": {
                "answer": Answer(text="Unverified output"),
                "usage_log": [Usage(model="test", cost_usd=0.51)],
            }
        }
        yield {"guardrail_verify": {"verified": None}}


def test_smoke_suite_has_ten_questions_and_one_unanswerable():
    assert len(QUESTIONS) == 10
    assert sum(not answerable for _, _, answerable in QUESTIONS) == 1


@pytest.mark.asyncio
async def test_cost_cap_stops_graph_before_next_node_and_hides_unverified_answer():
    state = {
        "usage_log": [],
        "answer": None,
        "abstained": False,
        "verified": None,
        "retry_count": 0,
        "cache_hit": False,
    }
    current, latencies, cost, interrupted = await _invoke_timed(
        _OverBudgetGraph(), state, cost_before_run=0
    )

    assert interrupted is True
    assert cost == pytest.approx(0.51)
    assert "generate" in latencies
    assert "guardrail_verify" not in latencies
    row = _row("tenant_alpha", "q", "question", 1, current, latencies, cost, True)
    assert row["answer"] == ""
    assert row["interrupted_by_cost_cap"] is True
