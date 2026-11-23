from __future__ import annotations

from app.graph.state import initial_state
from app.models.schemas import Answer, Citation, GradeOutput, Usage, Verdict


def _round_trip(model):
    assert type(model).model_validate_json(model.model_dump_json()) == model


def test_grade_output_json_round_trip() -> None:
    _round_trip(GradeOutput(score="ambiguous", missing_aspect="required pressure"))


def test_citation_json_round_trip() -> None:
    _round_trip(Citation(doc_name="Chiller Manual.pdf", page=12))


def test_answer_json_round_trip() -> None:
    _round_trip(
        Answer(
            text="Replace the valve.",
            citations=[Citation(doc_name="Chiller Manual.pdf", page=12)],
        )
    )


def test_verdict_json_round_trip() -> None:
    _round_trip(
        Verdict(supported=False, unsupported_claims=["The valve lasts ten years."])
    )


def test_usage_json_round_trip_contains_only_usage_metadata() -> None:
    usage = Usage(
        model="local-model",
        input_tokens=42,
        output_tokens=18,
        cost_usd=0.002,
        latency_ms=35.4,
    )
    _round_trip(usage)
    assert set(usage.model_dump()) == {
        "model",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_ms",
    }


def test_initial_crag_state_has_safe_fresh_defaults() -> None:
    state = initial_state("tenant_alpha", ["all"], "Why is the chiller offline?")
    assert state["trace"] == []
    assert state["usage_log"] == []
    assert state["docs"] == []
    assert state["retry_count"] == state["regen_count"] == 0
    assert state["answer"] is None
    assert state["cache_hit"] is False
