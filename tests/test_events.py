from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.api.events import (
    CitationsEvent,
    ErrorEvent,
    FinalEvent,
    NodeEvent,
    TokenEvent,
    to_sse,
)


@pytest.mark.parametrize(
    "event,expected_type",
    [
        (NodeEvent(id=1, label="Retrieving..."), "node"),
        (TokenEvent(id=2, text="Answer text only."), "token"),
        (
            CitationsEvent(
                id=3,
                citations=[{"doc_name": "manual.pdf", "page": 2}],
            ),
            "citations",
        ),
        (
            FinalEvent(
                id=4, answer="Verified answer.", abstained=False, cache_hit=True
            ),
            "final",
        ),
        (ErrorEvent(id=5), "error"),
    ],
)
def test_events_serialize_as_sse(event, expected_type):
    frame = to_sse(event)
    lines = frame.splitlines()
    assert lines[0] == f"id: {event.id}"
    assert lines[1] == f"event: {expected_type}"
    assert json.loads(lines[2].removeprefix("data: ")) == event.model_dump(mode="json")


@pytest.mark.parametrize(
    "event_type,fields",
    [
        (NodeEvent, {"label": "Retrieving..."}),
        (TokenEvent, {"text": "Answer"}),
        (CitationsEvent, {"citations": []}),
        (FinalEvent, {"answer": "Answer", "abstained": False, "cache_hit": False}),
        (ErrorEvent, {}),
    ],
)
@pytest.mark.parametrize(
    "forbidden",
    ["grader_score", "prompt", "rewritten_query", "stack_trace", "tenant_id"],
)
def test_forbidden_fields_cannot_be_constructed(event_type, fields, forbidden):
    with pytest.raises(ValidationError):
        event_type(id=1, **fields, **{forbidden: "must not be exposed"})
