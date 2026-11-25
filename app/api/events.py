from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from app.models.schemas import Citation


class EventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: int = Field(gt=0)


class NodeEvent(EventPayload):
    type: Literal["node"] = "node"
    label: str


class TokenEvent(EventPayload):
    type: Literal["token"] = "token"
    text: str


class CitationsEvent(EventPayload):
    type: Literal["citations"] = "citations"
    citations: list[Citation]


class FinalEvent(EventPayload):
    type: Literal["final"] = "final"
    answer: str
    abstained: bool
    cache_hit: bool


class ErrorEvent(EventPayload):
    type: Literal["error"] = "error"
    message: Literal["The request could not be completed. Please try again."] = (
        "The request could not be completed. Please try again."
    )


SSEEvent: TypeAlias = NodeEvent | TokenEvent | CitationsEvent | FinalEvent | ErrorEvent


def to_sse(event: SSEEvent) -> str:
    """Serialize a validated browser event as one SSE frame."""
    if not isinstance(
        event, (NodeEvent, TokenEvent, CitationsEvent, FinalEvent, ErrorEvent)
    ):
        raise TypeError("event must be a typed SSE event model")
    return f"id: {event.id}\nevent: {event.type}\ndata: {event.model_dump_json()}\n\n"
