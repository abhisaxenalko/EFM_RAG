from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from typing import Protocol

from pydantic import BaseModel

from app.models.schemas import Usage


class LLMClient(Protocol):
    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> tuple[BaseModel, Usage]: ...

    async def complete_text(
        self,
        *,
        system: str,
        user: str,
    ) -> tuple[str, Usage]: ...


StructuredScriptItem = (
    BaseModel | Mapping[str, object] | tuple[BaseModel | Mapping[str, object], Usage]
)
TextScriptItem = str | tuple[str, Usage]


class FakeLLM:
    """Scripted, network-free LLM test double that records every prompt."""

    def __init__(
        self,
        structured_responses: Mapping[
            type[BaseModel], Iterable[StructuredScriptItem]
        ]
        | None = None,
        text_responses: Iterable[TextScriptItem] = (),
    ) -> None:
        self._structured = defaultdict(deque)
        if structured_responses is not None:
            for schema, responses in structured_responses.items():
                self._structured[schema].extend(responses)
        self._text = deque(text_responses)
        self.calls: list[tuple[str, str]] = []

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> tuple[BaseModel, Usage]:
        self.calls.append((system, user))
        responses = self._structured.get(schema)
        if not responses:
            raise RuntimeError(
                "FakeLLM script exhausted for structured schema "
                f"{schema.__name__}"
            )

        item = responses.popleft()
        if isinstance(item, tuple):
            response, usage = item
        else:
            response = item
            usage = Usage(model="fake")
        if isinstance(response, schema):
            parsed = response
        else:
            parsed = schema.model_validate(response)
        return parsed, usage

    async def complete_text(
        self,
        *,
        system: str,
        user: str,
    ) -> tuple[str, Usage]:
        self.calls.append((system, user))
        if not self._text:
            raise RuntimeError("FakeLLM script exhausted for text completion")
        item = self._text.popleft()
        if isinstance(item, tuple):
            return item
        return item, Usage(model="fake")
