from __future__ import annotations

import contextvars
import json
import logging
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.models.schemas import Usage

logger = logging.getLogger(__name__)


class EfmLLMError(RuntimeError):
    """Provider or response error with safe model context."""

    def __init__(self, model: str, reason: str) -> None:
        self.model = model
        self.reason = reason
        super().__init__(f"LLM model {model!r} failed: {reason}")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    base_url: str
    fallback: str | None = None
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0


def load_models(path: str | Path | None = None) -> dict[str, ModelConfig]:
    """Load friendly model aliases from the repository's models.yaml."""
    catalog = (
        Path(path) if path else Path(__file__).resolve().parents[1] / "models.yaml"
    )
    try:
        raw = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        entries = raw["models"]
        models = {
            name: ModelConfig(name=name, **config) for name, config in entries.items()
        }
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not load model catalog {catalog}: {exc}") from exc
    for config in models.values():
        if not config.model or not config.base_url:
            raise ValueError(f"Model {config.name!r} needs model and base_url")
        if config.fallback and config.fallback not in models:
            raise ValueError(
                f"Model {config.name!r} has unknown fallback {config.fallback!r}"
            )
    return models


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
        structured_responses: Mapping[type[BaseModel], Iterable[StructuredScriptItem]]
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
                f"FakeLLM script exhausted for structured schema {schema.__name__}"
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


class OpenAICompatClient:
    """Async OpenAI chat-completions client with one retry and one fallback."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        *,
        base_url: str | None = None,
        catalog_path: str | Path | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.models = load_models(catalog_path)
        if model_name not in self.models:
            raise ValueError(f"Unknown LLM_MODEL alias: {model_name!r}")
        self.model_name = model_name
        self.api_key = api_key
        self.base_url_override = base_url
        self.catalog_path = Path(catalog_path) if catalog_path else None
        self.http_client = http_client
        self.timeout = timeout
        self._aliases: dict[str, OpenAICompatClient] = {model_name: self}
        self._usage_entries: contextvars.ContextVar[list[Usage]] = (
            contextvars.ContextVar(f"llm_usage_{id(self)}", default=[])
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        **kwargs: Any,
    ) -> OpenAICompatClient:
        return cls(
            settings.llm_model,
            settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            **kwargs,
        )

    def consume_usage_log(self) -> list[Usage]:
        """Return this async task's provider-attempt and fallback entries."""
        entries = self._usage_entries.get()
        self._usage_entries.set([])
        return entries

    def for_alias(self, model_name: str) -> OpenAICompatClient:
        """Return a client bound to a catalog alias, reusing it per process."""
        if model_name not in self.models:
            raise ValueError(f"Unknown LLM model alias: {model_name!r}")
        if model_name not in self._aliases:
            self._aliases[model_name] = OpenAICompatClient(
                model_name,
                self.api_key,
                base_url=self.base_url_override,
                catalog_path=self.catalog_path,
                http_client=self.http_client,
                timeout=self.timeout,
            )
        return self._aliases[model_name]

    async def _post(
        self,
        config: ModelConfig,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> tuple[str, Usage]:
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
        }
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}
        base_url = (
            self.base_url_override
            if config.name == self.model_name and self.base_url_override
            else config.base_url
        )
        url = base_url.rstrip("/")
        started = time.perf_counter()
        try:
            if self.http_client is None:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                    )
            else:
                response = await self.http_client.post(
                    f"{url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("provider returned non-text message content")
            prompt_tokens = int(data.get("usage", {}).get("prompt_tokens", 0))
            completion_tokens = int(data.get("usage", {}).get("completion_tokens", 0))
            cost = (
                prompt_tokens * config.input_cost_per_million
                + completion_tokens * config.output_cost_per_million
            ) / 1_000_000
            usage = Usage(
                model=config.model,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cost_usd=cost,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            logger.info(
                "llm_call model=%s input_tokens=%d output_tokens=%d latency_ms=%.2f",
                config.model,
                prompt_tokens,
                completion_tokens,
                usage.latency_ms,
            )
            return content, usage
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            usage = Usage(model=config.model, latency_ms=latency)
            self._usage_entries.set([*self._usage_entries.get(), usage])
            logger.warning(
                "llm_call_failed model=%s input_tokens=0 output_tokens=0 "
                "latency_ms=%.2f reason=%s",
                config.model,
                latency,
                type(exc).__name__,
            )
            raise EfmLLMError(config.model, str(exc)) from exc

    async def _complete_model(
        self,
        config: ModelConfig,
        system: str,
        user: str,
        schema: type[BaseModel] | None,
    ) -> tuple[str, Usage]:
        if schema is not None:
            system = (
                f"{system}\nReturn only a JSON object matching this JSON Schema:\n"
                f"{json.dumps(schema.model_json_schema())}"
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                content, usage = await self._post(
                    config,
                    messages,
                    json_schema=schema.model_json_schema()
                    if schema is not None
                    else None,
                )
                self._usage_entries.set([*self._usage_entries.get(), usage])
                if schema is None:
                    return content, usage
                try:
                    parsed = schema.model_validate_json(content)
                except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                    last_error = exc
                    if attempt == 0:
                        messages = [
                            *messages,
                            {"role": "assistant", "content": content},
                            {
                                "role": "user",
                                "content": (
                                    "Repair the response. Return JSON matching schema: "
                                    f"{json.dumps(schema.model_json_schema())}. "
                                    f"Validation error: {exc}"
                                ),
                            },
                        ]
                        continue
                    raise EfmLLMError(
                        config.model, f"invalid structured response: {exc}"
                    ) from exc
                return parsed.model_dump_json(), usage
            except EfmLLMError as exc:
                last_error = exc
                if attempt == 1:
                    break
            except Exception as exc:
                last_error = exc
                if attempt == 1:
                    break
        raise EfmLLMError(config.model, str(last_error or "request failed"))

    async def _complete(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel] | None,
    ) -> tuple[str, Usage]:
        self._usage_entries.set([])
        primary = self.models[self.model_name]
        try:
            return await self._complete_model(primary, system, user, schema)
        except EfmLLMError as primary_error:
            if primary.fallback is None:
                raise primary_error
            fallback = self.models[primary.fallback]
            marker = Usage(model=fallback.model, event="model_fallback")
            self._usage_entries.set([*self._usage_entries.get(), marker])
            try:
                return await self._complete_model(fallback, system, user, schema)
            except EfmLLMError as fallback_error:
                raise EfmLLMError(
                    f"{primary.model} -> {fallback.model}",
                    f"primary failed ({primary_error.reason}); fallback failed "
                    f"({fallback_error.reason})",
                ) from fallback_error

    async def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[BaseModel],
    ) -> tuple[BaseModel, Usage]:
        content, usage = await self._complete(system=system, user=user, schema=schema)
        return schema.model_validate_json(content), usage

    async def complete_text(
        self,
        *,
        system: str,
        user: str,
    ) -> tuple[str, Usage]:
        return await self._complete(system=system, user=user, schema=None)
