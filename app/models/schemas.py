from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GradeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: Literal["relevant", "ambiguous", "irrelevant"]
    missing_aspect: str = ""


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_name: str
    page: int = Field(ge=1)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = ""
    citations: list[Citation] = Field(default_factory=list)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supported: bool = False
    unsupported_claims: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    """Token and timing metadata only; retrieved or generated text is excluded."""

    model_config = ConfigDict(extra="forbid")

    model: str = "unknown"
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
