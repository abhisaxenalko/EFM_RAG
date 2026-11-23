from __future__ import annotations

import os
import re
import threading
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from app.config import get_settings
from app.retrieval.hybrid_search import Candidate

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_lock = threading.Lock()
_model: Any | None = None
_model_kind: str | None = None


class FakeReranker:
    """Deterministic CI scorer based on query and passage token overlap."""

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        **_: Any,
    ) -> list[float]:
        scores: list[float] = []
        for query, passage in pairs:
            query_tokens = {token.casefold() for token in _TOKEN_RE.findall(query)}
            passage_tokens = {token.casefold() for token in _TOKEN_RE.findall(passage)}
            scores.append(
                len(query_tokens & passage_tokens) / max(len(query_tokens), 1)
            )
        return scores


def _passage_limit() -> int:
    limit = int(os.getenv("RERANKER_MAX_PASSAGE_TOKENS", "384"))
    if limit < 1:
        raise ValueError("RERANKER_MAX_PASSAGE_TOKENS must be at least 1")
    return limit


def _load_model() -> Any:
    if os.getenv("RERANKER", "cross-encoder").strip().lower() == "fake":
        return FakeReranker()

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(
        _MODEL_NAME,
        device=device,
        max_length=_passage_limit() + 128,
    )


def _get_model() -> Any:
    global _model, _model_kind
    kind = os.getenv("RERANKER", "cross-encoder").strip().lower()
    with _lock:
        if _model is None:
            _model = _load_model()
            _model_kind = kind
        elif kind != _model_kind:
            raise RuntimeError("RERANKER cannot change after the model is loaded")
        return _model


def _truncate_passage(model: Any, passage: str, limit: int) -> str:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return " ".join(passage.split()[:limit])
    token_ids = tokenizer.encode(
        passage,
        add_special_tokens=False,
        truncation=True,
        max_length=limit,
    )
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def rerank(
    query: str,
    candidates: Sequence[Candidate],
    top_n: int | None = None,
) -> list[Candidate]:
    """Score candidates, keep the highest scoring child per parent, and sort."""
    if top_n is None:
        top_n = get_settings().k_final
    if top_n < 0:
        raise ValueError("top_n cannot be negative")
    if top_n == 0 or not candidates:
        return []

    model = _get_model()
    passage_limit = _passage_limit()
    batch_size = int(os.getenv("RERANKER_BATCH_SIZE", "16"))
    if batch_size < 1:
        raise ValueError("RERANKER_BATCH_SIZE must be at least 1")

    indexed_scored: list[tuple[int, Candidate]] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        passages = [
            _truncate_passage(model, candidate.child_text, passage_limit)
            for candidate in batch
        ]
        pairs = list(zip([query] * len(batch), passages, strict=True))
        batch_scores = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        if len(batch_scores) != len(batch):
            raise ValueError("The reranker returned a different number of scores")
        scored_batch = zip(batch, batch_scores, strict=True)
        for offset, (candidate, score) in enumerate(scored_batch):
            indexed_scored.append(
                (
                    start + offset,
                    replace(candidate, rerank_score=float(score)),
                )
            )

    best_by_parent: dict[int, tuple[int, Candidate]] = {}
    for index, candidate in indexed_scored:
        current = best_by_parent.get(candidate.parent_id)
        candidate_score = candidate.rerank_score
        current_score = current[1].rerank_score if current is not None else None
        if current is None or (
            candidate_score is not None
            and (current_score is None or candidate_score > current_score)
        ):
            best_by_parent[candidate.parent_id] = (index, candidate)

    ranked = sorted(
        best_by_parent.values(),
        key=lambda item: (-(item[1].rerank_score or 0), item[0]),
    )
    return [candidate for _, candidate in ranked[:top_n]]
