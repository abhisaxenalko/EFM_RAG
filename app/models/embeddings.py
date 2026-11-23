from __future__ import annotations

import hashlib
import math
import os
import threading
from collections.abc import Sequence
from typing import Any

DIMENSIONS = 1024
MODEL_NAME = "BAAI/bge-m3"


class FakeEmbedder:
    """Small deterministic local embedder for CI; it needs no model or API key."""

    def encode(self, texts: Sequence[str], **_: Any) -> list[list[float]]:
        return [_hash_vector(text) for text in texts]


def _hash_vector(text: str) -> list[float]:
    # Expand a stable digest stream into signed coordinates, then normalize.
    raw = bytearray()
    counter = 0
    encoded = text.encode("utf-8")
    while len(raw) < DIMENSIONS * 2:
        raw.extend(hashlib.sha256(encoded + counter.to_bytes(4, "big")).digest())
        counter += 1
    values = [
        (int.from_bytes(raw[i : i + 2], "big") / 32767.5) - 1.0
        for i in range(0, DIMENSIONS * 2, 2)
    ]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


_lock = threading.Lock()
_model: Any | None = None
_model_kind: str | None = None


def _load_model() -> Any:
    if os.getenv("EMBEDDER", "sentence-transformers").strip().lower() == "fake":
        return FakeEmbedder()
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(MODEL_NAME, device=device)


def _get_model() -> Any:
    global _model, _model_kind
    kind = os.getenv("EMBEDDER", "sentence-transformers").strip().lower()
    with _lock:
        if _model is None:
            _model = _load_model()
            _model_kind = kind
        elif kind != _model_kind:
            raise RuntimeError("EMBEDDER cannot change after the model is loaded")
        return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Encode text in batches with normalized vectors of exactly 1024 values."""
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        raise TypeError("texts must be a list of strings")
    if not texts:
        return []

    model = _get_model()
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    if batch_size < 1:
        raise ValueError("EMBEDDING_BATCH_SIZE must be at least 1")
    output: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors = model.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        for vector in vectors:
            values = [float(value) for value in vector]
            if len(values) != DIMENSIONS:
                raise ValueError(
                    f"Embedder returned {len(values)} values; expected {DIMENSIONS}"
                )
            output.append(values)
    return output
