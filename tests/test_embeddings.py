from __future__ import annotations

import math

from app.models import embeddings


def test_fake_embeddings_are_unit_length_and_model_is_reused(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDER", "fake")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "1")
    embeddings._model = None
    embeddings._model_kind = None
    load_calls = 0
    original_loader = embeddings._load_model

    def counted_loader():
        nonlocal load_calls
        load_calls += 1
        return original_loader()

    monkeypatch.setattr(embeddings, "_load_model", counted_loader)
    try:
        first = embeddings.embed(["ERR-409B", "another phrase"])
        second = embeddings.embed(["ERR-409B"])
        assert len(first) == 2
        assert all(len(vector) == 1024 for vector in [*first, *second])
        assert all(
            math.isclose(
                math.sqrt(sum(x * x for x in vector)), 1.0, abs_tol=1e-6
            )
            for vector in [*first, *second]
        )
        assert first[0] == second[0]
        assert load_calls == 1
    finally:
        embeddings._model = None
        embeddings._model_kind = None
