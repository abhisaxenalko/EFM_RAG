from __future__ import annotations

from app.retrieval import reranker
from app.retrieval.hybrid_search import Candidate


def _candidate(chunk_id: int, parent_id: int, text: str) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        parent_id=parent_id,
        doc_name="manual.md",
        page=1,
        child_text=text,
        parent_text=text,
        rrf_score=0.01,
    )


def test_fake_reranker_prefers_answer_and_collapses_parent_duplicates(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RERANKER", "fake")
    monkeypatch.setenv("RERANKER_MAX_PASSAGE_TOKENS", "32")
    reranker._model = None
    reranker._model_kind = None
    candidates = [
        _candidate(1, 10, "The compressor operates normally at rated flow."),
        _candidate(2, 10, "ERR-409B indicates a pressure relief valve failure."),
        _candidate(3, 11, "Inspect the refrigerant charge and condenser fan."),
    ]

    try:
        ranked = reranker.rerank(
            "What does ERR-409B indicate for a pressure relief valve?",
            candidates,
            top_n=5,
        )
        assert [candidate.parent_id for candidate in ranked] == [10, 11]
        assert ranked[0].chunk_id == 2
        assert ranked[0].rerank_score is not None
        assert ranked[0].rerank_score > ranked[1].rerank_score
    finally:
        reranker._model = None
        reranker._model_kind = None
