from __future__ import annotations

import secrets

import pytest
import redis

from app.cache.semantic_cache import SemanticCache
from app.config import get_settings
from app.models.schemas import Answer, Citation


@pytest.mark.isolation
def test_semantic_cache_is_scoped_by_tenant_role_and_corpus_version() -> None:
    settings = get_settings()
    tenant = f"cachetest{secrets.token_hex(6)}"
    vector = [1.0] + [0.0] * 1023
    answer = Answer(
        text="ERR-409B indicates low flow. [Chiller Manual, Page 7]",
        citations=[Citation(doc_name="Chiller Manual", page=7)],
    )
    with redis.Redis.from_url(settings.redis_url, decode_responses=True) as client:
        cache = SemanticCache(client, cache_sim=settings.cache_sim, ttl_seconds=120)
        try:
            cache.store(
                tenant, ["engineer", "all"], "What is ERR-409B?", vector, answer
            )
            assert cache.lookup(tenant, ["all", "engineer"], vector) == answer
            assert cache.lookup(f"{tenant}other", ["all", "engineer"], vector) is None
            assert cache.lookup(tenant, ["all"], vector) is None

            cache.bump_corpus_version(tenant)
            assert cache.lookup(tenant, ["all", "engineer"], vector) is None
        finally:
            for key in client.scan_iter(match=f"tenant:{tenant}:*"):
                client.delete(key)
