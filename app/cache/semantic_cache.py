from __future__ import annotations

import hashlib
import json
import math
import secrets
from collections.abc import Sequence
from typing import Any

from app.models.schemas import Answer


class SemanticCache:
    """Cosine-similarity cache isolated by tenant, roles, and corpus version."""

    def __init__(
        self,
        redis_client: Any,
        *,
        cache_sim: float,
        ttl_seconds: int = 3600,
    ) -> None:
        if not 0 <= cache_sim <= 1:
            raise ValueError("CACHE_SIM must be between 0 and 1")
        if ttl_seconds < 1:
            raise ValueError("Cache TTL must be positive")
        self.redis = redis_client
        self.cache_sim = cache_sim
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def role_hash(roles: Sequence[str]) -> str:
        if isinstance(roles, str) or not roles or any(not role for role in roles):
            raise ValueError("roles must be a non-empty sequence of role names")
        canonical = json.dumps(sorted(set(roles)), separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def version_key(tenant_id: str) -> str:
        if not tenant_id:
            raise ValueError("tenant_id must not be empty")
        return f"tenant:{tenant_id}:corpus_version"

    @classmethod
    def prefix(cls, tenant_id: str, roles: Sequence[str], version: int) -> str:
        return (
            f"tenant:{tenant_id}:role:{cls.role_hash(roles)}:"
            f"corpus:{version}:"
        )

    def _current_version(self, tenant_id: str) -> int:
        key = self.version_key(tenant_id)
        self.redis.set(key, "1", nx=True)
        value = self.redis.get(key)
        if value is None:
            raise RuntimeError("Could not read the tenant corpus version")
        return int(value)

    def bump_corpus_version(self, tenant_id: str) -> int:
        """Invalidate all old entries for this tenant without scanning them."""
        key = self.version_key(tenant_id)
        initialized = self.redis.set(key, "1", nx=True)
        return 1 if initialized else int(self.redis.incr(key))

    @staticmethod
    def _validated_embedding(embedding: Sequence[float]) -> list[float]:
        values = [float(value) for value in embedding]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("Embedding must contain finite values")
        return values

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right):
            return -1.0
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return -1.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _escape_glob(value: str) -> str:
        return "".join(f"\\{char}" if char in "\\*?[]" else char for char in value)

    @staticmethod
    def _answer_data(answer: Answer | dict[str, Any]) -> dict[str, Any]:
        model = answer if isinstance(answer, Answer) else Answer.model_validate(answer)
        return model.model_dump(mode="json")

    def store(
        self,
        tenant: str,
        roles: Sequence[str],
        query: str,
        embedding: Sequence[float],
        answer: Answer | dict[str, Any],
    ) -> None:
        if not query:
            raise ValueError("query must not be empty")
        vector = self._validated_embedding(embedding)
        prefix = self.prefix(tenant, roles, self._current_version(tenant))
        key = f"{prefix}entry:{secrets.token_hex(16)}"
        payload = {
            "query": query,
            "embedding": vector,
            "answer": self._answer_data(answer),
        }
        self.redis.set(
            key,
            json.dumps(payload, separators=(",", ":")),
            ex=self.ttl_seconds,
        )

    def lookup(
        self,
        tenant: str,
        roles: Sequence[str],
        query_embedding: Sequence[float],
    ) -> Answer | None:
        vector = self._validated_embedding(query_embedding)
        prefix = self.prefix(tenant, roles, self._current_version(tenant))
        pattern = f"{self._escape_glob(prefix)}entry:*"
        best_answer: Answer | None = None
        best_similarity = self.cache_sim
        for key in self.redis.scan_iter(match=pattern):
            raw = self.redis.get(key)
            if raw is None:
                continue
            try:
                payload = json.loads(raw)
                similarity = self._cosine_similarity(vector, payload["embedding"])
                if similarity >= best_similarity:
                    best_similarity = similarity
                    best_answer = Answer.model_validate(payload["answer"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return best_answer
